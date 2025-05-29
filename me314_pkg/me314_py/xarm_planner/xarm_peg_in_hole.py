#!/usr/bin/env python3
"""
red_cylinder_to_blue_square.py – example ROS 2 node

★ TASK OVERVIEW ★
1. 粗略识别红色‑圆柱 & 蓝色‑正方形的投影中心点 (u, v)
2. 移动机械臂到红色圆柱上方 (基于粗略中心)
3. 再次取图 → 精细识别红色圆环中心 (霍夫圆检测)
4. 二次对准：移动到精细红色圆心上方
5. 向下抓取红色圆柱
6. 移动到蓝色正方形粗略中心上方
7. 再次取图 → 精细识别蓝色正方形中心 (轮廓拟合)
8. 二次对准：移动到精细蓝色正中心上方，插入圆柱
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import tf_transformations
from tf2_ros import Buffer, TransformListener

from me314_msgs.msg import CommandQueue, CommandWrapper


def to_pose_array(xyz, q=(1.0, 0.0, 0.0, 0.0)):
    """Helper to build [x,y,z,qx,qy,qz,qw] list."""
    return [xyz[0], xyz[1], xyz[2], q[0], q[1], q[2], q[3]]


class PickAndInsert(Node):
    def __init__(self):
        super().__init__('pick_insert_node')

        # ------- Image / Depth -------
        self.bridge = CvBridge()
        qos = 10
        self.image_sub = self.create_subscription(Image, '/color/image_raw', self.image_cb, qos)
        self.depth_sub = self.create_subscription(Image, '/aligned_depth_to_color/image_raw', self.depth_cb, qos)
        self.info_sub = self.create_subscription(CameraInfo, '/color/camera_info', self.info_cb, qos)
        self.latest_img, self.latest_depth, self.K = None, None, None

        # ------- TF -------
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ------- Command Queue -------
        self.pub_queue = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', qos)

        # ------- Misc -------
        self.base_frame = 'link_base'
        self.cam_frame = 'camera_link'
        self.safe_height = 0.25  # m – 安全高度 (可按实际修改)
        self.approach_down = 0.01  # m – 插入/抓取向下距离

    # ---------- Sub callbacks ----------
    def image_cb(self, msg):
        self.latest_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

    def depth_cb(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')

    def info_cb(self, msg):
        if self.K is None:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])  # fx, fy, cx, cy
            self.get_logger().info('Camera intrinsics ready.')

    # ---------- Utility ----------
    def wait_for_io(self, timeout=5.0):
        """Block until image/depth/intrinsics are ready."""
        import time
        start = time.time()
        while rclpy.ok() and (self.latest_img is None or self.latest_depth is None or self.K is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            if time.time() - start > timeout:
                raise RuntimeError('Timeout waiting for camera topics')

    def depth_to_xyz(self, u: int, v: int):
        fx, fy, cx, cy = self.K
        d = int(self.latest_depth[v, u])
        if d == 0:
            return None
        z = d / 1000.0
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        return [z, -x, -y]  # camera → ROS convention (x forward)

    def cam_to_base(self, xyz_cam):
        if xyz_cam is None:
            return None
        try:
            trans = self.tf_buffer.lookup_transform(self.base_frame, self.cam_frame, rclpy.time.Time(), rclpy.duration.Duration(seconds=1.0))
        except Exception as e:
            self.get_logger().error(f'TF error: {e}')
            return None
        t = trans.transform.translation
        q = trans.transform.rotation
        T = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])
        T[:3, 3] = [t.x, t.y, t.z]
        pt = np.array(list(xyz_cam) + [1.0])
        xyz = T @ pt
        return xyz[:3].tolist()

    def publish_queue(self, wrappers):
        msg = CommandQueue()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.commands.extend(wrappers)
        self.pub_queue.publish(msg)

    def make_pose_cmd(self, pose_array):
        w = CommandWrapper()
        w.command_type = 'pose'
        w.pose_command.x, w.pose_command.y, w.pose_command.z = pose_array[:3]
        w.pose_command.qx, w.pose_command.qy, w.pose_command.qz, w.pose_command.qw = pose_array[3:]
        return w

    def make_gripper_cmd(self, pos: float):
        w = CommandWrapper()
        w.command_type = 'gripper'
        w.gripper_command.gripper_position = pos
        return w

    # ---------- Vision ----------
    def coarse_centers(self):
        """Return (red_uv, blue_uv) or (None, None) if failure."""
        img = self.latest_img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        red_mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        blue_mask = cv2.inRange(hsv, (100, 80, 60), (130, 255, 255))
        def center(mask):
            c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if c:
                cnt = max(c, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    return (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            return None
        return center(red_mask), center(blue_mask)

    def refine_red_circle(self):
        """Use HoughCircles on red mask to get precise circle center."""
        hsv = cv2.cvtColor(self.latest_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)) | cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        blur = cv2.GaussianBlur(mask, (9, 9), 2)
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20, param1=100, param2=15, minRadius=5, maxRadius=100)
        if circles is not None:
            x, y, _ = circles[0][0]
            return int(x), int(y)
        return None

    def refine_blue_square(self):
        hsv = cv2.cvtColor(self.latest_img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (100, 80, 60), (130, 255, 255))
        c, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not c:
            return None
        cnt = max(c, key=cv2.contourArea)
        rect = cv2.minAreaRect(cnt)
        return (int(rect[0][0]), int(rect[0][1]))

    # ---------- Motion pipeline ----------
    def run_task(self):
        self.wait_for_io()

        # 1 粗略识别红蓝中心
        red_uv, blue_uv = self.coarse_centers()
        if red_uv is None or blue_uv is None:
            self.get_logger().error('Failed coarse detection')
            return
        red_xyz = self.cam_to_base(self.depth_to_xyz(*red_uv))
        blue_xyz = self.cam_to_base(self.depth_to_xyz(*blue_uv))
        if red_xyz is None or blue_xyz is None:
            self.get_logger().error('Depth/TF failed')
            return

        # 2 移动到红色上方 (安全高度)
        above_red = red_xyz.copy(); above_red[2] = self.safe_height
        self.publish_queue([self.make_pose_cmd(to_pose_array(above_red))])

        # 3 精细识别圆心
        rclpy.spin_once(self, timeout_sec=1.0)  # 等新帧
        fine_uv = self.refine_red_circle()
        if fine_uv:
            fine_xyz = self.cam_to_base(self.depth_to_xyz(*fine_uv))
            if fine_xyz:
                above_red = fine_xyz.copy(); above_red[2] = self.safe_height
                red_xyz = fine_xyz
                self.publish_queue([self.make_pose_cmd(to_pose_array(above_red))])

        # 4 下探抓取
        grasp_pose = red_xyz.copy(); grasp_pose[2] -= self.approach_down
        wrappers = [
            self.make_pose_cmd(to_pose_array(grasp_pose)),
            self.make_gripper_cmd(1.0),  # close
            self.make_pose_cmd(to_pose_array(above_red))
        ]
        self.publish_queue(wrappers)

        # 5 上移后去往蓝色粗略中心
        above_blue = blue_xyz.copy(); above_blue[2] = self.safe_height
        self.publish_queue([self.make_pose_cmd(to_pose_array(above_blue))])

        # 6 精细蓝色中心 (插孔)
        rclpy.spin_once(self, timeout_sec=1.0)
        fine_uv = self.refine_blue_square()
        if fine_uv:
            fine_xyz = self.cam_to_base(self.depth_to_xyz(*fine_uv))
            if fine_xyz:
                above_blue = fine_xyz.copy(); above_blue[2] = self.safe_height
                blue_xyz = fine_xyz
                self.publish_queue([self.make_pose_cmd(to_pose_array(above_blue))])

        # 7 插入并松爪
        insert_pose = blue_xyz.copy(); insert_pose[2] -= self.approach_down
        wrappers = [
            self.make_pose_cmd(to_pose_array(insert_pose)),
            self.make_gripper_cmd(0.0),  # open
            self.make_pose_cmd(to_pose_array(above_blue))
        ]
        self.publish_queue(wrappers)
        self.get_logger().info('Task completed')


def main(args=None):
    rclpy.init(args=args)
    node = PickAndInsert()
    try:
        node.run_task()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
