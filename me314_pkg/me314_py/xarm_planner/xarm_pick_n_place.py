#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import pyrealsense2 as rs
from sensor_msgs.msg import CameraInfo
from tf2_ros import Buffer, TransformListener
import tf_transformations


# Import the command queue message types from the reference code
from me314_msgs.msg import CommandQueue, CommandWrapper


class Example(Node):
    def __init__(self):
        super().__init__('example_node')

        # Add new camera node
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, '/color/image_raw', self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, '/aligned_depth_to_color/image_raw', self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(CameraInfo, '/color/camera_info', self.camera_info_callback, 10)

        self.latest_image = None
        self.latest_depth = None
        self.intrinsics = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Replace the direct publishers with the command queue publisher
        self.command_queue_pub = self.create_publisher(CommandQueue, '/me314_xarm_command_queue', 10)
        
        # Subscribe to current arm pose and gripper position for status tracking (optional)
        self.current_arm_pose = None
        self.pose_status_sub = self.create_subscription(Pose, '/me314_xarm_current_pose', self.arm_pose_callback, 10)
        
        self.current_gripper_position = None
        self.gripper_status_sub = self.create_subscription(Float64, '/me314_xarm_gripper_position', self.gripper_position_callback, 10)

    def arm_pose_callback(self, msg: Pose):
        self.current_arm_pose = msg
    
    def image_callback(self, msg):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def depth_callback(self, msg):
        self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')  # 16UC1

    def camera_info_callback(self, msg: CameraInfo):
        if self.intrinsics is None:
            self.intrinsics = {
                "fx": msg.k[0],
                "fy": msg.k[4],
                "ppx": msg.k[2],
                "ppy": msg.k[5]
            }
            self.get_logger().info("Camera intrinsics received.")


    def gripper_position_callback(self, msg: Float64):
        self.current_gripper_position = msg.data

    def publish_pose(self, pose_array: list):
        """
        Publishes a pose command to the command queue using an array format.
        pose_array format: [x, y, z, qx, qy, qz, qw]
        """
        # Create a CommandQueue message containing a single pose command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the pose command
        wrapper = CommandWrapper()
        wrapper.command_type = "pose"
        
        # Populate the pose_command with the values from the pose_array
        wrapper.pose_command.x = pose_array[0]
        wrapper.pose_command.y = pose_array[1]
        wrapper.pose_command.z = pose_array[2]
        wrapper.pose_command.qx = pose_array[3]
        wrapper.pose_command.qy = pose_array[4]
        wrapper.pose_command.qz = pose_array[5]
        wrapper.pose_command.qw = pose_array[6]
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published Pose to command queue:\n"
                               f"  position=({pose_array[0]}, {pose_array[1]}, {pose_array[2]})\n"
                               f"  orientation=({pose_array[3]}, {pose_array[4]}, "
                               f"{pose_array[5]}, {pose_array[6]})")

    def publish_gripper_position(self, gripper_pos: float):
        """
        Publishes a gripper command to the command queue.
        For example:
          0.0 is "fully open"
          1.0 is "closed"
        """
        # Create a CommandQueue message containing a single gripper command
        queue_msg = CommandQueue()
        queue_msg.header.stamp = self.get_clock().now().to_msg()
        
        # Create a CommandWrapper for the gripper command
        wrapper = CommandWrapper()
        wrapper.command_type = "gripper"
        wrapper.gripper_command.gripper_position = gripper_pos
        
        # Add the command to the queue and publish
        queue_msg.commands.append(wrapper)
        self.command_queue_pub.publish(queue_msg)
        
        self.get_logger().info(f"Published gripper command to queue: {gripper_pos:.2f}")

    def get_3d_point_from_uv(self, u, v):
        if self.latest_depth is None or self.intrinsics is None:
            return None

        
        depth_raw = self.latest_depth[v, u]  # 注意顺序：行v，列u
        if depth_raw == 0:
            self.get_logger().info(f"FUCKED")
            return None
        # depth_raw = 500
        depth_meters = depth_raw / 1000.0  # 假设是以 mm 存储的 16UC1

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        ppx = self.intrinsics["ppx"]
        ppy = self.intrinsics["ppy"]

        x = (u - ppx) * depth_meters / fx
        y = (v - ppy) * depth_meters / fy
        z = depth_meters

        return [z,-x,-y]


    def debug_visualization(self, img, red_mask, green_mask, red_uv=None, green_uv=None):
        debug_img = img.copy()

        # 显示红色区域 mask
        red_mask_bgr = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
        red_mask_bgr[:, :, 1:] = 0  # 保留 R 通道

        # 显示绿色区域 mask
        green_mask_bgr = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
        green_mask_bgr[:, :, [0, 2]] = 0  # 保留 G 通道

        overlay = cv2.addWeighted(red_mask_bgr, 0.5, green_mask_bgr, 0.5, 0)
        debug_img = cv2.addWeighted(debug_img, 0.7, overlay, 0.3, 0)

        # Draw Red Dot
        if red_uv:
            cv2.circle(debug_img, red_uv, 5, (0, 0, 255), -1)
            cv2.putText(debug_img, "Red", (red_uv[0]+5, red_uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Draw Green Dot
        if green_uv:
            cv2.circle(debug_img, green_uv, 5, (0, 255, 0), -1)
            cv2.putText(debug_img, "Green", (green_uv[0]+5, green_uv[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow("Debug View", debug_img)
        cv2.waitKey(1)  
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord('p'):
                break

    def transform_camera_to_eef(self, point_in_camera):
        """
        
        """
        try:
            import time
            import tf2_ros

            try:
                for _ in range(50): 
                    rclpy.spin_once(self, timeout_sec=0.1)
                    if self.tf_buffer.can_transform('link_base', 'camera_link', rclpy.time.Time()):
                        trans = self.tf_buffer.lookup_transform('link_base', 'camera_link', rclpy.time.Time())
                        break
                    else:
                        self.get_logger().warn("Waiting for transform from camera_link to link_eef...")
                else:
                    self.get_logger().error("TF transform still unavailable after timeout.")
                    return
            except Exception as e:
                self.get_logger().error(f"[TF ERROR] {str(e)}")
                return

            # trans = self.tf_buffer.lookup_transform(
            #     target_frame='link_base',
            #     source_frame='camera_depth_frame',
            #     time=rclpy.time.Time(),
            #     timeout=rclpy.duration.Duration(seconds=1.0)
            # )
            # 
            t = trans.transform.translation
            translation = np.array([t.x, t.y, t.z])
            
            # Rotate
            q = trans.transform.rotation
            rotation = tf_transformations.quaternion_matrix([q.x, q.y, q.z, q.w])[:3, :3]

            point_camera = np.array(point_in_camera).reshape((3, 1))
            point_eef = rotation @ point_camera + translation.reshape((3, 1))
            return point_eef.flatten().tolist()

        except Exception as e:
            self.get_logger().error(f"[TF ERROR] {e}")
            return None

            


    def get_centers_from_image(self):
        if self.latest_image is None or self.latest_depth is None or self.intrinsics is None:
            self.get_logger().warn("Image or depth or intrinsics not ready.")
            return None, None

        img = self.latest_image.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # ======== Red HSV ========
        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower1, red_upper1) | cv2.inRange(hsv, red_lower2, red_upper2)

        # ======== Green HSV ========
        green_lower = np.array([40, 60, 60])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        
        def find_center_pixel(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                M = cv2.moments(cnt)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    return (cx, cy)
            return None


        red_uv = find_center_pixel(red_mask)
        green_uv = find_center_pixel(green_mask)

        red_xyz = self.get_3d_point_from_uv(*red_uv) if red_uv else None
        green_xyz = self.get_3d_point_from_uv(*green_uv) if green_uv else None

        self.debug_visualization(img, red_mask, green_mask, red_uv, green_uv)

        return red_xyz, green_xyz
    
    


def main(args=None):
    rclpy.init(args=args)
    node = Example()

    # Define poses using the array format [x, y, z, qx, qy, qz, qw]
    p0 = [0.3408, 0.0021, 0.3029, 1.0, 0.0, 0.0, 0.0]
    p1 = [p0[0], p0[1], 0.1, 1.0, 0.0, 0.0, 0.0]

    import time

    timeout_sec = 5.0
    start_time = time.time()
    while (node.latest_image is None or node.latest_depth is None or node.intrinsics is None):
        rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start_time > timeout_sec:
            node.get_logger().error("Timeout waiting for image/depth/intrinsics.")
            return


    red_xyz_cam, green_xyz_cam = node.get_centers_from_image()

    red_xyz = node.transform_camera_to_eef(red_xyz_cam) if red_xyz_cam else None
    green_xyz = node.transform_camera_to_eef(green_xyz_cam) if green_xyz_cam else None

    if red_xyz is None or green_xyz is None:
        node.get_logger().error("TF transform failed or invalid detection")
        return

    print(red_xyz)
    red_xyz[2] -= 0.01
    green_xyz[2] += 0.01

    # 1. Move above red block
    node.publish_pose([*red_xyz, 1.0, 0.0, 0.0, 0.0])  # 填充 orientation

    # 2. grab
    node.publish_gripper_position(1.0)

    # 3. Go back
    node.publish_pose(p0)

    # 4. Move to Green
    node.publish_pose([*green_xyz, 1.0, 0.0, 0.0, 0.0])

    # 4. Let Go
    node.publish_gripper_position(0.0)

    # 5. Go Back
    node.publish_pose(p0)


    # # don't car below
    # poses = [p0, p1]

    # # Let's first open the gripper (0.0 to 1.0, where 0.0 is fully open and 1.0 is fully closed)
    # node.get_logger().info("Opening gripper...")
    # node.publish_gripper_position(0.0)

    # # Move the arm to each pose
    # for i, pose in enumerate(poses):
    #     node.get_logger().info(f"Publishing Pose {i+1}...")
    #     node.publish_pose(pose)

    # # Now close the gripper.
    # node.get_logger().info("Closing gripper...")
    # node.publish_gripper_position(1.0)

    # node.get_logger().info("All actions done. Shutting down.")

    # node.destroy_node()
    # rclpy.shutdown()


if __name__ == '__main__':
    main()