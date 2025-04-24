#!/usr/bin/env python3

"""
XArm commander node that implements a command queue system.
Commands are received through a dedicated topic and executed sequentially.
The node only moves to the next command after the current one is fully completed.
"""

import rclpy
import math
import threading
import time
from collections import deque
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Pose, PoseStamped, WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String, Bool
from xarm_msgs.srv import SetInt16, Call

# Custom message type for command queue (which contains an array of CommandWrapper messages)
from me314_msgs.msg import CommandQueue

from moveit_msgs.srv import (GetPositionIK, GetCartesianPath, GetMotionPlan, GetPositionFK)
from moveit_msgs.msg import (
    RobotTrajectory,
    MotionPlanRequest,
    Constraints,
    JointConstraint,
    RobotState,
    PlanningScene,
    LinkPadding
)
from moveit_msgs.srv import GetPlanningScene, ApplyPlanningScene
from shape_msgs.msg import SolidPrimitive
from control_msgs.action import FollowJointTrajectory
from moveit_msgs.msg import CollisionObject
from moveit_msgs.action import ExecuteTrajectory
from rclpy import spin_until_future_complete


class ME314_XArm_Queue_Commander(Node):
    def __init__(self):
        """
        Initialize the ME314_XArm_Queue_Commander node, setting up the command queue,
        clients, publishers, subscribers, and initial robot state.
        """
        super().__init__('ME314_XArm_Queue_Commander_Node')
        
        # Define ANSI color codes for terminal output
        self.GREEN = "\033[32m"
        self.RED = "\033[31m"
        self.RESET = "\033[0m"
        self.BOLD = "\033[1m"

        ####################################################################
        # QUEUE SYSTEM
        ####################################################################
        self.command_queue = deque()
        self.queue_lock = threading.Lock()
        self.is_executing = False

        ####################################################################
        # CLASS ATTRIBUTES
        ####################################################################
        self.declare_parameter('use_sim', False)
        self.use_sim = self.get_parameter('use_sim').value
        self.log_info(f"Running with use_sim={self.use_sim}")

        self.declare_parameter('ft_threshold', 4.0)
        self.ft_threshold = self.get_parameter('ft_threshold').value
        self.last_ext_force_mag = 0.0
        self.controller_goal_handle = None

        self.current_gripper_position = 0.0
        self.home_joints_deg = [0.2, -67.2, -0.2, 24.2, 0.4, 91.4, 0.3]
        self.home_joints_rad = [math.radians(angle) for angle in self.home_joints_deg]
        self.joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6', 'joint7']
        self.curr_joint_positions = [None] * len(self.joint_names)

        self.gripper_group_name = "xarm_gripper"  
        self.gripper_joint_names = ["drive_joint"]
        self.initialization_complete = False
        self.command_failed = False

        # Define the workspace bounds in millimeters
        self.x_bounds = (150, 500)   
        self.y_bounds = (-300, 300) 
        if self.use_sim:
            self.z_bounds = (0, 400) 
        else:
            self.z_bounds = (35, 400)

        # Initialize planning scene
        self.planning_scene = PlanningScene()
        self.planning_scene.is_diff = True
        self.planning_scene.robot_state.is_diff = True    

        ####################################################################
        # CLIENTS
        ####################################################################
        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')
        self.compute_ik_client = self.create_client(GetPositionIK, '/compute_ik')
        self.plan_path_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.execute_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')
        self.arm_ctrl_client = ActionClient(self, FollowJointTrajectory, '/xarm7_traj_controller/follow_joint_trajectory')
        self.get_plan_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        self.apply_plan_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        
        if not self.use_sim:
            self.enable_FT_client = self.create_client(SetInt16, '/xarm/ft_sensor_enable')
            self.zero_FT_client = self.create_client(Call, '/xarm/ft_sensor_set_zero')
            self.set_arm_state_client = self.create_client(SetInt16, '/xarm/set_state')

        self.wait_for_all_services_and_action()

        ####################################################################
        # SUBSCRIBERS
        ####################################################################
        self.queue_cmd_sub = self.create_subscription(CommandQueue, '/me314_xarm_command_queue', self.command_queue_callback, 10)
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.ft_sub = self.create_subscription(WrenchStamped, '/xarm/uf_ftsensor_ext_states', self.ft_sensor_callback, 50)

        ####################################################################
        # PUBLISHERS
        ####################################################################
        self.current_pose_pub = self.create_publisher(Pose, '/me314_xarm_current_pose', 10)
        self.curr_joint_position_deg_pub = self.create_publisher(JointState, '/me314_xarm_current_joint_positions_deg', 10)
        self.gripper_position_pub = self.create_publisher(Float64, '/me314_xarm_gripper_position', 10)

        self.queue_size_pub = self.create_publisher(Float64, '/me314_xarm_queue_size', 10)
        self.is_executing_pub = self.create_publisher(Bool, '/me314_xarm_is_executing', 10)
        self.current_command_pub = self.create_publisher(String, '/me314_xarm_current_command', 10)
        self.rejected_command_pub = self.create_publisher(String, '/me314_xarm_rejected_command', 10)
        self.collision_pub = self.create_publisher(Bool, '/me314_xarm_collision', 10)

        # Timers for status publishing and processing the queue
        self.timer_pose = self.create_timer(0.1, self.publish_current_pose)
        self.timer_gripper = self.create_timer(0.1, self.publish_gripper_position)
        self.timer_joint_positions = self.create_timer(0.1, self.publish_current_joint_positions)
        self.timer_queue_status = self.create_timer(0.1, self.publish_queue_status)
        self.timer_queue_processor = self.create_timer(0.1, self.process_command_queue)
        self.ft_timer = self.create_timer(0.01, self.check_ft_threshold)

        ####################################################################
        # INITIALIZATION
        ####################################################################
        self.log_info("Moving to home position (joint-based).")
        self.plan_execute_joint_target_async(self.home_joints_rad, callback=self.home_move_done_callback)
        self.log_info("XArm queue commander node is starting initialization...")

        # Log the workspace bounds
        self.log_info(f"Workspace bounds initialized: X={self.x_bounds}mm, Y={self.y_bounds}mm, Z={self.z_bounds}mm")

        # Store collision objects here so we can reapply them in each planning scene update
        self.boundary_collision_objects = []

        self.publish_collision_status(False)

    ####################################################################
    # INITIALIZATION METHODS
    ####################################################################
    def init_ft_sensor(self):
        """
        Initialize the force/torque sensor by enabling it, zeroing it, and clearing any errors.
        """
        self.call_service(self.enable_FT_client, SetInt16.Request(data=1), "FT sensor enabled successfully!", "FT enable")
        self.call_service(self.zero_FT_client, Call.Request(), "FT sensor zeroed successfully!", "FT zero")
        self.call_service(self.set_arm_state_client, SetInt16.Request(data=0), "Robot state set to READY successfully 🎉", "Set state")   

    def call_service(self, client, request, success_msg, name, pre_delay=0):
        """
        Call a service and handle the response.
        """
        if pre_delay > 0:
            time.sleep(pre_delay)
        
        if not client.wait_for_service(5.0):
            self.log_error(f"{name} service unavailable")
            return
            
        fut = client.call_async(request)
        spin_until_future_complete(self, fut, timeout_sec=2.0)
        
        if not fut.done() or fut.result() is None:
            self.log_error(f"{name} call timed out")
        elif fut.result().ret != 0:
            self.log_error(f"{name} failed: {fut.result().message}")
        else:
            self.log_info(success_msg)

    def gripper_init_callback(self, success: Bool):
        """
        Callback invoked after the gripper initialization command is completed.
        """
        if success:
            self.log_info("Initialization complete. Ready to process commands.")
            self.initialization_complete = True
        else:
            self.log_warn("Initialization failed. Gripper open command failed.")
            self.initialization_complete = True

    def wait_for_all_services_and_action(self):
        """Wait for all services and action servers to become available."""
        services = [
            (self.cartesian_path_client, 'compute_cartesian_path'),
            (self.compute_ik_client, 'compute_ik'),
            (self.plan_path_client, 'plan_kinematic_path'),
            (self.fk_client, 'compute_fk'),
            (self.get_plan_scene_client, 'get_planning_scene'),
            (self.apply_plan_scene_client, 'apply_planning_scene')
        ]
        
        if not self.use_sim:
            services.extend([
                (self.enable_FT_client, 'xarm/ft_sensor_enable'),
                (self.zero_FT_client, 'xarm/ft_sensor_set_zero'),
                (self.set_arm_state_client, 'xarm/set_state')
            ])
        
        for client, name in services:
            while not client.wait_for_service(timeout_sec=1.0):
                self.log_info(f'Waiting for {name} service...')
                
        # Action clients        
        action_clients = [
            (self.execute_client, 'execute_trajectory'),
            (self.arm_ctrl_client, 'follow_joint_trajectory')
        ]
        
        # Wait for action clients
        for client, name in action_clients:
            while not client.wait_for_server(timeout_sec=1.0):
                self.log_info(f'Waiting for {name} action server...')
            
        self.log_info('All services and action servers are available!')

    def home_move_done_callback(self, success: bool):
        """
        Callback for when the home position move is completed.
        Sets up the planning scene if the move succeeded.
        """
        if not success:
            self.log_warn("Failed to move to home position (joint-based).")
        else:
            self.log_info("Home position move completed successfully (joint-based).")

            # Open the gripper fully after home position is reached
            self.log_info("Opening gripper fully.")
            self.plan_execute_gripper_async(0.0, callback=self.gripper_init_callback)

            # Setup planning scene with workspace bounds now that the robot is at home
            self.setup_planning_scene()

            # Initialize the force/torque sensor
            if not self.use_sim:
                self.log_info("Initializing FT sensor.")
                self.init_ft_sensor()

    ####################################################################
    # PLANNING SCENE METHODS
    ####################################################################
    def setup_planning_scene(self):
        """
        Setup the planning scene with box constraints to create an open-top box
        that restricts the robot's end-effector movement.
        """
        self.log_info("Setting up planning scene with workspace boundaries...")
        
        # Convert mm bounds to meters for planning scene
        x_l, x_u = self.x_bounds[0] / 1000, self.x_bounds[1] / 1000
        y_l, y_u = self.y_bounds[0] / 1000, self.y_bounds[1] / 1000
        z_l = self.z_bounds[0] / 1000

        x_c, y_c = (x_l + x_u) / 2, (y_l + y_u) / 2
        x_w, y_w = (x_u - x_l) + 0.1, (y_u - y_l) + 0.1 
        
        # Create collision objects for the 5 planes of the box (open top)
        collision_objects = []
        
        # (From robot's perspective))
        collision_objects.append(self.create_box_collision_object("front_wall", 0.001, y_w, 0.6, x_u + 0.005, y_c, 0.25 + z_l))
        collision_objects.append(self.create_box_collision_object("back_wall", 0.001, y_w, 0.6, x_l - 0.5, y_c, 0.25 + z_l))
        collision_objects.append(self.create_box_collision_object("right_wall", x_w, 0.001, 0.6, x_c, y_l - 0.05, 0.25 + z_l))
        collision_objects.append(self.create_box_collision_object("left_wall", x_w, 0.001, 0.6, x_c, y_u + 0.025, 0.25 + z_l))
        # collision_objects.append(self.create_box_collision_object("floor", x_w, y_w, 0.001, x_c, y_c, z_l - 0.002))
        
        # Store our boundary collision objects in class attribute
        self.boundary_collision_objects = collision_objects
        
        # Add all collision objects to the planning scene message
        self.planning_scene.world.collision_objects = self.boundary_collision_objects
        
        # Relevant robot links to add padding for self-collision
        acm_links = ["link_tcp", "link_eef", "link_base", "left_inner_knuckle", "left_outer_knuckle",
            "right_inner_knuckle", "right_outer_knuckle", "right_finger", "left_finger", "xarm_gripper_base_link", "ft_sensor_link", "link7"]
        
        # Add padding to all robot links
        self.planning_scene.link_padding = [LinkPadding(link_name=name, padding=0.002) for name in acm_links]
        self.log_info("Added link padding to robot links for self-collision avoidance.")
       
        self.apply_planning_scene(self.planning_scene)
        self.log_info("Planning scene with workspace boundaries has been set up.")

    def create_box_collision_object(self, name, x_dim, y_dim, z_dim, x_pos, y_pos, z_pos):
        """
        Create a box collision object for the planning scene.
        """
        collision_object = CollisionObject()
        collision_object.header.frame_id = "link_base"
        collision_object.id = name
        
        # Define the box as a solid primitive
        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [x_dim, y_dim, z_dim]
        
        # Set the pose of the box
        pose = PoseStamped()
        pose.header.frame_id = "link_base"
        pose.pose.position.x = x_pos
        pose.pose.position.y = y_pos
        pose.pose.position.z = z_pos
        pose.pose.orientation.w = 1.0
        
        # Add the primitive to the collision object
        collision_object.primitives.append(box)
        collision_object.primitive_poses.append(pose.pose)
        collision_object.operation = CollisionObject.ADD
        
        return collision_object

    def apply_planning_scene(self, planning_scene):
        """
        Apply the planning scene through the proper service and publish updates.
        """
        # Make a copy to avoid modifying the original
        scene_to_publish = PlanningScene()
        scene_to_publish.is_diff = True
        
        # Only include what's changed
        if hasattr(planning_scene, 'world') and planning_scene.world.collision_objects:
            scene_to_publish.world.collision_objects = planning_scene.world.collision_objects
        
        if hasattr(planning_scene, 'robot_state') and planning_scene.robot_state.joint_state.name:
            scene_to_publish.robot_state = planning_scene.robot_state
        
        if hasattr(planning_scene, 'link_padding') and planning_scene.link_padding:
            scene_to_publish.link_padding = planning_scene.link_padding
        
        # Use the service for applying the scene
        req = ApplyPlanningScene.Request()
        req.scene = scene_to_publish
        
        future = self.apply_plan_scene_client.call_async(req)
        future.add_done_callback(self.planning_scene_applied_callback)

    def planning_scene_applied_callback(self, future):
        """
        Callback for when the planning scene has been applied.
        """
        try:
            response = future.result()
        except Exception as e:
            self.log_error(f"Error applying planning scene: {e}")

    ####################################################################
    # QUEUE METHODS
    ####################################################################
    def command_queue_callback(self, msg: CommandQueue):
        """
        Process the incoming CommandQueue message and add each command to the internal queue.
        """
        with self.queue_lock:
            for command in msg.commands:
                if command.command_type == "pose":
                    pose_cmd = command.pose_command
                    pose = Pose()
                    pose.position.x = pose_cmd.x
                    pose.position.y = pose_cmd.y
                    pose.position.z = pose_cmd.z
                    pose.orientation.x = pose_cmd.qx
                    pose.orientation.y = pose_cmd.qy
                    pose.orientation.z = pose_cmd.qz
                    pose.orientation.w = pose_cmd.qw
                    self.command_queue.append(("pose", pose))

                elif command.command_type == "gripper":
                    gripper_cmd = command.gripper_command
                    self.command_queue.append(("gripper", gripper_cmd.gripper_position))
                    self.log_info(f"Queued gripper command: {gripper_cmd.gripper_position}")

                elif command.command_type == "joint":
                    j = command.joint_command
                    joint_pos = [j.joint1, j.joint2, j.joint3, j.joint4, j.joint5, j.joint6, j.joint7]
                    self.command_queue.append(("joint", joint_pos))
                    self.log_info(f"Queued joint command: {[math.degrees(j) for j in joint_pos]}")

                elif command.command_type == "home":
                    self.command_queue.append(("home", None))
                    self.log_info("Queued home command")
                else:
                    self.log_warn(f"Unknown command type: {command.command_type}")

        self.log_info(f"Added {len(msg.commands)} commands. Current queue size: {len(self.command_queue)}")

    def process_command_queue(self):
        """
        Check and process the next command in the queue if the system is not currently executing a command.
        """
        if not self.initialization_complete or self.command_failed:
            return

        with self.queue_lock:
            if not self.command_queue or self.is_executing:
                return

            cmd_type, cmd = self.command_queue[0]
            self.is_executing = True

        if cmd_type == "pose":
            self.log_info(f"Executing pose command: [{cmd.position.x:.3f}, {cmd.position.y:.3f}, {cmd.position.z:.3f}]")
            self.compute_ik_and_execute_joint_async(cmd, callback=self.command_execution_complete)

        elif cmd_type == "joint":
            self.log_info(f"Executing joint command: {[math.degrees(j) for j in cmd]}")
            self.plan_execute_joint_target_async(cmd, callback=self.command_execution_complete)

        elif cmd_type == "gripper":
            self.log_info(f"Executing gripper command: {cmd}")
            self.plan_execute_gripper_async(cmd, callback=self.command_execution_complete)

    def command_execution_complete(self, success: bool):
        """
        Callback function to handle completion of a command execution. Remove the command from the queue if successful.
        """
        with self.queue_lock:
            if success:
                self.log_info("Command executed successfully")
                if self.command_queue:
                    self.command_queue.popleft()
            else:
                self.log_warn("Command execution failed")
                self.command_failed = True

            self.is_executing = False

    def publish_queue_status(self):
        """
        Publish the current status of the command queue, execution state, and information about the active command.
        """
        with self.queue_lock:
            queue_size = len(self.command_queue)
            is_executing = self.is_executing
            cmd = ""
            if queue_size > 0:
                cmd_type, data = self.command_queue[0]
                if cmd_type == "pose":
                    cmd = (f"Pose: [{data.position.x:.3f}, {data.position.y:.3f}, {data.position.z:.3f}]")
                elif cmd_type == "gripper":
                    cmd = f"Gripper: {data}"
                elif cmd_type == "joint":
                    cmd = f"Joint: {[math.degrees(j) for j in data]}"

        queue_size_msg = Float64()
        queue_size_msg.data = float(queue_size)
        self.queue_size_pub.publish(queue_size_msg)

        is_executing_msg = Bool()
        is_executing_msg.data = is_executing
        self.is_executing_pub.publish(is_executing_msg)

        current_command_msg = String()
        current_command_msg.data = cmd
        self.current_command_pub.publish(current_command_msg)

    ####################################################################
    # CORE MOVEMENT METHODS
    ####################################################################
    def joint_state_callback(self, msg: JointState):
        """
        Callback for processing new joint state messages.
        Updates the current joint positions and gripper state, and reapplies the planning scene.
        """
        for name, position in zip(msg.name, msg.position):
            if name in self.joint_names:
                i = self.joint_names.index(name)
                self.curr_joint_positions[i] = position
            if name == "xarm_gripper_drive_joint":
                self.current_gripper_position = position

        # If we have a complete joint state, republish it to update RViz visualization
        if None not in self.curr_joint_positions and self.initialization_complete:
            robot_state = RobotState()
            robot_state.joint_state.name = self.joint_names.copy()
            robot_state.joint_state.position = self.curr_joint_positions.copy()

            # Set the robot state in the planning scene
            self.planning_scene.robot_state = robot_state
            self.planning_scene.is_diff = True
            self.apply_planning_scene(self.planning_scene)

    def ft_sensor_callback(self, msg: WrenchStamped):
        """
        Callback for processing force/torque sensor data.
        Updates the last external force magnitude and checks against the threshold.
        """
        fx = msg.wrench.force.x
        fy = msg.wrench.force.y
        fz = msg.wrench.force.z
        self.last_ext_force_mag = math.sqrt(fx**2 + fy**2 + fz**2)

    def check_ft_threshold(self):
        """
        Called at high rate. If a trajectory is running and the net force exceeds
        the threshold, cancel the action immediately.
        """
        if not self.is_executing or self.controller_goal_handle is None:
            return

        if self.last_ext_force_mag > self.ft_threshold:
            self.log_warn(f"FT threshold exceeded: {self.last_ext_force_mag:.2f} N > {self.ft_threshold:.2f} N")
            self.publish_collision_status(True)
            
            # Cancel the low-level controller goal
            cancel_fut = self.controller_goal_handle.cancel_goal_async()
            spin_until_future_complete(self, cancel_fut, timeout_sec=1.0)
            self.log_info("Controller trajectory canceled")
            
            # Mark as not executing so new commands can be processed
            with self.queue_lock:
                self.is_executing = False
                if self.command_queue:
                    self.command_queue.popleft()  # Remove the command that caused collision
                    self.log_info("Command removed from queue due to collision")

    def publish_collision_status(self, is_collision):
        """
        Publish the collision status to notify other nodes.
        """
        msg = Bool()
        msg.data = is_collision
        self.collision_pub.publish(msg)
        if is_collision:
            self.log_warn("Collision detected!")

    def publish_current_joint_positions(self):
        """
        Publish the current joint positions (converted to degrees) to the relevant topic.
        """
        if None in self.curr_joint_positions:
            return
        msg = JointState()
        msg.name = self.joint_names
        msg.position = [math.degrees(pos) for pos in self.curr_joint_positions]
        self.curr_joint_position_deg_pub.publish(msg)

    def publish_current_pose(self):
        """
        Compute and publish the current end-effector pose using forward kinematics.
        """
        if None in self.curr_joint_positions:
            return
        
        robot_state = RobotState(joint_state=JointState(name=self.joint_names, position=self.curr_joint_positions))

        req = GetPositionFK.Request()
        req.header.frame_id = "link_base"
        req.fk_link_names = ["link_tcp"]
        req.robot_state = robot_state

        future = self.fk_client.call_async(req)
        future.add_done_callback(self.publish_current_pose_cb)

    def publish_current_pose_cb(self, future):
        """
        Callback function for handling the forward kinematics service response and publishing the pose.
        """
        try:
            res = future.result()
            if res is not None and len(res.pose_stamped) > 0:
                ee_pose = res.pose_stamped[0].pose
                self.current_pose_pub.publish(ee_pose)
        except Exception as e:
            self.log_error(f"FK service call failed: {e}")

    def publish_gripper_position(self):
        """
        Publish the current gripper position to the relevant topic.
        """
        if self.current_gripper_position is None:
            return
        msg = Float64()
        msg.data = self.current_gripper_position
        self.gripper_position_pub.publish(msg)

    def compute_ik_and_execute_joint_async(self, target_pose: Pose, callback=None):
        """
        Compute inverse kinematics for the given target pose and execute the resulting joint trajectory asynchronously.
        """
        ik_req = GetPositionIK.Request()
        ik_req.ik_request.group_name = "xarm7"
        ik_req.ik_request.robot_state.is_diff = True
        ik_req.ik_request.pose_stamped.header.frame_id = "link_base"
        ik_req.ik_request.pose_stamped.pose = target_pose
        ik_req.ik_request.timeout.sec = 2

        future_ik = self.compute_ik_client.call_async(ik_req)
        future_ik.add_done_callback(lambda f: self.compute_ik_done_cb(f, callback))

    def compute_ik_done_cb(self, future, callback=None):
        """
        Callback for processing the inverse kinematics result.
        Executes a joint motion if the IK solution is valid.
        """
        try:
            res = future.result()
        except Exception as e:
            self.log_error(f"IK call failed: {e}")
            if callback:
                callback(False)
            return

        if res.error_code.val != 1:
            self.log_warn(f"IK did not succeed, error code: {res.error_code.val}")
            if callback:
                callback(False)
            return

        joint_solution = res.solution.joint_state
        desired_positions = [0.0] * len(self.joint_names)
        for name, pos in zip(joint_solution.name, joint_solution.position):
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                desired_positions[idx] = pos

        self.log_info("IK succeeded; now planning joint motion to that IK solution.")
        self.plan_execute_joint_target_async(desired_positions, callback=callback)

    def plan_execute_joint_target_async(self, j_pos, callback=None):
        """
        Plan and execute a joint trajectory asynchronously to move the robot to the specified joint positions.
        """
        req = GetMotionPlan.Request()
        motion_req = MotionPlanRequest()
        motion_req.workspace_parameters.header.frame_id = "link_base"
        motion_req.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        motion_req.start_state.is_diff = True

        # Create joint constraints for all joints
        c = Constraints()
        motion_req.goal_constraints.append(c)
        
        # Create constraints with named parameters instead of positional arguments
        for i, joint_name in enumerate(self.joint_names):
            constraint = JointConstraint()
            constraint.joint_name = joint_name
            constraint.position = j_pos[i]
            constraint.tolerance_above = 0.001
            constraint.tolerance_below = 0.001
            constraint.weight = 1.0
            c.joint_constraints.append(constraint)
            
        motion_req.group_name = "xarm7"
        motion_req.num_planning_attempts = 10
        motion_req.allowed_planning_time = 5.0
        motion_req.path_constraints.name = "disable_collisions"
        if self.use_sim:
            motion_req.max_velocity_scaling_factor = 0.25
            motion_req.max_acceleration_scaling_factor = 0.25 
        else:
            motion_req.max_velocity_scaling_factor = 0.04
            motion_req.max_acceleration_scaling_factor = 0.04

        req.motion_plan_request = motion_req

        self.log_info(f'Planning joint motion to positions (deg): {[math.degrees(a) for a in j_pos]}')
        future = self.plan_path_client.call_async(req)
        future.add_done_callback(lambda f: self.plan_path_done_cb(f, callback))

    def plan_path_done_cb(self, future, callback):
        """
        Callback function to handle the result of joint path planning and initiate trajectory execution.
        """
        try:
            result = future.result()
        except Exception as e:
            self.log_error(f"Joint path plan service call failed: {e}")
            if callback:
                callback(False)
            return

        if result.motion_plan_response.error_code.val != 1:
            self.log_error(f"Planning failed, error code = {result.motion_plan_response.error_code.val}")
            if callback:
                callback(False)
            return
        
        # Check for collision-related planning errors
        if result.motion_plan_response.error_code.val == -31: # PLANNING_FAILED due to collision
            self.log_error("Planning failed due to collision!")
            self.publish_collision_status(True)
            if callback:
                callback(False)
            return
        elif result.motion_plan_response.error_code.val != 1:
            self.log_error(f"Planning failed, error code = {result.motion_plan_response.error_code.val}")
            if callback:
                callback(False)
            return
        else:
            # Clear any previous collision status
            self.publish_collision_status(False)

        self.log_info("Joint motion plan succeeded, executing trajectory...")
        self.execute_trajectory_async(result.motion_plan_response.trajectory, callback)

    def execute_trajectory_async(self, trajectory: RobotTrajectory, callback=None):
        """
        Execute a planned trajectory asynchronously.
        Uses low-level controller for arm, MoveIt for gripper.
        """
        # Determine if this is a gripper trajectory by checking the joint names
        is_gripper_trajectory = False
        if trajectory.joint_trajectory.joint_names:
            for joint_name in trajectory.joint_trajectory.joint_names:
                if joint_name in self.gripper_joint_names or "gripper" in joint_name:
                    is_gripper_trajectory = True
                    break
        
        # If it's a gripper trajectory, use the original MoveIt execution approach
        if is_gripper_trajectory:
            self.log_info("Sending gripper trajectory via MoveIt...")
            goal_msg = ExecuteTrajectory.Goal()
            goal_msg.trajectory = trajectory
            send_goal_future = self.execute_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(lambda f: self.gripper_action_send_callback(f, callback))
        else:
            # For arm trajectories, use the direct low-level controller
            jt = trajectory.joint_trajectory
            goal_msg = FollowJointTrajectory.Goal()
            goal_msg.trajectory = jt
            self.log_info("Sending arm trajectory to low-level controller...")
            send_goal_future = self.arm_ctrl_client.send_goal_async(goal_msg)
            send_goal_future.add_done_callback(lambda f: self.low_level_controller_send_callback(f, callback))

    def low_level_controller_send_callback(self, future, callback):
        """
        Callback for handling the response after sending a trajectory goal to the low-level controller.
        """
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.log_error("FollowJointTrajectory goal was rejected by controller server.")
            if callback:
                callback(False)
            return
        
        # Store the goal handle as a class member - this is the ACTUAL controller handle
        self.controller_goal_handle = goal_handle
        
        self.log_info("Goal accepted by controller, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self.low_level_controller_execute_callback(f, callback))

    def low_level_controller_execute_callback(self, future, callback):
        """
        Callback function for processing the trajectory execution result from the low-level controller.
        """
        try:
            result = future.result().result
            
            # Check the error code
            if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                self.log_error(f"Trajectory execution failed with error code: {result.error_code}")
                if callback:
                    callback(False)
                return
                
            # Clear any previous collision status
            self.publish_collision_status(False)
            
            self.log_info("Trajectory execution succeeded.")
            if callback:
                callback(True)
        except Exception as e:
            self.log_error(f"Error in trajectory execution: {e}")
            if callback:
                callback(False)

    def gripper_action_send_callback(self, future, callback):
        """
        Callback for handling the response after sending a gripper trajectory goal to MoveIt.
        """
        goal_handle = future.result()
        self.controller_goal_handle = goal_handle
        if not goal_handle.accepted:
            self.log_error("Gripper ExecuteTrajectory goal was rejected by server.")
            if callback:
                callback(False)
            return

        self.log_info("Gripper goal accepted by MoveIt, waiting for result...")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(lambda f: self.gripper_action_execute_callback(f, callback))

    def gripper_action_execute_callback(self, future, callback):
        """
        Callback function for processing the gripper trajectory execution result from MoveIt.
        """
        result = future.result().result
        # Check for collision-related execution errors
        if result.error_code.val in [-10, -11, -12]:  # Path or goal collision errors
            self.log_error(f"Trajectory execution failed due to collision, error code: {result.error_code.val}")
            self.publish_collision_status(True)
            if callback:
                callback(False)
            return
        elif result.error_code.val != 1:
            self.log_error(f"Trajectory execution failed with error code: {result.error_code.val}")
            if callback:
                callback(False)
            return
        else:
            # Clear any previous collision status
            self.publish_collision_status(False)

        self.log_info("Trajectory execution succeeded.")
        if callback:
            callback(True)

    def plan_execute_gripper_async(self, position: float, callback=None):
        """
        Plan and execute a gripper motion asynchronously to move the gripper to the specified position.
        """
        req = GetMotionPlan.Request()
        motion_req = MotionPlanRequest()
        motion_req.workspace_parameters.header.frame_id = "link_base"
        motion_req.workspace_parameters.header.stamp = self.get_clock().now().to_msg()
        motion_req.start_state.is_diff = True
        
        # Create joint constraints for gripper
        c = Constraints()
        motion_req.goal_constraints.append(c)
        
        # Add joint constraints for the gripper
        c.joint_constraints = [
            JointConstraint(
                joint_name=jn,
                position=position,
                tolerance_above=0.01,
                tolerance_below=0.01,
                weight=1.0
            )
            for jn in self.gripper_joint_names
        ]
        
        motion_req.group_name = self.gripper_group_name
        motion_req.num_planning_attempts = 10
        motion_req.allowed_planning_time = 5.0
        motion_req.max_velocity_scaling_factor = 0.1
        motion_req.max_acceleration_scaling_factor = 0.1

        # Set request and call service
        req.motion_plan_request = motion_req
        self.log_info(f"Planning gripper motion to {math.degrees(position):.2f}")

        # Call service and add callback
        future = self.plan_path_client.call_async(req)
        future.add_done_callback(lambda f: self.plan_path_done_cb(f, callback))

    ####################################################################
    # LOGGING METHODS
    ####################################################################
    def log_info(self, message):
        """Log information with green color"""
        colored_message = f"{self.GREEN}[XArm] {message}{self.RESET}"
        self.get_logger().info(colored_message)
    
    def log_warn(self, message):
        """Log warnings with red color"""
        colored_message = f"{self.RED}{self.BOLD}[XArm] WARNING: {message}{self.RESET}"
        self.get_logger().warn(colored_message)
    
    def log_error(self, message):
        """Log errors with red color and bold"""
        colored_message = f"{self.RED}{self.BOLD}[XArm] ERROR: {message}{self.RESET}"
        self.get_logger().error(colored_message)



def main(args=None):
    """
    Main function to initialize and run the ME314_XArm_Queue_Commander node.
    """
    rclpy.init(args=args)
    commander = ME314_XArm_Queue_Commander()

    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        pass

    commander.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
