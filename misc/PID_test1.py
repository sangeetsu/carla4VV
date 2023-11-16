import carla
import time
import pygame
import numpy as np
import queue

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from transforms import carla_rotation_to_ros_quaternion
# import cv2

# ROS2 Node for publishing Path
class VehicleTrajectoryPublisher(Node):

    def __init__(self):
        super().__init__('vehicle_trajectory_publisher')
        self.path_publisher = self.create_publisher(Path, '/ego_vehicle/path', 10)
        self.pose_publisher = self.create_publisher(PoseStamped, '/ego_vehicle/pose', 10)
        self.path_msg = Path()
        self.path_msg.header.frame_id = "map"

    def publish_path(self, x, y, z, qx, qy, qz, qw):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw
        self.path_msg.poses.append(pose_stamped)
        self.path_publisher.publish(self.path_msg)
    
    def publish_pose(self, x, y, z, qx, qy, qz, qw):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "map"
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z
        pose_stamped.pose.orientation.x = qx
        pose_stamped.pose.orientation.y = qy
        pose_stamped.pose.orientation.z = qz
        pose_stamped.pose.orientation.w = qw
        self.pose_publisher.publish(pose_stamped)

# Initializations
rclpy.init()
ros_node = VehicleTrajectoryPublisher()
pygame.init()
image_queue = queue.Queue()


# HUD related functions
class HUD:
    def __init__(self, width, height):
        self.dim = (width, height)
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.display = pygame.display.set_mode(self.dim)
        pygame.display.set_caption('CARLA HUD')
    
    def display_image(self, image_surface):
        """Display the given image surface on the HUD."""
        self.display.blit(image_surface, (0, 0))
        pygame.display.flip()

    def tick(self):
        pygame.event.pump()

# Callback for the camera sensor to process its data
def process_image(data):
    image_data = bytes(data.raw_data)
    array = np.frombuffer(image_data, dtype=np.dtype("uint8")).reshape((data.height, data.width, 4))
    # Convert to BGRA to RGBA
    array = array[:, :, [2, 1, 0, 3]]
    image_surface = pygame.image.fromstring(array.tobytes(), (data.width, data.height), 'RGBA')
    image_queue.put(image_surface)



# PID gains
kp = 1.0
ki = 0.1
kd = 0.1

# Initialize the client
client = carla.Client('localhost', 2000)
client.set_timeout(2.0)

# Get the world and map
world = client.get_world()
map = world.get_map()

# Spawn a random ego vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = map.get_spawn_points()[0]
# print(map.get_spawn_points())
ego_vehicle = None

while ego_vehicle is None:
    ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)


# Initialize the HUD
hud = HUD(800, 600)

# Spawn a camera
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute("image_size_x", str(hud.dim[0]))
camera_bp.set_attribute("image_size_y", str(hud.dim[1]))
camera_bp.set_attribute("fov", "110")
camera_transform = carla.Transform(carla.Location(x=-6, z=3))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
camera.listen(process_image)

# Initialize the PID controller
error_sum = 0.0
last_error = 0.0

try:
    # Control loop
    while True:
        # Get the current transform of the ego vehicle
        transform = ego_vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation

        # Convert rotation to quaternion for ROS (or use provided quaternion if available)
        quaternion = carla_rotation_to_ros_quaternion(rotation)  

        # Calculate the error (distance from the center of the lane)
        error = transform.location.x

        # Update the error sum and calculate the error derivative
        error_sum += error
        error_diff = error - last_error
        last_error = error

        # Calculate the control signal using the PID formula
        control_signal = kp * error + ki * error_sum + kd * error_diff
        # print("Control signal: {}".format(control_signal))

        # Clamp control_signal to [-1, 1]
        control_signal = max(-1.0, min(1.0, control_signal))

        # Apply the control signal to the ego vehicle
        control = carla.VehicleControl(throttle=0.3, steer=control_signal)
        ego_vehicle.apply_control(control)

        rclpy.spin_once(ros_node, timeout_sec=0.1)

        # Publish the trajectory
        ros_node.publish_path(location.x, location.y, location.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w)

        # Publish the current pose
        ros_node.publish_pose(location.x, location.y, location.z, quaternion.x, quaternion.y, quaternion.z, quaternion.w)

        if not image_queue.empty():
            image_surface = image_queue.get()
            hud.display_image(image_surface)

        # Wait for a short time
        hud.tick()
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Simulation terminated by user. Cleaning up...")
finally:
    if ego_vehicle:
        ego_vehicle.destroy()
    if camera:
        camera.destroy()
    pygame.quit()
    rclpy.shutdown()
