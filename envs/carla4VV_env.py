import gymnasium as gym
from gym import spaces
from stable_baselines3 import PPO
import carla
import math
import numpy as np
import pandas as pd
import time

class VV_ENV(gym.Env):

    metadata = {'render.modes': ['human']}

    def __init__(self, host, port, track_data_csv, finish_line_coords, off_track_threshold, stationary_threshold, *args, **kwargs):
        super(VV_ENV, self).__init__()
        # self.env
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.track_data = self.load_track_data(track_data_csv)
        self.finish_line_coords = finish_line_coords
        self.off_track_threshold = off_track_threshold
        self.stationary_threshold = stationary_threshold
        self.world = None
        self.vehicle = None
        self.current_step = 0
        self.last_movement_step = 0
        self.initialize_carla()

        # Define action space (throttle, steer, brake)
        self.action_space = spaces.Box(low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)

        # Define observation space (X, Y, Z, Velocity, Pitch, Yaw, Roll)
        self.observation_space = spaces.Box(
            low=np.array([-500, -500, -500, 0, -180, -180, -180]),
            high=np.array([500, 500, 500, 50, 180, 180, 180]),
            dtype=np.float32
        )

    def initialize_carla(self):
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        # self.destroy_all_actors()
        blueprint_library = self.world.get_blueprint_library()
        self.model_3_bp = blueprint_library.filter('model3')[0]

        # set color of vehicle to blue
        # self.model_3_bp.set_attribute('color', '0, 0, 255')

        #spawn x.y.z: 127.1	 -2.1	0.1538
        spawn_point = carla.Transform(carla.Location(x=127.1, y=-2.1, z=2.1538), carla.Rotation(yaw=180))

        # # Spawn a random ego vehicle
        # blueprint_library = self.world.get_blueprint_library()
        # vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        # spawn_point = self.map.get_spawn_points()[0]
        # # print(map.get_spawn_points())
        # self.vehicle = None

        # while self.vehicle is None:
        #     self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        self.vehicle = self.world.spawn_actor(self.model_3_bp, spawn_point)
    
    def load_track_data(self, track_data_csv):
        track_data = pd.read_csv(track_data_csv)
        # track_data['typical_speed'] = track_data['typical_speed'].interpolate()
        # track_data['typical_steer'] = track_data['typical_steer'].interpolate()
        return track_data
    
    def close(self):
        # Perform any necessary cleanup here
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

        # Add any other cleanup tasks (e.g., closing connections, releasing resources)
        print("Environment closed and resources released.")


    def render(self, mode='human'):
        # Check if CARLA provides its own rendering
        if mode == 'human':
            # Display the rendering provided by CARLA (if any).
            # This is just a placeholder as CARLA's simulation might already be visible in a separate window.
            print("Rendering the current state of the environment.")
        else:
            # Other rendering modes can be handled here (e.g., saving frames to a file)
            pass

    def reset(self):
        # Reset the environment
        if self.vehicle is not None:
            self.vehicle.destroy()
            self.vehicle = None
        self.close()
        spawn_point = carla.Transform(carla.Location(x=127.1, y=-2.1, z=0.1538), carla.Rotation(yaw=180))
        self.vehicle = self.world.spawn_actor(self.model_3_bp, spawn_point)

        # # Spawn a random ego vehicle
        # blueprint_library = self.world.get_blueprint_library()
        # vehicle_bp = blueprint_library.filter('vehicle.*')[0]
        # spawn_point = self.map.get_spawn_points()[0]

        # while self.vehicle is None:
        #     self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)

        self.current_step = 0
        self.last_movement_step = 0
        
        return self.get_car_state()

    def step(self, action):
        throttle, steer, brake = action

        vehicle_control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        self.vehicle.apply_control(vehicle_control)
        if throttle > 0 or abs(steer) > 0:
            self.last_movement_step = self.current_step
        time.sleep(1 / 60.0)  # Assuming a simulation step is 1/60 seconds

        self.current_step += 1
        new_state = self.get_car_state()
        done = self.check_termination_conditions(new_state)
        reward = self.calculate_reward(new_state)

        info = {
            'step': self.current_step,
            'velocity': self.calculate_velocity(self.vehicle),
        }  # Placeholder for any additional diagnostic information

        return new_state, reward, done, info

    def calculate_velocity(self, vehicle):
        # Get the vehicle's velocity vector
        velocity_vector = vehicle.get_velocity()
        
        # Calculate the scalar velocity as the magnitude of the velocity vector
        scalar_velocity_mps = math.sqrt(velocity_vector.x**2 + velocity_vector.y**2 + velocity_vector.z**2)
        
        # Convert from m/s to mph (1 meter per second is equal to 2.23694 miles per hour)
        scalar_velocity_mph = scalar_velocity_mps * 2.23694
        
        return scalar_velocity_mph


    def calculate_reward(self, current_state):
        # Get the current vehicle's telemetry data
        vehicle_telemetry = self.get_car_state()

        # Calculate the current velocity using the current time, x, and y
        # current_time = [self.current_step / 60.0]  # Assuming a simulation step is 1/60 seconds
        velocity_mph = self.calculate_velocity(self.vehicle)

        # Find the closest point on the participant's trajectory
        closest_point_index = ((self.track_data['x'] - vehicle_telemetry['x'])**2 +
                               (self.track_data['y'] - vehicle_telemetry['y'])**2).idxmin()
        closest_point = self.track_data.iloc[closest_point_index]

        # Get the participant's typical speed and steering angle for the closest point
        typical_speed = closest_point['velocity']
        typical_steer = closest_point['steer']

        # Calculate the differences from the current speed and steer to the participant's typical values
        speed_difference = abs(velocity_mph - typical_speed)
        steer_difference = abs(current_state['steer'] - typical_steer)

        # Define the reward components
        reward = 0  # Start with a neutral reward
        reward -= speed_difference  # Penalize deviation from the participant's typical speed
        reward -= steer_difference  # Penalize deviation from the participant's typical steering angle

        # Penalize for excessive control inputs
        reward -= 0.1 * abs(current_state['steer'])  # Less penalty for steering
        reward -= 0.2 * abs(current_state['brake'])  # More penalty for braking

        # Penalize for going backwards or being stationary which is not desired
        if velocity_mph < 0 or velocity_mph == 0:
            reward -= 10

        return reward

    def check_termination_conditions(self, current_state):
        # Check if car has reached the finish line
        if (self.finish_line_coords['x_min'] <= current_state['x'] <= self.finish_line_coords['x_max']) and \
           (self.finish_line_coords['y_min'] <= current_state['y'] <= self.finish_line_coords['y_max']):
            return True  # Finish line reached

        # # Check if car is off track
        # if np.linalg.norm([current_state['x'], current_state['y']]) > self.off_track_threshold:
        #     self.reset()  # Reset the environment because the car is off track
        #     return True  # Termination condition met

        # # Check if car is stationary for too long
        # if current_state['throttle'] == 0 and current_state['steer'] == 0 and current_state['brake'] > 0:
        #     if self.current_step - self.last_movement_step > self.stationary_threshold:
        #         self.reset()  # Reset the environment because the car is stationary for too long
        #         return True  # Termination condition met

        # If none of the above conditions are met, continue the episode
        return False

    # def apply_action(self, action):
    #     # Placeholder for code to apply actions to the vehicle
    #     pass


    def get_car_state(self):
        # Here you would get the vehicle's telemetry data
        vehicle_location = self.vehicle.get_location()
        vehicle_transform = self.vehicle.get_transform()
        vehicle_control = self.vehicle.get_control()
        return {
            'x': vehicle_location.x,
            'y': vehicle_location.y,
            'z': vehicle_location.z,
            'pitch': vehicle_transform.rotation.pitch,
            'yaw': vehicle_transform.rotation.yaw,
            'roll': vehicle_transform.rotation.roll,
            'throttle': vehicle_control.throttle,
            'steer': vehicle_control.steer,
            'brake': vehicle_control.brake,
        }

    def destroy_vehicle(self):
        if self.vehicle:
            self.vehicle.destroy()
            self.vehicle = None

