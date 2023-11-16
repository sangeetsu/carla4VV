import gymnasium as gym
from stable_baselines3 import PPO
from carla4VV.envs.carla4VV_env import carla4VV_v2 as VV_ENV
from carla4VV.config.config import ENV_CONFIG

# Define your track data and finish line coordinates here
track_data_csv = '/home/sangeetsu/Virtuous-Vehicle/combined_scrapes/AA1498final.csv'
finish_line_coords = {'x_min': 40, 'x_max': 50, 'y_min': 50, 'y_max': 60}
off_track_threshold = 10
stationary_threshold = 10

num_episodes = 100

# Initialize the environment
env = VV_ENV(**ENV_CONFIG)
# Instantiate the model
model = PPO('MlpPolicy', env, verbose=1)
# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        # Query the RL model for an action
        action = model.predict(state)
        
        # Apply the action to the environment
        next_state, reward, done = env.step(action)
        
        # Train (update) the RL model
        model.train(state, action, reward, next_state, done)

        # Update the state, total reward, and steps
        state = next_state
        total_reward += reward
        steps += 1

        if done:
            print(f"Episode {episode+1}/{num_episodes} finished after {steps} steps with total reward {total_reward:.2f}")
            break

env.destroy_vehicle()