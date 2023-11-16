import gymnasium as gym
from stable_baselines3 import PPO
from carla4VV_env import VV_ENV  # Importing the custom environment
from env_config import ENV_CONFIG  # Importing the configuration

# Create an instance of the environment with the specified configuration
# env = VV_ENV(**ENV_CONFIG)
env = gym.make('VV_ENV-v2', config=ENV_CONFIG)
'''
# Initialize the PPO model with your environment
model = PPO("MlpPolicy", env, verbose=1)

# Train the model
total_timesteps = 10000  # Adjust this value as needed
model.learn(total_timesteps=total_timesteps)

# Optionally, save the model
model.save("ppo_carla4VV")

# Close the environment after training
env.close()
'''