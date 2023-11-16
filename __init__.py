from gym.envs.registration import register
from envs.carla4VV_env import VV_ENV

register(
    id='VV_ENV-v2',
    entry_point=VV_ENV,
)
# register(
#     id='VV_ENV-v2',
#     entry_point=lambda config: VV_ENV(
#         config['host'],
#         config['port'],
#         config['track_data_csv'],
#         config['finish_line_coords'],
#         config['off_track_threshold'],
#         config['stationary_threshold']
#     ),
#     # You might want to add kwargs such as max_episode_steps if relevant
# )