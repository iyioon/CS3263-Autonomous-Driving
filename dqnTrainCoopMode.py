import highway_env
import gymnasium
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import DQN

# Referenced from
# https://highway-env.farama.org/quickstart/#training-an-agent

# filepath = "highway_dqnCoop/"
# filename = "dqn_Coop_model2"

# config = {
#     "min_safe_distance": 53,
#     "unsafe_lane_change_distance": 53,
#     "distance_penalty": -0.3,
#     "lane_change_penalty": -0.1,
#     "unsafe_change_penalty": -0.7,
#     "collision_reward": -1,
# }

# filepath = "highway_dqnCoop/"
# filename = "dqn_Coop_model3"

# config = {
#     "min_safe_distance": 53,
#     "unsafe_lane_change_distance": 53,
#     "distance_penalty": -0.7,
#     "lane_change_penalty": -0.1,
#     "unsafe_change_penalty": -0.7,
#     "collision_reward": -1,
# }

# filepath = "highway_dqnCoop/"
# filename = "dqn_Coop_model4"

# config = {
#     "min_safe_distance": 53,
#     "unsafe_lane_change_distance": 53,
#     "distance_penalty": -1,
#     "lane_change_penalty": -0.1,
#     "unsafe_change_penalty": -1,
#     "collision_reward": -1,
#     "high_speed_reward": 0.6,
# }

filepath = "highway_dqnCoop/"
filename = "dqn_Coop_model5"

config = {}

env = gymnasium.make(
    "highway-fast-v0", config=config)


model = DQN('MlpPolicy', env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=5e-4,
            buffer_size=15000,
            learning_starts=200,
            batch_size=32,
            gamma=0.8,
            train_freq=1,
            gradient_steps=1,
            target_update_interval=50,
            verbose=1,
            tensorboard_log="highway_dqnCoop/")
model.learn(int(2e4))

model.save(filepath + filename)
