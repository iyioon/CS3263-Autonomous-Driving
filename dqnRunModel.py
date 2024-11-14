import gymnasium
from stable_baselines3 import DQN
import highway_env

# filepath = "highway_dqnCoop/"
# filename = "dqn_Coop_model2"


# config = {
#     "min_safe_distance": 53,
#     "unsafe_lane_change_distance": 53,
#     "distance_penalty": -1,
#     "lane_change_penalty": -0.1,
#     "unsafe_change_penalty": -1,
#     "collision_reward": -1,
#     "high_speed_reward": 0.6,
# }

config = {}
filepath = "highway_dqnCoop/"
filename = "dqn_Coop_model5"

env = gymnasium.make('highway-v0', render_mode='human',
                     config=config)


model = DQN.load(filepath + filename)
while True:
    done = truncated = False
    obs, info = env.reset()
    total_reward = 0
    while not (done or truncated):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()

    print("Finished episode with total reward:", total_reward)
