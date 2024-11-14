import gymnasium
from stable_baselines3 import DQN
import highway_env
from gymnasium.wrappers import RecordVideo
import sys

try:
    # Initialize environment with rgb_array render mode and video recording
    config = {"initial_lane_id": 0}
    env = gymnasium.make(
        'highway-v0', render_mode='rgb_array', config=config)
    env = RecordVideo(env,
                      video_folder="run",
                      episode_trigger=lambda e: True)  # record all episodes

    # Provide the video recorder to the wrapped environment
    # so it can send it intermediate simulation frames.
    env.unwrapped.set_record_video_wrapper(env)

    # Load the saved model
    filepath = "highway_dqnCoop/"
    filename = "dqn_Coop_model5"
    model = DQN.load(filepath + filename)

    # Run episodes
    num_episodes = 1  # Limit number of episodes to record
    for episode in range(num_episodes):
        done = truncated = False
        obs, info = env.reset()

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)

        if done or truncated:
            print(f"Episode {episode + 1} completed")

except KeyboardInterrupt:
    print("\nExiting gracefully...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    env.close()
    sys.exit(0)
