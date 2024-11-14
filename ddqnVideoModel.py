import gymnasium
import highway_env
import numpy as np
from rl_agents.agents.deep_q_network.pytorch import DQNAgent
import torch
from gymnasium.wrappers import RecordVideo

# Initialize environment
env = gymnasium.make("highway-v0", render_mode='rgb_array',
                     config={"duration": 40, "initial_lane_id": 0})
env = RecordVideo(env,
                  video_folder="run",
                  episode_trigger=lambda e: True)

# Provide the video recorder to the wrapped environment
# so it can send it intermediate simulation frames.
env.unwrapped.set_record_video_wrapper(env)
env.reset()


agent_config = {
    "model": {
        "type": "MultiLayerPerceptron",
        "layers": [256, 256]  # Optuna-selected architecture
    },
    "gamma": 0.9859297882341389,  # Optuna-selected gamma for better long-term focus
    "n_steps": 1,
    "batch_size": 38,  # Optuna-selected batch size for more stable updates
    "memory_capacity": 8068,  # Optuna-selected memory capacity for efficient training
    # Start training after 1000 steps to collect sufficient experience
    "learning_starts": 1000,
    # Optuna-selected learning rate for balanced convergence
    "learning_rate": 0.0002817553371974497,
    "train_frequency": 4,  # Update every 4 steps
    "target_update": 100,  # Update target network less frequently to stabilize training
    "double": True,  # Enable Double DQN to reduce overestimation
    "exploration": {
        "method": "EpsilonGreedy",
        "tau": 1215,  # Optuna-selected epsilon decay rate for prolonged exploration
        "temperature": 1.0,
        # Optuna-selected final epsilon for better exploitation
        "final_temperature": 0.030210359431679448
    },
    # Use MPS on Mac M1 if available
    "device": "mps" if torch.backends.mps.is_available() else "cpu"
}


# Instantiate the DQN agent
agent = DQNAgent(env, agent_config)

# Load the saved model
model_path = "highway_ddqn/saved_models/ddqn_tuned.tar"
agent.load(model_path)
agent.eval()  # Set the agent in evaluation mode

# Run the agent

state, info = env.reset()
done = truncated = False
total_reward = 0
while not (done or truncated):
    action = agent.act(state)
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward
    env.render()  # Visualize the agent's behavior

env.close()
