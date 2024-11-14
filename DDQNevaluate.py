import csv
import gymnasium
from stable_baselines3 import DQN
import torch
import highway_env
import numpy as np
import time
from rl_agents.agents.deep_q_network.pytorch import DQNAgent


config = {

}

# Initialize environment
env = gymnasium.make("highway-fast-v0", render_mode='rgb_array', config=config)

# Reset environment before starting evaluation
env.reset()

# Define configuration for DQN agent
ddqn_config = {
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


filepath = "highway_ddqn/saved_models/"
filename = "ddqn_tuned.tar"
model = DQNAgent(env, ddqn_config)
model.load(filepath + filename)
model.eval()  # Set the agent in evaluation mode

# Initialize metrics
total_collisions = 0
total_lane_changes = 0
total_unsafe_lane_changes = 0
total_speed = 0
total_right_lane_time = 0
total_front_car_distance = 0
episodes = 100
episode_lengths = []

# Initialize a dictionary to store metrics for each vehicle
episode_metrics = {}


def is_unsafe_lane_change(env, prev_lane, new_lane):
    ego_vehicle = env.unwrapped.vehicle
    agent_position = ego_vehicle.position[0]
    unsafe_distance_front = 25.0  # Threshold for vehicles in front
    unsafe_distance_behind = 25.0  # Threshold for vehicles behind
    reasons = []  # To store reasons for the lane change being unsafe

    # Only check if actually changing lanes
    if prev_lane == new_lane:
        return False, reasons

    for vehicle in env.unwrapped.road.vehicles:
        if vehicle == ego_vehicle:  # Skip ego vehicle
            continue

        # Check if the vehicle is in the target lane
        if vehicle.lane_index[2] == new_lane:
            distance = vehicle.position[0] - agent_position

            # Check if the vehicle is within unsafe distance in front
            if distance > 0 and distance < unsafe_distance_front:
                reasons.append(
                    f"Vehicle in front at distance {distance:.2f}m within unsafe front threshold {unsafe_distance_front}m")

            # Check if the vehicle is within unsafe distance behind
            elif distance < 0 and abs(distance) < unsafe_distance_behind:
                reasons.append(
                    f"Vehicle behind at distance {abs(distance):.2f}m within unsafe behind threshold {unsafe_distance_behind}m")

    # Return True if any reasons for unsafety were found
    is_unsafe = bool(reasons)
    return is_unsafe, reasons


# Run episodes
for episode in range(episodes):
    done = truncated = False
    obs, info = env.reset()
    episode_speed = 0
    episode_right_lane_time = 0
    episode_lane_changes = 0
    episode_unsafe_lane_changes = 0
    episode_front_car_distance = 0
    episode_steps = 0
    collision_occurred = False
    previous_lane_index = env.unwrapped.vehicle.target_lane_index  # Initial lane index

    print(f"\nStarting Episode {episode + 1}/{episodes}")

    while not (done or truncated):
        action = model.act(obs)
        obs, reward, done, truncated, info = env.step(action)

        # Render each step for real-time observation
        env.render()

        # Track metrics per step
        current_speed = info['speed']
        current_lane_index = tuple(
            map(int, env.unwrapped.vehicle.target_lane_index))
        front_car_distance = env.unwrapped.vehicle.distance_to_front_car()

        # Increment accumulators
        episode_speed += current_speed
        episode_front_car_distance += front_car_distance
        episode_steps += 1

        # Check if agent is in the right-most lane
        if current_lane_index[2] == env.unwrapped.config["lanes_count"] - 1:
            episode_right_lane_time += 1

        # Detect lane change
        lane_change_occurred = False
        if current_lane_index != previous_lane_index:
            episode_lane_changes += 1
            lane_change_occurred = True

            # Check for unsafe lane change with debugging details
            unsafe_lane_change, reasons = is_unsafe_lane_change(
                env, previous_lane_index[2], current_lane_index[2])
            if unsafe_lane_change:
                episode_unsafe_lane_changes += 1

                # Pause execution to inspect the unsafe lane change
                print(
                    f"\nUnsafe lane change detected at Episode {episode + 1}, Step {episode_steps}")
                print("Reasons for unsafety:")
                for reason in reasons:
                    print(f" - {reason}")
                # print("Pausing execution. Press Enter to continue.")
                # input()  # Wait for user input to continue

            # Update previous lane index
            previous_lane_index = current_lane_index

        # Check for collisions
        collision_occurred = info['crashed']

        # Display the current state information
        print(
            f"Episode {episode + 1} | Step {episode_steps} | "
            f"Speed: {current_speed:.2f} | Lane: {current_lane_index[2]} | "
            f"Distance to Front Car: {front_car_distance:.2f} | "
            f"Unsafe Lane Change: {'Yes' if lane_change_occurred and unsafe_lane_change else 'No'} | "
            f"Collision: {'Yes' if collision_occurred else 'No'}",
            flush=True
        )

        # Optional delay to slow down the updates for easier observation
        time.sleep(0.01)

    # Calculate episode-based metrics
    total_collisions += int(collision_occurred)
    total_lane_changes += episode_lane_changes
    total_unsafe_lane_changes += episode_unsafe_lane_changes
    total_speed += episode_speed / episode_steps
    total_right_lane_time += episode_right_lane_time / episode_steps
    total_front_car_distance += episode_front_car_distance / episode_steps
    episode_lengths.append(episode_steps)

    # Store vehicle-specific metrics
    episode_metrics[episode] = {
        "collision": int(collision_occurred),
        "average_speed": episode_speed / episode_steps,
        "right_lane_percentage": (episode_right_lane_time / episode_steps) * 100,
        "lane_changes": episode_lane_changes,
        "unsafe_lane_changes": episode_unsafe_lane_changes,
        "average_front_car_distance": episode_front_car_distance / episode_steps
    }

# Close the environment after simulation
env.close()

# Write metrics to a CSV file
with open("DDQN_episode_metrics.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(["Episode", "Collision", "Average Speed", "Right Lane Percentage",
                     "Lane Changes", "Unsafe Lane Changes", "Average Front Car Distance"])

    # Write each episode's metrics
    for episode, metrics in episode_metrics.items():
        writer.writerow([episode + 1, metrics["collision"], metrics["average_speed"],
                         metrics["right_lane_percentage"], metrics["lane_changes"],
                         metrics["unsafe_lane_changes"], metrics["average_front_car_distance"]])

print("Metrics have been saved to DDQN_episode_metrics.csv")
