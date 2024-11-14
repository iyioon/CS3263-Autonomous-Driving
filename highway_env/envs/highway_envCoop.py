from __future__ import annotations

import numpy as np

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.envs.common.action import Action
from highway_env.road.road import Road, RoadNetwork
from highway_env.utils import near_split
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle


Observation = np.ndarray


def is_unsafe_lane_change(env, prev_lane, new_lane, unsafe_distance=25):
    ego_vehicle = env.unwrapped.vehicle
    agent_position = ego_vehicle.position[0]
    unsafe_distance_front = unsafe_distance  # Threshold for vehicles in front
    unsafe_distance_behind = unsafe_distance  # Threshold for vehicles behind
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


class HighwayEnv(AbstractEnv):
    """
    A highway driving environment.

    The vehicle is driving on a straight highway with several lanes, and is rewarded for reaching a high speed,
    staying on the rightmost lanes and avoiding collisions.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {"type": "Kinematics"},
                "action": {
                    "type": "DiscreteMetaAction",
                },
                "lanes_count": 4,
                "vehicles_count": 50,
                "controlled_vehicles": 1,
                "initial_lane_id": None,
                "duration": 40,  # [s]
                "ego_spacing": 2,
                "vehicles_density": 1,
                # The reward received when colliding with a vehicle.
                "collision_reward": -1,
                # The reward received when driving on the right-most lanes, linearly mapped to
                "right_lane_reward": 0.1,
                # zero for other lanes.
                # The reward received when driving at full speed, linearly mapped to zero for
                "high_speed_reward": 0.4,
                # lower speeds according to config["reward_speed_range"].
                # The reward received at each lane change action.
                "lane_change_reward": 0,
                "reward_speed_range": [20, 30],
                "normalize_reward": True,
                "offroad_terminal": False,
                "unsafe_distance_penalty": -0.5,  # Penalty for unsafe lane changes
                "unsafe_distance": 25,            # Unsafe distance threshold in meters
                "distance_penalty": -0.5,         # Penalty for not keeping a safe distance
            }
        )
        return config

    def _reset(self) -> None:
        self._create_road()
        self._create_vehicles()

    def _create_road(self) -> None:
        """Create a road composed of straight adjacent lanes."""
        self.road = Road(
            network=RoadNetwork.straight_road_network(
                self.config["lanes_count"], speed_limit=30
            ),
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        """Create some new random vehicles of a given type, and add them on the road."""
        other_vehicles_type = utils.class_from_path(
            self.config["other_vehicles_type"])
        other_per_controlled = near_split(
            self.config["vehicles_count"], num_bins=self.config["controlled_vehicles"]
        )

        self.controlled_vehicles = []
        for others in other_per_controlled:
            vehicle = Vehicle.create_random(
                self.road,
                speed=25,
                lane_id=self.config["initial_lane_id"],
                spacing=self.config["ego_spacing"],
            )
            vehicle = self.action_type.vehicle_class(
                self.road, vehicle.position, vehicle.heading, vehicle.speed
            )
            self.controlled_vehicles.append(vehicle)
            self.road.vehicles.append(vehicle)

            for _ in range(others):
                vehicle = other_vehicles_type.create_random(
                    self.road, spacing=1 / self.config["vehicles_density"]
                )
                vehicle.randomize_behavior()
                self.road.vehicles.append(vehicle)

    def _reward(self, action: Action) -> float:
        # Calculate individual reward components
        rewards = self._rewards(action)

        # Sum the weighted reward components
        reward = sum(
            self.config.get(name, 0) * value for name, value in rewards.items()
        )

        # Define min and max reward bounds for normalization, including all penalties
        if self.config["normalize_reward"]:
            min_reward = (
                self.config["collision_reward"]
                + self.config.get("unsafe_distance_penalty", 0)
                + self.config.get("lane_change_penalty", 0)
                + self.config.get("distance_penalty", 0)
            )
            max_reward = (
                self.config.get("high_speed_reward", 0)
                + self.config.get("right_lane_reward", 0)
            )

            # Normalize reward to [0, 1] based on min and max bounds
            reward = utils.lmap(
                reward,
                [min_reward, max_reward],
                [0, 1],
            )

        # Apply on-road reward to zero out reward if off-road
        reward *= rewards["on_road_reward"]
        return reward

    def _rewards(self, action: Action) -> dict[str, float]:
        # Determine if a lane change action was taken
        prev_lane = self.vehicle.lane_index[2]
        new_lane = self.vehicle.target_lane_index[2] if action in [
            0, 2] else prev_lane  # 0 and 2 correspond to left and right lane changes

        # Check for unsafe lane change
        is_unsafe, reasons = is_unsafe_lane_change(
            self, prev_lane, new_lane, unsafe_distance=self.config.get(
                "unsafe_distance", 25)
        )

        # Apply a penalty if the lane change is unsafe
        unsafe_distance_penalty = self.config.get(
            "unsafe_distance_penalty", -0.5) if is_unsafe else 0

        # Check for unsafe distance to the front vehicle
        unsafe_front_distance_penalty = 0
        unsafe_distance_threshold = self.config.get("unsafe_distance", 25)

        # Get all vehicles in the same lane
        same_lane_vehicles = [v for v in self.road.vehicles
                              if v.lane_index == self.vehicle.lane_index
                              and v is not self.vehicle]

        # Find the closest vehicle in front of the ego vehicle
        front_vehicles = [v for v in same_lane_vehicles
                          if self.vehicle.lane_distance_to(v) > 0]
        if front_vehicles:
            closest_vehicle = min(
                front_vehicles, key=lambda v: self.vehicle.lane_distance_to(v))
            distance_to_front = self.vehicle.lane_distance_to(closest_vehicle)

            # Apply penalty if within unsafe distance to the front vehicle
            if distance_to_front < unsafe_distance_threshold:
                unsafe_front_distance_penalty = self.config.get(
                    "distance_penalty", -0.3)

        # Lane and speed calculations for other reward components
        neighbours = self.road.network.all_side_lanes(self.vehicle.lane_index)
        lane = (
            self.vehicle.target_lane_index[2]
            if isinstance(self.vehicle, ControlledVehicle)
            else self.vehicle.lane_index[2]
        )

        # Use forward speed rather than speed, see https://github.com/eleurent/highway-env/issues/268
        forward_speed = self.vehicle.speed * np.cos(self.vehicle.heading)
        scaled_speed = utils.lmap(
            forward_speed, self.config["reward_speed_range"], [0, 1]
        )

        # Return the reward components as a dictionary
        return {
            "collision_reward": float(self.vehicle.crashed),
            "right_lane_reward": lane / max(len(neighbours) - 1, 1),
            "high_speed_reward": np.clip(scaled_speed, 0, 1),
            "on_road_reward": float(self.vehicle.on_road),
            "unsafe_distance_penalty": unsafe_distance_penalty,
            "unsafe_front_distance_penalty": unsafe_front_distance_penalty,
        }

    def _is_terminated(self) -> bool:
        """The episode is over if the ego vehicle crashed."""
        return (
            self.vehicle.crashed
            or self.config["offroad_terminal"]
            and not self.vehicle.on_road
        )

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]


class HighwayEnvFast(HighwayEnv):
    """
    A variant of highway-v0 with faster execution:
        - lower simulation frequency
        - fewer vehicles in the scene (and fewer lanes, shorter episode duration)
        - only check collision of controlled vehicles with others
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "simulation_frequency": 5,
                "lanes_count": 3,
                "vehicles_count": 20,
                "duration": 30,  # [s]
                "ego_spacing": 1.5,
            }
        )
        return cfg

    def _create_vehicles(self) -> None:
        super()._create_vehicles()
        # Disable collision check for uncontrolled vehicles
        for vehicle in self.road.vehicles:
            if vehicle not in self.controlled_vehicles:
                vehicle.check_collisions = False
