# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


#print("TRYING TO IMPORT IN SIMULATION")
#import gr00t.eval.register_my_lift_env #I added this
#print("IMPORT IN SIMULATION DONE")
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

import gymnasium as gym
#print(gym.envs.registry.keys())
import numpy as np
import torch

# Required for robocasa environments
import robocasa  # noqa: F401
import robosuite  # noqa: F401
from robocasa.utils.gym_utils import GrootRoboCasaEnv  # noqa: F401

from gr00t.data.dataset import ModalityConfig
from gr00t.eval.service import BaseInferenceClient
from gr00t.eval.wrappers.multistep_wrapper import MultiStepWrapper
from gr00t.eval.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)

from gr00t.model.policy import BasePolicy

# from gymnasium.envs.registration import registry

# print("Available environments:")
# for env_spec in registry.values():
#     print(env_spec.id)


@dataclass
class VideoConfig:
    """Configuration for video recording settings."""

    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 10
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    """Configuration for multi-step environment settings."""

    video_delta_indices: np.ndarray = field(default=np.array([0]))
    state_delta_indices: np.ndarray = field(default=np.array([0]))
    n_action_steps: int = 16
    max_episode_steps: int = 1440


@dataclass
class SimulationConfig:
    """Main configuration for simulation environment."""

    env_name: str
    n_episodes: int = 2
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


class SimulationInferenceClient(BaseInferenceClient, BasePolicy):
    """Client for running simulations and communicating with the inference server."""

    def __init__(self, host: str = "localhost", port: int = 5555):
        """Initialize the simulation client with server connection details."""
        super().__init__(host=host, port=port)
        self.env = None

    def get_action(self, observations: Dict[str, Any], output_dir=None) -> Dict[str, Any]:
        """Get action from the inference server based on observations."""
        print("output dir in simulation_get action: ", output_dir)
        # NOTE(YL)!
        # hot fix to change the video.ego_view_bg_crop_pad_res256_freq20 to video.ego_view
        if "video.ego_view_bg_crop_pad_res256_freq20" in observations:
            observations["video.ego_view"] = observations.pop(
                "video.ego_view_bg_crop_pad_res256_freq20"
            )
        return self.call_endpoint("get_action", observations, output_dir=output_dir)

    def get_modality_config(self) -> Dict[str, ModalityConfig]:
        """Get modality configuration from the inference server."""
        return self.call_endpoint("get_modality_config", requires_input=False)

    def setup_environment(self, config: SimulationConfig) -> gym.vector.VectorEnv:
        """Set up the simulation environment based on the provided configuration."""
        # Create environment functions for each parallel environment
        env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
        # Create vector environment (sync for single env, async for multiple)
        if config.n_envs == 1:
            return gym.vector.SyncVectorEnv(env_fns)
        else:
            return gym.vector.AsyncVectorEnv(
                env_fns,
                shared_memory=False,
                context="spawn",
            )

    def run_simulation(self, config: SimulationConfig, output_dir=None) -> Tuple[str, List[bool]]:
        """Run the simulation for the specified number of episodes."""
        #print("Output dir: ", output_dir)
        
        start_time = time.time()
        print(
            f"Running {config.n_episodes} episodes for {config.env_name} with {config.n_envs} environments"
        )
        # Set up the environment
        self.env = self.setup_environment(config)
        #print("All Env properties: ", vars(self.env))
        #base=self.env

        # Initialize tracking variables
        episode_lengths = []
        current_rewards = [0] * config.n_envs
        current_lengths = [0] * config.n_envs
        completed_episodes = 0
        current_successes = [False] * config.n_envs
        episode_successes = []
        

        # Initial environment reset
        obs, _ = self.env.reset()
        #print("-----------------------------------------------------------------")
        filepath=""
        for sub_env in self.env.envs:
            while not isinstance(sub_env, VideoRecordingWrapper):
                sub_env = sub_env.env
            filepath=str(sub_env.file_path).replace(".mp4", "")

        # Main simulation loop
        all_hs=[]
        all_attentions=[]
        all_actions=[]
        while completed_episodes < config.n_episodes:
            filepath=""
            for sub_env in self.env.envs:
                while not isinstance(sub_env, VideoRecordingWrapper):
                    sub_env = sub_env.env
                filepath=str(sub_env.file_path).replace(".mp4", "")

            

            # Process observations and get actions from the server
            if output_dir != None:
                output_dir=filepath
                actions, hs, at = self._get_actions_from_server(obs, filepath)
                #print("actions output: ")
                all_hs.append(hs)
                all_attentions.append(at)
                all_actions.append(actions)
                print("Actions keys: ", actions.keys())
            else:
                actions = self._get_actions_from_server(obs, output_dir=None)
                print("Actions: ", len(actions))

            ##A little bit hacky: 
            #print("Actions: ", actions["action.left_arm"].shape)
            actions["action.waist"]=np.zeros((1, 16, 3)) #maybe should not be zeros, maybe should be previous positions?

            # Step the environment
            next_obs, rewards, terminations, truncations, env_infos = self.env.step(actions)


            # Update episode tracking
            for env_idx in range(config.n_envs):
                #print("Env infos: ", env_infos["success"][env_idx][0])
                current_successes[env_idx] |= bool(env_infos["success"][env_idx][0])
                current_rewards[env_idx] += rewards[env_idx]
                current_lengths[env_idx] += 1
                # If episode ended, store results
                if terminations[env_idx] or truncations[env_idx]:
                    episode_lengths.append(current_lengths[env_idx])
                    episode_successes.append(current_successes[env_idx])
                    current_successes[env_idx] = False
                    completed_episodes += 1
                    
                    if output_dir != None:
                        #print("Len attentions: ", len(all_attentions))#So we have 32 actions, which is 500 steps/16, I will keep it that way so we know the internal structure
                        #print("Attention shape before saving: ", all_attentions[0].shape)
                        torch.save(all_attentions, filepath+"_attentions.pt")
                        torch.save(all_hs, filepath+"_hidden_states.pt")

                        #We now basically decompress the timeline here for the arms and hands for plotting
                        right_arm_segments=np.array([a["action.right_arm"] for a in all_actions])
                        right_hand_segments=np.array([a["action.right_hand"] for a in all_actions])
                        joint_names_arms=["Shoulder Pitch", "Shoulder Roll", "Shoulder Yaw", "Elbow Pitch", "Wrist Yaw", "Wrist Roll", "Wrist Pitch"]
                        joint_names_hands=["Little Finger", "Ring Finger", "Middle Finger", "Index Finger", "Thumb Rotation", "Thumb Bending"]

                        #I also make the action plots for each episode to visually track the actions with the attention layers and the hidden layers
                        self.plot_joint_actions(right_arm_segments, joint_names_arms, right_hand_segments, joint_names_hands, filepath)
                    

                    # Reset trackers for this environment
                    current_rewards[env_idx] = 0
                    current_lengths[env_idx] = 0

                    all_hs=[]
                    all_attentions=[]
                    all_actions=[]


            obs = next_obs
        # Clean up
        self.env.reset()
        self.env.close()
        self.env = None
        print(
            f"Collecting {config.n_episodes} episodes took {time.time() - start_time:.2f} seconds"
        )
        assert (
            len(episode_successes) >= config.n_episodes
        ), f"Expected at least {config.n_episodes} episodes, got {len(episode_successes)}"
        return config.env_name, episode_successes

    def _get_actions_from_server(self, observations: Dict[str, Any], output_dir=None) -> Dict[str, Any]:
        """Process observations and get actions from the inference server."""
        # Get actions from the server
        action_dict = self.get_action(observations, output_dir=output_dir)
        # Extract actions from the response
        if "actions" in action_dict:
            actions = action_dict["actions"]
        else:
            actions = action_dict
        # Add batch dimension to actions
        return actions
    
    def plot_joint_actions(self, right_arm_segments, joint_names_arms, right_hand_segments, joint_names_hands, filepath):
        total_timesteps=len(right_arm_segments) * 16
        times=np.arange(total_timesteps)

        arm_actions = np.concatenate(right_arm_segments.squeeze(), axis=0)      # shape (T,7)
        hand_actions = np.concatenate(right_hand_segments.squeeze(), axis=0) 

        print("Total Timesteps in plot: ", total_timesteps)
        
        fig, (ax_arm, ax_hand) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={'height_ratios': [3, 2]}
            )
        
        save_path=filepath+"_action_plots.png"

        #Arm plot
        for j, name in enumerate(joint_names_arms):
            #print("J: ", j)
            print(right_arm_segments.squeeze().shape)
            ax_arm.plot(times, arm_actions[:, j], label=name)

        ax_arm.set_ylabel('Arm Action Value')
        ax_arm.set_title('Arm Joint Actions Over Time')
        ax_arm.legend(loc='upper right', fontsize='small', ncol=2)
        ax_arm.grid(True)

        #Hand Plot:
        for j, name in enumerate(joint_names_hands):
            ax_hand.plot(times, hand_actions[:, j], label=name)
        ax_hand.set_xlabel('Timestep')
        ax_hand.set_ylabel('Hand Action Value')
        ax_hand.set_title('Hand Joint Actions Over Time')
        ax_hand.legend(loc='upper right', fontsize='small', ncol=2)
        #ax_hand.grid(True)

        num_segments = int(np.ceil(total_timesteps//16))
        for k in range(1, num_segments):
            boundary = k * 16 - 0.5
            ax_arm.axvline(boundary, linestyle='--', color='red', alpha=0.5)
            ax_hand.axvline(boundary, linestyle='--', color='red', alpha=0.5)
        fig.tight_layout()
        plt.savefig(save_path, dpi=300)


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    """Create a single environment with appropriate wrappers."""
    # Create base environment
    env = gym.make(config.env_name, enable_render=True)
    # Add video recording wrapper if needed (only for the first environment)
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
        
    # Add multi-step wrapper
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_evaluation(
    env_name: str,
    host: str = "localhost",
    port: int = 5555,
    video_dir: Optional[str] = None,
    n_episodes: int = 2,
    n_envs: int = 1,
    n_action_steps: int = 2,
    max_episode_steps: int = 100,
) -> Tuple[str, List[bool]]:
    """
    Simple entry point to run a simulation evaluation.
    Args:
        env_name: Name of the environment to run
        host: Hostname of the inference server
        port: Port of the inference server
        video_dir: Directory to save videos (None for no videos)
        n_episodes: Number of episodes to run
        n_envs: Number of parallel environments
        n_action_steps: Number of action steps per environment step
        max_episode_steps: Maximum number of steps per episode
    Returns:
        Tuple of environment name and list of episode success flags
    """
    # Create configuration
    config = SimulationConfig(
        env_name=env_name,
        n_episodes=n_episodes,
        n_envs=n_envs,
        video=VideoConfig(video_dir=video_dir),
        multistep=MultiStepConfig(
            n_action_steps=n_action_steps, max_episode_steps=max_episode_steps
        ),
    )
    # Create client and run simulation
    client = SimulationInferenceClient(host=host, port=port)
    results = client.run_simulation(config)
    # Print results
    print(f"Results for {env_name}:")
    print(f"Success rate: {np.mean(results[1]):.2f}")
    return results


if __name__ == "__main__":
    # Example usage
    run_evaluation(
        env_name="robocasa_gr1_arms_only_fourier_hands/TwoArmPnPCarPartBrakepedal_GR1ArmsOnlyFourierHands_Env",
        host="localhost",
        port=5555,
        video_dir="./videos",
    )
