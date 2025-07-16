# register_lift.py
import gymnasium as gym
from gymnasium import Env
import robosuite as suite
from robosuite.wrappers import GymWrapper

class LiftCubeEnv(Env):
    def __init__(self, enable_render=True):
        rs = suite.make(
            "Lift",
            robots="GR1ArmsOnly",
            use_camera_obs=True,
            has_renderer=False,
            has_offscreen_renderer=True,
            reward_shaping=True,
            control_freq=20,
            camera_names="robot0_robotview",
            camera_heights=256,
            camera_widths=256,
            horizon=16,
        )
        self.env = GymWrapper(rs, flatten_obs=False)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def close(self):
        self.env.close()

print("TRYING TO REGISTER ENV")
gym.register(
    id="LiftCubeEnv-v0",
    entry_point="gr00t.eval.register_my_lift_env:LiftCubeEnv",
    max_episode_steps=16,
    disable_env_checker=True,
)
