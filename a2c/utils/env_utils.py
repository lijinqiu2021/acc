import os.path

import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper


class CustomObsWrapper(FullyObsWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        obs_image_space = env.observation_space["image"]
        super().__init__(env)
        state_image_space = self.observation_space["image"]
        self.observation_space = spaces.Dict(
            {
                "obs_image": obs_image_space,
                "state_image": state_image_space,
            }
        )

    def observation(self, obs):
        obs_image = obs["image"]
        obs = super().observation(obs)
        state_image = obs["image"]
        return {"obs_image": obs_image, "state_image": state_image}


def make_env(
    env_key,
    seed=None,
    idx=None,
    capture_video=False,
    log_dir=None,
    render_mode=None,
):
    def thunk():
        render_mode_ = "rgb_array" if capture_video else render_mode
        env = gym.make(env_key, render_mode=render_mode_)
        env = CustomObsWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, os.path.join(log_dir, "videos"))
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk
