from my_simple_rl.env.env import BaseEnv

import torch as th
from typing import Tuple


class LeggedEnvWrapper(BaseEnv) :
    def __init__(self, env, reward_scale:float=1.0) :
        self.device = env.device
        self.n_env = env.n_env
        self.n_action = env.n_action
        self.n_obs = env.n_obs
        self.n_history = env.n_history
        self.n_privileged_obs = env.n_privileged_obs
        self.reward_scale = reward_scale
        self.env = env

    def reset(self) -> Tuple[Tuple[th.Tensor,th.Tensor,th.Tensor], dict]:
        obs_pack, info = self.env.reset()
        obs_pack = obs_pack['policy']
        return obs_pack, info

    def step(self, action) -> Tuple[Tuple[th.Tensor,th.Tensor,th.Tensor], th.Tensor, th.Tensor, th.Tensor, dict]:
        obs_pack, reward, terminated, truncated, info = self.env.step(action)
        obs_pack = obs_pack['policy']
        reward = reward.view(self.n_env,1) * self.reward_scale
        terminated = terminated.view(self.n_env,1)
        truncated = truncated.view(self.n_env,1)
        return obs_pack, reward, terminated, truncated, info

    def close(self) :
        return self.env.close()
