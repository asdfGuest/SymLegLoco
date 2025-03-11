import torch as th
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple


@dataclass
class EnvSpec :
    device: th.device
    n_env: int
    n_action: int
    n_obs: int
    n_history: int
    n_privileged_obs: int


class BaseEnv(ABC) :
    device: th.device
    n_env: int
    n_action: int
    n_obs: int
    n_history: int
    n_privileged_obs: int

    @abstractmethod
    def reset(self) -> Tuple[Tuple[th.Tensor], dict]:
        pass

    @abstractmethod
    def step(self, action:th.Tensor) -> Tuple[Tuple[th.Tensor], th.Tensor, th.Tensor, th.Tensor, dict]:
        pass

    @abstractmethod
    def close(self) :
        pass

    @property
    def spec(self) :
        return EnvSpec(
            device=self.device,
            n_env=self.n_env,
            n_action=self.n_action,
            n_obs=self.n_obs,
            n_history=self.n_history,
            n_privileged_obs=self.n_privileged_obs,
        )
