from dataclasses import dataclass
from simple_rl.utils.types import SchedulerT


@dataclass
class PPOCfg:
    n_rollout:int
    n_epoch:int
    n_minibatch:int

    gamma:float
    gae_lambda:float

    learning_rate:float
    desired_kl:float|None

    ratio_clip_param:SchedulerT|float
    value_clip_param:SchedulerT|float|None
    grad_norm_clip:float|None

    normalize_advantage:bool

    entropy_loss_coeff:SchedulerT|float
    value_loss_coeff:float

    internal_n_minibatch:int
