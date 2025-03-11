import torch as th
import math

from typing import List, Type

from simple_rl.modules.layers import MLP
from simple_rl.modules.distributions import DiagonalGaussian
from simple_rl.modules.modules import BasePolicy, BaseValue, BaseActorCritic

from my_simple_rl.sym_utils import merge_action


class MlpPolicy(BasePolicy) :
    def __init__(
            self,
            n_obs:int,
            n_action:int,
            init_std:float,
            net_arch:List[int],
            activ_fn:Type[th.nn.Module],
            is_sym:bool=False,
        ):
        super().__init__()
        self.n_obs = n_obs
        self.n_action = n_action
        self.is_sym = is_sym

        net_arch = [n_obs] + net_arch + [n_action // (2 if self.is_sym else 1)]
        self.mean = MLP(net_arch, activ_fn)
        self.logstd = th.nn.Parameter(th.full(size=(1,self.n_action), fill_value=math.log(init_std)))

    def compute(self, state:th.Tensor) :
        state_l, state_r = state[:,0,:], state[:,1,:]
        if self.is_sym :
            mean = merge_action(
                self.mean(state_l),
                self.mean(state_r)
            )
        else :
            mean = self.mean(state_l)
        std = th.exp(self.logstd).expand_as(mean)
        return DiagonalGaussian(mean, std)
    
    def get_pdf_cls(self):
        return DiagonalGaussian


class MlpValue(BaseValue) :
    def __init__(
            self,
            n_obs:int,
            net_arch:List[int],
            activ_fn:Type[th.nn.Module],
            is_sym:bool=False,
        ):
        super().__init__()
        self.is_sym = is_sym

        net_arch = [n_obs] + net_arch + [1]
        self.mlp = MLP(net_arch, activ_fn)
    
    def compute(self, state:th.Tensor):
        state_l = state[:,0,:]
        return self.mlp(state_l)


class MlpActorCritic(BaseActorCritic) :
    def __init__(
            self,
            n_obs:int,
            n_action:int,
            init_std:float,
            net_arch:List[int],
            activ_fn:Type[th.nn.Module],
            is_sym:bool=False,
        ):
        super().__init__(
            policy=MlpPolicy(n_obs, n_action, init_std, net_arch, activ_fn, is_sym),
            value=MlpValue(n_obs, net_arch, activ_fn, is_sym)
        )
    
    def mean_std(self) :
        return th.detach(self.policy.logstd).exp().mean().item()
