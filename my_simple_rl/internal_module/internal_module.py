from my_simple_rl.sym_utils import decompose_observation

import torch as th
from typing import Tuple


class InternalModule :
    def __init__(
            self,
            estimator:th.nn.Module,
            learning_rate:float,
            device:th.device,
            estimator_free:bool=False,
        ) :
        self.estimator = estimator
        self.learning_rate = learning_rate
        self.device = device
        self.estimator_free = estimator_free

        self.estimator.to(self.device)

        self.optimizer = th.optim.Adam(
            params=self.estimator.parameters(),
            lr=self.learning_rate
        )
    
    def state_dict(self) :
        return {
            'estimator' : self.estimator.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'learning_rate' : self.learning_rate,
        }
    
    def load_state_dict(self, state_dict:dict) :
        self.estimator.load_state_dict(state_dict['estimator'])
        self.learning_rate = state_dict['learning_rate']
        
        self.optimizer = th.optim.Adam(
            params=self.estimator.parameters(),
            lr=self.learning_rate
        )
        self.optimizer.load_state_dict(state_dict['optimizer'])
    
    def compute_state(self, obs_pack:Tuple[th.Tensor, th.Tensor]) :
        '''
        obs(obs_pack[0]) : (n_env, n_history, n_obs)
        privileged_obs(obs_pack[1]) : (n_env, n_privileged_obs)
        '''
        obs, privileged_obs = obs_pack

        if not self.estimator_free :
            with th.no_grad() :
                privileged_obs_hat = self.estimator(obs.flatten(1))
        else :
            privileged_obs_hat = privileged_obs
        
        dec_obs, dec_privileged_obs = decompose_observation(obs, privileged_obs_hat)
        state_l = th.cat([dec_obs[0].flatten(1), dec_privileged_obs[0]], dim=-1)
        state_r = th.cat([dec_obs[1].flatten(1), dec_privileged_obs[1]], dim=-1)

        state = th.cat([state_l[:,None,:], state_r[:,None,:]], dim=1)
        return state
    
    def update(self, obs_pack:Tuple[th.Tensor, th.Tensor]) :
        '''
        obs(obs_pack[0]) : (n_env, n_history, n_obs)
        privileged_obs(obs_pack[1]) : (n_env, n_privileged_obs)
        '''
        if self.estimator_free :
            return {}
        
        obs, privileged_obs = obs_pack
        info = {}

        privileged_obs_hat = self.estimator(obs.flatten(1))
        loss = th.sum(th.square(privileged_obs - privileged_obs_hat), dim=-1).mean()
        
        info['internal_module/loss'] = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return info
