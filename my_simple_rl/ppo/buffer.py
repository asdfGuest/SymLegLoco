import torch as th
from typing import List, Dict

from my_simple_rl.ppo.config import PPOCfg


class PPOBuffer :

    data: Dict[str, List[th.Tensor]]
    
    # essential keys
    @property
    def reward(self) :
        return self.data['reward']
    @property
    def done(self) :
        return self.data['done']
    @property
    def value(self) :
        return self.data['value']
    # As advantage and returns will be computed in compute_gae function, push None for these key.
    @property
    def advantage(self) :
        return self.data['advantage']
    @property
    def returns(self) :
        return self.data['returns']
    
    def __init__(self, cfg:PPOCfg, keys:List[str]) :
        self.cfg = cfg
        self.data = {key:[] for key in keys}
    
    def clear(self) :
        for key in self.data :
            self.data[key].clear()
    
    def push(
            self,
            **kwargs
        ) :
        for key, value in kwargs.items() :
            self.data[key].append(value)
    
    def compute_gae(self, next_value:th.Tensor) :
        next_advantage = th.zeros_like(next_value)
        
        for idx in reversed(range(self.size)):
            not_done = (~self.done[idx]).to(th.float32)
            delta = self.reward[idx] + not_done * (self.cfg.gamma * next_value) - self.value[idx]
            self.advantage[idx] = delta + not_done * (self.cfg.gamma * self.cfg.gae_lambda) * next_advantage
            self.returns[idx] = self.advantage[idx] + self.value[idx]

            next_value = self.value[idx]
            next_advantage = self.advantage[idx]
    
    def get_minibatch_generator(self) :
        len_list = [len(tensor_list) for tensor_list in self.data.values()]
        if len(set(len_list)) != 1 :
            raise Exception()
        
        batch_list = [th.cat(tensor_list) for tensor_list in self.data.values()]

        batch_size = batch_list[0].shape[0]
        minibatch_size = batch_size // self.cfg.n_minibatch # drop last

        for _ in range(self.cfg.n_epoch) :
            rand_indices = th.randperm(batch_size, device=batch_list[0].device)
            batch_list = [batch[rand_indices] for batch in batch_list]
            
            for minibatch_idx in range(self.cfg.n_minibatch) :
                low = minibatch_size * minibatch_idx
                high = minibatch_size * (minibatch_idx + 1)
                yield [batch[low:high] for batch in batch_list]
    
    @property
    def size(self) :
        return len(self.data['advantage'])
