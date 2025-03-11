from simple_rl.modules.layers import MLP

import torch as th
from typing import List, Type


class MLPEstimator(th.nn.Module) :
    def __init__(self, input_dim:int, output_dim:int, net_arch:List[int], activ_fn:Type[th.nn.Module]) :
        super().__init__()
        self.mlp = MLP(
            net_arch=[input_dim] + net_arch + [output_dim],
            activ_fn=activ_fn,
        )    
    def forward(self, x:th.Tensor) :
        return self.mlp(x)
