# Copyright 2022 Maximilien Le Clei.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch

from bots.network.static.base import StaticNetworkBotBase
from nets.static.conv_rnn_convT import Net


class Bot(StaticNetworkBotBase):

    def initialize_nets(self):

        if self.pop_nb == 0:
            self.function = 'generator'
            self.net = Net(transpose=True)
        else: # self.pop_nb == 1:
            self.function = 'discriminator'
            self.net = Net(transpose=False)

        self.nets = [self.net]

        self.sigma = 0.001
    
    def __call__(self, x):
    
        x = self.env_to_net(x)
        x = self.net(x)
        x = self.net_to_env(x)

        return x

    def env_to_net(self, x):

        if isinstance(x, np.ndarray):

            x = x[None, None, :, :]
            x = torch.Tensor(x)

        return x

    def net_to_env(self, x):

        if self.function == 'generator':
            x = torch.clamp(x, 0, 1)
            
        else: # self.function == 'discriminator':
            x = x.numpy().squeeze()
            x = np.minimum(x, 1)
            
        return x