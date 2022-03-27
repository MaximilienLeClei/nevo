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
from nets.static.conv_rnn import Net
from utils.functions.atari import get_action
from utils.functions.misc import grayscale_rescale_divide


class Bot(StaticNetworkBotBase):

    def initialize_nets(self):

        if self.pop_nb == 0:
            self.function = 'generator'
        else: # self.pop_nb == 1:
            self.function = 'discriminator'

        if ('merge' in self.args.additional_arguments and
            self.args.additional_arguments['merge'] == 'yes'):

            self.net = Net(6)

        else:

            if self.function == 'generator':
                self.net = Net(5)
            else: # self.function == 'discriminator':
                self.net = Net(1)

        self.nets = [self.net]

        self.sigma = 0.001
    
    def __call__(self, x):
    
        x = self.env_to_net(x)
        x = self.net(x)
        x = self.net_to_env(x)

        return x

    def env_to_net(self, x):
        
        x = grayscale_rescale_divide(x[3:], 64)
        x = x[None,None,:,:]
        x = torch.Tensor(x)
        x = x.to(self.device)

        return x

    def net_to_env(self, x):

        x = x.to('cpu')
        x = x.numpy().squeeze()

        if ('merge' in self.args.additional_arguments and
            self.args.additional_arguments['merge'] == 'yes'):
            x = x[:-1] if self.function == 'generator' else x[-1] # discrim

        if self.function == 'generator':
            x = x > np.random.uniform(size=x.shape)
            x = get_action(x)
        else: # self.function == 'discriminator':
            x = np.minimum(x, 1)

        return x