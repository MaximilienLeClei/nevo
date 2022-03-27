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

from bots.network.dynamic.base import DynamicNetworkBotBase
from nets.dynamic.convolutional import Net as ConvolutionalNet
from nets.dynamic.recurrent import Net as RecurrentNet
from utils.functions.atari import get_action
from utils.functions.misc import grayscale_rescale_divide


class Bot(DynamicNetworkBotBase):

    def initialize_nets(self):

        self.d_input = [1, 64, 64]

        if self.pop_nb == 0:
            self.function = 'generator'
        else: # self.pop_nb == 1:
            self.function = 'discriminator'
            
        self.convolutional_net = ConvolutionalNet(self.d_input)

        if ('merge' in self.args.additional_arguments and
            self.args.additional_arguments['merge'] == 'yes'):

            self.recurrent_net = RecurrentNet(self.convolutional_net, 6)

        else:

            if self.function == 'generator':
                # FIRE, UP, DOWN, LEFT, RIGHT
                self.recurrent_net = RecurrentNet(self.convolutional_net, 5)
            else: # self.function == 'discriminator':
                self.recurrent_net = RecurrentNet(self.convolutional_net, 1)

        self.convolutional_net.output_net = self.recurrent_net

        self.nets = [self.convolutional_net, self.recurrent_net]
            
    def __call__(self, x):

        x = self.env_to_convolutional_net(x)
        x = self.convolutional_net(x)
        x = self.convolutional_net_to_recurrent_net(x)
        x = self.recurrent_net(x)
        x = self.recurrent_net_to_env(x)
    
        return x

    def env_to_convolutional_net(self, x):

        x = grayscale_rescale_divide(x[3:], 64)
        x = x[None,None,:,:]    
        x = torch.Tensor(x)

        return x

    def convolutional_net_to_recurrent_net(self, x):

        x = np.array(torch.Tensor(x))

        return x

    def recurrent_net_to_env(self, x):

        x = np.array(x).squeeze()
        
        if ('merge' in self.args.additional_arguments and
            self.args.additional_arguments['merge'] == 'yes'):
            x = x[:-1] if self.function == 'generator' else x[-1] # discrim

        if self.function == 'generator':
            x = x > np.random.uniform(size=x.shape)
            x = get_action(x)
        else: # self.function == 'discriminator':
            x = np.minimum(x, 1)

        return x