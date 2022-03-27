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


class Bot(DynamicNetworkBotBase):

    def initialize_nets(self):

        self.d_input = [1, 16, 16]

        if self.pop_nb == 0:
            self.function = 'generator'
        else: # self.pop_nb == 1:
            self.function = 'discriminator'

        self.convolutional_net = ConvolutionalNet(self.d_input, 'forward')

        if self.function == 'generator':
            self.transpose_convolution_net = ConvolutionalNet(
                self.d_input, 'backward')

        if self.function == 'discriminator':
            self.recurrent_net = RecurrentNet(self.convolutional_net, 1)
        else: # self.function == 'generator':
            self.recurrent_net = RecurrentNet(
                self.convolutional_net, self.transpose_convolution_net)

        self.convolutional_net.output_net = self.recurrent_net
        if self.function == 'generator':
            self.transpose_convolution_net.output_net = self.recurrent_net

        self.nets = [self.convolutional_net, self.recurrent_net]
        if self.function == 'generator':
            self.nets.append(self.transpose_convolution_net)

    def __call__(self, x):

        x = self.env_to_convolutional_net(x)
        x = self.convolutional_net(x)
        x = self.convolutional_net_to_recurrent_net(x)
        x = self.recurrent_net(x)

        if self.function == 'generator':

            x = self.recurrent_net_to_transpose_convolution_net(x)
            x = self.transpose_convolution_net(x)
            x = self.transpose_convolution_net_to_env(x)

        else: # self.function == 'discriminator':
            
            x = self.recurrent_net_to_env(x)
    
        return x
            
    def env_to_convolutional_net(self, x):

        if isinstance(x, np.ndarray):

            x = x[None,None,:,:]
            x = torch.Tensor(x)

        return x

    def convolutional_net_to_recurrent_net(self, x):

        x = np.array(torch.Tensor(x))

        return x

    def recurrent_net_to_env(self, x):

        x = np.array(x).squeeze()

        x = np.minimum(x, 1)

        return x

    def recurrent_net_to_transpose_convolution_net(self, x):

        for i in range(len(x)):
            x[i] = torch.Tensor(x[i][None,None,None,:])

        return x

    def transpose_convolution_net_to_env(self, x):

        x = np.array(x).squeeze()
        
        x = np.minimum(x, 1)

        return x