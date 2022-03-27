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

from bots.network.dynamic.base import DynamicNetworkBotBase
from nets.dynamic.recurrent import Net
from utils.functions.control import get_info, standardize


class Bot(DynamicNetworkBotBase):

    def initialize_nets(self):

        if self.pop_nb == 0:
            self.function = 'generator'
        else: # self.pop_nb == 1:
            self.function = 'discriminator'

        info = get_info(self.args.additional_arguments['task'])
        d_input, d_output, self.discrete_output, self.output_range = info

        if ('merge' in self.args.additional_arguments and
            self.args.additional_arguments['merge'] == 'yes'):

            self.net = Net(d_input, d_output+1)

        else:

            if self.function == 'generator':
                self.net = Net(d_input, d_output)
            else: # self.function == 'discriminator':
                self.net = Net(d_input, 1)

        self.nets = [self.net]

        self.mean = self.v = self.std = self.n = 0

        self.mutate_nb_architectural_mutations = False
    
    def __call__(self, x):
    
        x = self.env_to_net(x)
        x = self.net(x)
        x = self.net_to_env(x)

        return x

    def update_mean_std(self, x):

        temp_m = self.mean + (x - self.mean) / self.n
        temp_v = self.v + (x - self.mean) * (x - temp_m)

        self.v = temp_v
        self.mean = temp_m
        self.std = np.sqrt(self.v / self.n)

    def env_to_net(self, x):
        
        if standardize(self.args.env_path,
                       self.args.additional_arguments['task']):
            self.n += 1
            self.update_mean_std(x)
            x = (x - self.mean) / (self.std + (self.std == 0))

        return x

    def net_to_env(self, x):
        
        x = np.array(x).squeeze(axis=1)
        
        if ('merge' in self.args.additional_arguments and
            self.args.additional_arguments['merge'] == 'yes'):
            x = x[:-1] if self.function == 'generator' else x[-1] # discrim

        # Backward compatibility
        if not hasattr(self, 'function') or self.function == 'generator':

            if self.discrete_output:
                x = np.argmax(x)
            else:
                x = np.minimum(x, 1)
                x = x * (self.output_range * 2) - self.output_range

        else: # self.function == 'discriminator':

            x = x.squeeze()
            x = np.minimum(x, 1)
            
        return x