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

import torch
import torch.nn as nn

from nets.static.base import StaticNetBase

class Net(StaticNetBase):

    def __init__(self, dimensions, recurrent=True, batch_size=1):
    
        super().__init__()

        # [d_input, d_hidden_0, ..., d_hidden_n, d_output]
        self.dimensions = torch.tensor(dimensions)
        
        # reucrrence on hidden_n layer (output layer if no hidden layer)
        self.recurrent = recurrent

        self.fc = nn.ModuleList()
        
        for i, _ in enumerate(dimensions[:-1]):

            if ((len(dimensions) == 2 or i+3 == len(dimensions))
                and recurrent == True):
                self.rnn = nn.RNN(dimensions[i], dimensions[i+1])
                self.h = torch.zeros(1, batch_size, dimensions[i+1])
            else:
                self.fc.append(nn.Linear(dimensions[i], dimensions[i+1]))

    def reset(self):

        if self.recurrent:
            self.h *= 0

    def forward(self, x):

        for i, _ in enumerate(self.dimensions[:-1]):

            if self.recurrent == True:

                if i+3 == len(self.dimensions) or len(self.dimensions) == 2:
                    x, self.h = self.rnn(x[None, :], self.h)

                elif i+2 == len(self.dimensions):
                    x = torch.relu(self.fc[-1](x[0,:]))

                else:
                    x = torch.relu(self.fc[i](x))

            else:
                
                x = torch.relu(self.fc[i](x))
        
        return x