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

    def __init__(self, transpose):

        super().__init__()

        self.transpose = transpose

        self.conv1 = nn.Conv2d( 1, 16, 4, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4)
        self.rnn1 = nn.RNN(64, 64)

        if not self.transpose:
            self.fc1 = nn.Linear(64,  1)

        else:
            self.convT1 = nn.ConvTranspose2d(64, 32, 4)
            self.convT2 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
            self.convT3 = nn.ConvTranspose2d(16,  1, 4, 2, 1)

        self.h = torch.zeros(1, 1, 64)

    def reset(self):
        
        self.h = torch.zeros(1, 1, 64).to(self.device)

    def pre_setup_to_run(self):

        self.h.to(self.device)

    def pre_setup_to_save(self):

        self.h.to('cpu')

    def forward(self, x):

        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x, self.h = self.rnn1(x[None, :, :, 0, 0], self.h)

        if not self.transpose:

            x = torch.relu(self.fc1(x[0, :, :]))

        else:

            x = torch.relu(self.convT1(x[0, :, :, None, None]))
            x = torch.relu(self.convT2(x))
            x = torch.relu(self.convT3(x))
        
        return x