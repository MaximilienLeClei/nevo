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

    def __init__(self, dimensions):
    
        super().__init__()
        self.dimensions = dimensions
        self.fc = nn.ModuleList()

        for i, _ in enumerate(dimensions[:-1]):
            self.fc.append(nn.Linear(dimensions[i], dimensions[i+1]))

    def forward(self, x):

        for i, _ in enumerate(self.dimensions[:-1]):
            x = torch.relu(self.fc[i](x))

        return x