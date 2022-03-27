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

import cv2
import numpy as np
import torch


def compute_padding(d_input):
  
    padding = ()
  
    for d in d_input[-1:0:-1]:

        if d == 1:
            padding += (1,1)
        elif d == 2:
            padding += (0,1)
        else:
            padding += (0,0)

    return padding

def neg(tup):
    return tuple(-x for x in tup)

def avg_pool(x, d):

    _, _, h, w = x.shape

    x = x.numpy()
    x = x[0]
    x = np.transpose(x, (1, 2, 0))
    x = cv2.resize(x, (h//d, w//d), interpolation=cv2.INTER_AREA)
    if x.ndim == 2:
        x = x[:, :, None]
    x = np.transpose(x, (2, 0, 1))
    x = x[None, :, :, :]
    x = torch.Tensor(x)

    return x

def torch_cat(x, i):

    for x_i in x:
        x_i = x_i.numpy()

    return torch.Tensor(np.concatenate(x, i))