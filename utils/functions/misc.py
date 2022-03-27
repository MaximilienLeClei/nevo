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

import argparse
from importlib import import_module
import json
import os
from typing import Any

import cv2
import numpy as np


def initialize_environment(args: argparse.Namespace,
                           rank: int,
                           size: int) -> object:

    if os.path.splitext(args.additional_arguments)[1] == ".json":

        with open(args.additional_arguments) as json_file:
            args.additional_arguments = json.load(json_file)

    else:
        
        args.additional_arguments = json.loads(args.additional_arguments)

    return getattr(
        import_module(args.env_path.replace('/', '.').replace('.py', '')),
        'Env')(args, rank, size)

def deterministic_set(x: list) -> list:

    set = list(dict.fromkeys(x))

    return set

def find_sublist_index(element: Any, layered_list: list) -> int:

    for sublist_index, sublist in enumerate(layered_list):
        if element in sublist:
            return sublist_index

def grayscale_rescale_divide(x: np.ndarray, d_in: int) -> np.ndarray:

    x = x.squeeze()

    if x.ndim == 3:
        x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

    x = cv2.resize(x, (d_in, d_in), interpolation=cv2.INTER_AREA)

    x = x.astype(np.float32)

    x = x / 255
    
    return x