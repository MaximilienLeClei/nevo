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

from abc import abstractmethod
import argparse
from typing import Any

import numpy as np
import torch
import random


class BotBase:
    """
    Bot Base class. Bots are artificial agents that get mutated and produce
    behaviour in environments. Concrete subclasses need to be named *Bot*.
    """
    
    def __init__(self,
                 args: argparse.Namespace,
                 rank: int,
                 pop_nb: int,
                 nb_pops: int):
        """
        Constructor.

        Args:
            args - Experiment specific arguments (obtained through argparse).
            rank - MPI rank of process.
            size - Number of MPI processes.
            io_path - Path to the environment's IO class.
            nb_pops - Total number of populations.
        """
        self.args = args
        self.rank = rank
        self.pop_nb = pop_nb
        self.nb_pops = nb_pops

    def initialize(self) -> None:
        """
        Initialize the bot before it starts getting built. Can be implemented
        or left blank if this function is not desired.
        """
        pass

    def build(self, seeds: np.ndarray) -> None:
        """
        Build the bot from scratch (N mutations) using its full list of seeds.
        This method is only called on generation 1 for 'ps_p2p' and
        'big_ps_p2p' but called every generation for the 'ps' protocol.

        Args:
            seeds - array of integers with which to seed mutation randomness.
        """
        self.initialize()

        for seed in seeds:
            self.extend(seed)

    def extend(self, seed: np.ndarray) -> None:
        """
        Extend the bot (1 mutation) based on a newly generated seed.
        Called every generation for the 'ps_p2p' and 'big_ps_p2p' protocols.

        Args:
            seeds - array of integers with which to seed mutation randomness.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        
        if seed > 0:
            self.mutate()

    @abstractmethod
    def mutate(self) -> None:
        """
        Method to mutate the bot. Should be implemented.
        """
        raise NotImplementedError()

    @abstractmethod
    def setup_to_run(self) -> None:
        """
        Setup the bot to run in an environment. Should be implemented.
        """
        raise NotImplementedError()
        
    @abstractmethod
    def setup_to_save(self) -> None:
        """
        Setup the bot to be saved (pickled to a file or to be sent to
        another process). Should be implemented.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the bot. Should be implemented.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, x) -> Any:
        """
        Run bot for one timestep given input 'x'. Should be implemented.

        Args:
            x - Input
        Returns:
            Any - Output
        """
        raise NotImplementedError()