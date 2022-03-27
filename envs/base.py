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

from abc import ABC, abstractmethod
import argparse
from importlib import import_module
from typing import Union

import numpy as np


class EnvBase(ABC):
    """
    Env Base class. Environments are virtual playgrounds for bots to evolve
    and produce behaviour. One environment makes interact, at a given time, as
    many bots as there are populations.
    Concrete subclasses need to be named *Env*.
    """
    def __init__(self,
                 args: argparse.Namespace,
                 rank: int,
                 size: int,
                 io_path: str,
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
        self.size = size
        self.io_path = io_path
        self.nb_pops = nb_pops

        self.initialize_io()
        self.initialize_bots()

    def initialize_io(self) -> None:
        """
        Called upon object initialization.
        Initializes an IO object.
        IO objects handle file input/output.
        """
        self.io = getattr(import_module(self.io_path), 'IO')(self.args,
                                                             self.rank,
                                                             self.size,
                                                             self.nb_pops)

    def initialize_bots(self) -> None:
        """
        Called upon object initialization.
        Initializes bots.
        Bots evolve and produce behaviour in the environment.
        """
        bot_path = self.args.bots_path.replace('/','.').replace('.py', '')

        self.bots = []

        for pop_nb in range(self.nb_pops):
            self.bots.append(
                getattr(import_module(bot_path), 'Bot')(self.args,
                                                        self.rank,
                                                        pop_nb,
                                                        self.nb_pops))
        
    def build_bots(self, seeds: np.ndarray) -> None:
        """
        Build bots from scratch with the full list of seeds.

        Args:
            seeds - array of integers with which to seed mutation randomness.
        """
        for pop_nb in range(self.nb_pops):
            self.bots[pop_nb].build(seeds[pop_nb])

    def extend_bots(self, seeds: np.ndarray) -> None:
        """
        Extend bots with their latest seed.

        Args:
            seeds - array of integers with which to seed mutation randomness.
        """
        for pop_nb in range(self.nb_pops):
            self.bots[pop_nb].extend(seeds[pop_nb])

    def setup_to_run(self) -> None:
        """
        Setup bots to later run them within the environment.
        """
        for bot in self.bots:
            bot.setup_to_run()

    def setup_to_save(self) -> None:
        """
        Setup bots to then later save them (pickling to a file or pickling to
        be sent to another process).
        """
        for bot in self.bots:
            bot.setup_to_save()

    def evaluate_bots(self, gen_nb: int) -> np.ndarray:
        """
        Method called once per iteration in order to evaluate and attribute
        fitnesses to bots. The added random jitter to fitnesses is to keep
        reproducibility accross communication protocols.

        Args:
            gen_nb - Generation number.
        Returns:
            np.ndarray - Numpy array of fitnesses.
        """
        self.setup_to_run()

        fitnesses = self.run(gen_nb) + np.random.rand(len(self.bots)) * 0.0001

        self.setup_to_save()
        
        return np.array(fitnesses, dtype=np.float32)

    @abstractmethod
    def run(self, gen_nb: int) -> Union[float, list, np.ndarray]:
        """
        Inner method of *evaluate_bots*.

        Args:
            gen_nb - Generation number.
        Returns:
            Union[float, list, np.ndarray] - List / Numpy array of fitnesses.
        """
        raise NotImplementedError()