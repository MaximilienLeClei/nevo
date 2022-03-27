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

import numpy as np

from bots.network.base import NetworkBotBase


class DynamicNetworkBotBase(NetworkBotBase):
    """
    Dynamic Network Bot Base class. These bots are composed of networks
    of dynamic complexity. Concrete subclasses need to be named *Bot*.
    """

    def __init__(self,
                 args: argparse.Namespace,
                 rank: int,
                 pop_nb: int,
                 nb_pops: int):

        super().__init__(args, rank, pop_nb, nb_pops)

    def initialize_bot(self) -> None:
        """
        Set up the bot's architectural mutations
        """
        self.nets_architectural_mutations = sum(
            [net.architectural_mutations for net in self.nets], [])

        self.nets_architectures_initialized = False

    def mutate(self) -> None:
        """
        Mutation method for the dynamic bot. First initializes the nets'
        architectures. Then, every iteration, mutate network parameters and
        randomly select N mutations among all net mutations.
        """
        if not self.nets_architectures_initialized:

            for net in self.nets:
                net.initialize_architecture()
            
            self.nets_architectures_initialized = True

        else:

            for net in self.nets:
                net.mutate_parameters()

            np.random.choice(self.nets_architectural_mutations)()