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

from bots.network.base import NetworkBotBase


class StaticNetworkBotBase(NetworkBotBase):
    """
    Static Network Bot Base class. These bot contains one static-sized
    PyTorch Neural Networks. Concrete subclasses need to be named *Bot*.
    """

    def initialize_bot(self) -> None:
        """
        Links up the bot's net to either CPU or GPU. If GPUs are available,
        nets are equally distributed accross GPUs, otherwise they are set up
        on CPUs.
        """
        if self.args.enable_gpu_use and torch.cuda.device_count() > 0:
            self.device = 'cuda:' + str(self.rank % torch.cuda.device_count())
        else:
            self.device = 'cpu'

        for net in self.nets:

            net.device = self.device

            for parameter in net.parameters():

                parameter.requires_grad = False
                parameter.data = torch.zeros_like(parameter.data)

    def mutate(self) -> None:
        """
        Mutation method for the static bot.
        Mutates the networks' parameters.
        """
        if not hasattr(self, 'sigma'):
            self.sigma = 0.01

        for net in self.nets:

            for parameter in net.parameters():
                parameter.data += \
                    self.sigma * torch.randn_like(parameter.data)