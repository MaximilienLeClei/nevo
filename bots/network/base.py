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

from bots.base import BotBase


class NetworkBotBase(BotBase):
    """
    Network Bot Base class. Network Bots contain Nets (Artificial Neural
    Networks). Concrete subclasses need to be named *Bot*.
    """
    
    def initialize_nets(self) -> None:
        """
        Initialize the bot's nets. Should be implemented to create a list
        containing the bot's nets as such :
        % self.nets = [self.net_0, self.net_1, ..., self.net_n] %
        """
        if not hasattr(self, 'nets'):
            raise NotImplementedError("Network Bots require the attribute "
                                      "'nets': a list of networks.")

    def initialize_bot(self) -> None:
        """
        Initialize the bot independently of its networks. Can be overriden.
        """
        pass

    def initialize_run_variables(self) -> None:
        """
        Initialize various run variables for use across transfer options.
        """
        self.state = None
        self.observation = None
        self.seed = 0
        self.steps = 0

        self.continual_fitness = 0
        self.episode_score = 0
        self.run_score = 0

    def initialize(self) -> None:
        """
        Initialize the bot before it starts getting built.
        """
        self.initialize_nets()
        self.initialize_bot()
        self.initialize_run_variables()
        
    def setup_to_run(self) -> None:
        """
        Setup the bot and its nets to then run in an environment.
        """
        self._setup_to_run()

        for net in self.nets:
            net.setup_to_run()

    def _setup_to_run(self) -> None:
        """
        Additional pre-run setup, can be implemented.
        """
        pass
        
    def setup_to_save(self) -> None:
        """
        Setup the bot and its nets to then be pickled (to a file or to be sent
        to another process).
        """
        self._setup_to_save()

        for net in self.nets:
            net.setup_to_save()

    def _setup_to_save(self) -> None:
        """
        Additional pre-save setup, can be implemented.
        """
        pass

    def reset(self) -> None:
        """
        Reset the bot and its nets' inner states.
        """
        self._reset()

        for net in self.nets:
            net.reset()

    def _reset(self) -> None:
        """
        Additional reset functionnality, can be implemented.
        """
        pass