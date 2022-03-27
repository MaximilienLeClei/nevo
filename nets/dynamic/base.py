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
from typing import Any


class DynamicNetBase:
    """
    Dynamic Net Base class. Dynamic nets are Artificial Neural Networks of
    Dynamic Complexity. Subclasses need to be named *Net*.
    """
    def __init__(self):
        """
        Constructor.
        """
        if not hasattr(self, 'architectural_mutations'):
            raise NotImplementedError("Dynamic Nets require the attribute "
                                       "'architectural_mutations': a list "
                                       "of mutation functions.")

    def initialize_architecture(self) -> None:
        """
        Initialize the net's architecture to set it up for future mutations.
        Will be called once on the first generation. Can be either implemented 
        or left blank if this function is not desired.
        """
        pass

    def mutate_parameters(self) -> None:
        """
        Method to mutate the net's parameters. Will be called every iteration.
        Can be either implemented or left blank if this function is not
        desired.
        """
        pass

    def reset(self) -> None:
        """
        Method to reset the net's inner state. Can be either implemented or
        left blank if this function is not desired.
        """
        pass

    def setup_to_run(self) -> None:
        """
        Method to setup the net for it to then be ran. Can be either
        implemented or left blank if this function is not desired.
        """
        pass
        
    def setup_to_save(self) -> None:
        """
        Method to setup the net for it to be pickled (to a file or to be sent
        to another process). Can be either implemented or left blank if this
        function is not desired.
        """
        pass

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """
        Run net for one timestep given input 'x'.
        
        Args:
            x - Input value
        Returns:
            Any - Output value
        """
        raise NotImplementedError()