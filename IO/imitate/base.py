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

from IO.base import IOBase


class TargetBase:
    """
    Target class. Targets are behaviour/models to imitate.
    Concrete subclasses need to be named *Target*.
    """

    @abstractmethod
    def reset(self, y: int, z: int) -> None:
        """
        Reset the target's state

        Args:
            y - seed.
            z - number of steps.
        """
        raise NotImplementedError()

    @abstractmethod
    def __call__(self, x: Any) -> Any:
        """
        Reset the target's state

        Args:
            x - Input value.
        Returns:
            Any - Output value.
        """
        raise NotImplementedError()

class ImitateIOBase(IOBase):

    @abstractmethod
    def load_target(self) -> TargetBase:
        """
        Load a target to imitate.

        Returns:
            TargetBase - The target.
        """
        raise NotImplementedError()