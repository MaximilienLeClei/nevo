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

import torch.nn as nn


class StaticNetBase(nn.Module):
    """
    Static Net Base class.
    Static Net objects are PyTorch-based Deep Neural Networks.
    Subclasses need to be named *Net*.
    """
    def __init__(self):
        
        super().__init__()
        
        self.device = None
        
    def reset(self) -> None:
        """
        Method to reset the static net's inner state. Can be either
        implemented or left blank if this function is not desired.
        """
        pass

    def setup_to_run(self) -> None:
        """
        Method to setup the net for it to then be ran.
        """
        self.pre_setup_to_run()

        self.to(self.device)

    def pre_setup_to_run(self) -> None:
        """
        Placeholder method that can be used for user-defined pre-run setup.
        Can be implemented to send tensors to the device among other things.
        """
        pass
        
    def setup_to_save(self) -> None:
        """
        Method to setup the net for it to be pickled (to a file or to be sent
        to another process).
        """
        self.pre_setup_to_save()
        
        self.to('cpu')

    def pre_setup_to_save(self) -> None:
        """
        Placeholder method that can be used for user-defined pre-run setup.
        Can be implemented to send tensors back to the CPU for saving among
        other things.
        """
        pass