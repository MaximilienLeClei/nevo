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

from envs.base import EnvBase


class MultistepEnvBase(EnvBase):
    """
    Multistep Env Base class. Concrete subclasses need to be named *Env* and
    create the attribute
    """
    def __init__(self,
                 args: argparse.Namespace,
                 rank: int,
                 size: int,
                 io_path: str,
                 nb_pops: int):

        if not hasattr(self, 'valid_tasks'):
            raise NotImplementedError("Multistep Environments require the "
                                       "attribute 'valid_tasks': a function "
                                       "returning a list of all valid tasks.")

        if 'task' not in args.additional_arguments:
            raise Exception("Additional argument 'task' missing. It needs to "
                            "be chosen from one of " + str(self.valid_tasks))
        elif args.additional_arguments['task'] not in self.valid_tasks:
            raise Exception("Additional argument 'task' invalid. It needs to "
                            "be chosen from one of " + str(self.valid_tasks))

        if 'steps' not in args.additional_arguments:
            args.additional_arguments['steps'] = 150
        elif not isinstance(args.additional_arguments['steps'], int):
            raise Exception("Additional argument 'steps' is of wrong type. "
                            "It needs to be an integer >= 0.")
        elif args.additional_arguments['steps'] < 0: # 0 : infinite
            raise Exception("Additional argument 'steps' needs to be >= 0.")

        transfer_options = ['no', 'yes', 'fit']
        if 'transfer' not in args.additional_arguments:
            args.additional_arguments['transfer'] = 'no'
        elif args.additional_arguments['transfer'] not in transfer_options:
            raise Exception("Additional argument 'transfer' invalid. It "
                            "needs be chosen from one of " +
                             str(transfer_options))

        super().__init__(args, rank, size, io_path, nb_pops)