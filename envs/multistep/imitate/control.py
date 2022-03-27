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

import gym

from envs.multistep.imitate.base import ImitateMultistepEnvBase
from utils.functions.control import get_task_name, get_all_potential_imitation_tasks
from utils.functions.control import seed


class Env(ImitateMultistepEnvBase):

    def __init__(self, args, rank, size):

        args.additional_arguments['steps'] = 0

        self.valid_tasks = get_all_potential_imitation_tasks()
        self.hide_score =  lambda x,_ : x
        self.seed, self.set_state, self.get_state = seed, None, None

        super().__init__(args, rank, size, io_path='IO.imitate.sb3')

        for key in args.additional_arguments:
            if key not in ['task', 'steps', 'transfer', 'merge']:
                raise Exception("Additional argument", key, "not supported.")

        self.target = self.io.load_target()

        self.emulators = [gym.make(
            get_task_name(args.additional_arguments['task']))]