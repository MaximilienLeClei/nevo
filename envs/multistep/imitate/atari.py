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

import ale_py
import gym

from envs.multistep.imitate.base import ImitateMultistepEnvBase
from utils.functions.atari import get_imitation_tasks, hide_score, wrap
from utils.functions.atari import get_task_name
from utils.functions.atari import seed, set_state, get_state


class Env(ImitateMultistepEnvBase):

    def __init__(self, args, rank, size):

        self.valid_tasks, self.hide_score = get_imitation_tasks(), hide_score
        self.seed, self.set_state, self.get_state = seed, set_state, get_state

        super().__init__(args, rank, size, io_path='IO.imitate.sb3')

        for key in args.additional_arguments:
            if key not in ['task', 'steps', 'transfer', 'merge']:
                raise Exception("Additional argument", key, "not supported.")
            
        self.target = self.io.load_target()

        self.emulators = [
            wrap(gym.make(
                id=get_task_name(args.additional_arguments['task']),
                frameskip = 1,
                repeat_action_probability = 0)),
            wrap(gym.make(
                id=get_task_name(args.additional_arguments['task']),
                frameskip=1,
                repeat_action_probability=0,
                full_action_space = False))]