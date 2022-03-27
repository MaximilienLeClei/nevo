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

from envs.multistep.score.base import ScoreMultistepEnvBase
from utils.functions.atari import get_score_tasks, get_task_name, wrap
from utils.functions.atari import seed, get_state, set_state


class Env(ScoreMultistepEnvBase):

    def __init__(self, args, rank, size):

        self.valid_tasks = get_score_tasks()
        self.seed, self.get_state, self.set_state = seed, get_state, set_state

        super().__init__(args, rank, size)

        for key in self.args.additional_arguments:
            if key not in ['task', 'steps', 'transfer', 'trials']:
                raise Exception("Additional argument", key, "not supported.")

        task_name = get_task_name(args.additional_arguments['task'])

        self.emulator = wrap(gym.make(id=task_name,
                                      frameskip=1,
                                      repeat_action_probability=0))