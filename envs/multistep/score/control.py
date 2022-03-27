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

from envs.multistep.score.base import ScoreMultistepEnvBase
from utils.functions.control import get_score_tasks, get_task_name
from utils.functions.control import seed


class Env(ScoreMultistepEnvBase):

    def __init__(self, args, rank, size):

        args.additional_arguments['steps'] = 0

        self.valid_tasks = get_score_tasks()
        self.seed = seed
        self.get_state, self.set_state = lambda: None, lambda: None

        super().__init__(args, rank, size)

        for key in args.additional_arguments:
            if key not in ['task', 'steps', 'transfer', 'trials']:
                raise Exception("Additional argument", key, "not supported.")

        self.emulator = gym.make(
            get_task_name(args.additional_arguments['task']))