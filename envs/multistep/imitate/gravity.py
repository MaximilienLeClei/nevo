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

from envs.multistep.imitate.base import ImitateMultistepEnvBase


class Emulator:

    def __init__(self, data, task):

        self.data = data
        self.task = task

        self.data_index = 0
        self.frame_index = 0

    def seed(self, data_index):

        self.data_index = data_index

    def reset(self):

        self.frame_index = 0

        return self.data[self.data_index][self.frame_index]

    def step(self, frame):
        
        if self.task == 'generate':
            if self.frame_index == 0:
                frame = self.data[self.data_index][0]

        else: # self.task == 'predict':
            frame = self.data[self.data_index][self.frame_index]

        self.frame_index += 1

        return frame, 0, self.frame_index == 4, None

class Env(ImitateMultistepEnvBase):

    def __init__(self, args, rank, size):

        self.valid_tasks = ['generate', 'predict']
        self.hide_score = lambda x, _ : x
        self.set_state, self.get_state = None, None
        self.seed = lambda x,y : x.seed(y)

        super().__init__(args, rank, size, io_path='IO.imitate.gravity')

        for key in args.additional_arguments:
            if key not in ['task', 'steps', 'transfer']:
                raise Exception("Additional argument", key, "not supported.")
        
        self.target = self.io.load_target()

        self.emulators = [Emulator(
            self.io.load_data(), args.additional_arguments['task'])]