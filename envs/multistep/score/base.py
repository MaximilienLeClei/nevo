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

from typing import Any, Tuple

import numpy as np

from envs.multistep.base import MultistepEnvBase


class ScoreMultistepEnvBase(MultistepEnvBase):
    """
    Score Multistep Env Base class.
    Concrete subclasses need to be named *Env*.
    """
    def __init__(self, args, rank, size):

        if not hasattr(self, 'seed'):
            raise NotImplementedError("Score Multistep Environments require "
                                      "the attribute 'seed': a function "
                                      "that seeds the emulator.")
        if not hasattr(self, 'get_state'):
            raise NotImplementedError("Score Multistep Environments require  "
                                      "theattribute 'get_state': a function "
                                      "that returns the emulator's state.")
        if not hasattr(self, 'set_state'):
            raise NotImplementedError("Score Multistep Environments require "
                                      "the attribute 'set_state': a function "
                                      "that sets the emulator's state.")

        if 'trials' not in args.additional_arguments:
            args.additional_arguments['trials'] = 1
        elif not isinstance(args.additional_arguments['trials'], int):
            raise Exception("Additional argument 'trials' is of wrong type. "
                            "It needs to be an integer >= 1.")
        elif args.additional_arguments['trials'] < 1:
            raise Exception("Additional argument 'trials' needs to be >= 1.")

        super().__init__(args, rank, size, io_path='IO.base', nb_pops=1)

        if (args.additional_arguments['trials'] > 1
            and args.additional_arguments['transfer'] in ['yes', 'fit']):
            raise Exception("Additional argument 'trials' > 1 requires "
                            "additional argument 'transfer' = 'no'.")

    def reset(self, gen_nb: int, trial_nb: int) -> np.ndarray:
        """
        First reset function called during the run.

        Args:
            gen_nb - Generation number.
            trial_nb - Trial number.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if self.args.additional_arguments['transfer'] in ['no', 'fit']:

            if 'seed' in self.args.additional_arguments:
                self.bot.seed = self.args.additional_arguments['seed']
            else:
                self.seed(
                    self.emulator,
                    gen_nb*self.args.additional_arguments['trials']+trial_nb)

            obs = self.emulator.reset()

            return obs

        else: # self.args.additional_arguments['transfer'] == 'yes':

            self.seed(self.emulator, self.bot.seed)

            obs = self.emulator.reset()

            if gen_nb != 0:
                self.set_state(self.emulator, self.bot.state)
                obs = self.bot.obs.copy()

            return obs

    def done_reset(self, gen_nb: int) -> Tuple[Any, bool]:
        """
        Reset function called whenever the emulator returns done.

        Args:
            gen_nb - Generation number.
        Returns:
            Any - A new environment observation (np.ndarray) or nothing.
            bool - Whether the episode should terminate.
        """
        if self.args.additional_arguments['transfer'] in ['no', 'fit']:

            return None, True

        else: # if self.args.additional_arguments['transfer'] == 'yes':

            if self.log:
                print(self.bot.episode_score)
                self.bot.episode_score = 0

            self.bot.reset()

            if 'seed' in self.args.additional_arguments:
                self.bot.seed = self.args.additional_arguments['seed']
            else:
                self.bot.seed = gen_nb

            self.seed(self.emulator, self.bot.seed)

            obs = self.emulator.reset()

            return obs, False

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every run.

        Args:
            obs - The final environment observation.
        """
        if self.args.additional_arguments['transfer'] in ['no', 'fit']:

            if self.log:
                if self.args.additional_arguments['transfer'] == 'fit':
                    print(self.bot.run_score)

            self.bot.reset()

        elif self.args.additional_arguments['transfer'] == 'yes':

            self.bot.state = self.get_state(self.emulator)
            self.bot.obs = obs.copy()

    def run(self, gen_nb: int) -> float:

        self.log = False

        if not hasattr(self, 'emulator'):
            raise NotImplementedError("Score Multistep Environments require "
                                      "the attribute 'emulator': the "
                                      "emulator to run the agents on.")

        [self.bot] = self.bots
        self.bot.run_score = 0

        for trial in range(self.args.additional_arguments['trials']):

            obs, done, nb_obs = self.reset(gen_nb, trial), False, 0

            while not done:

                obs, rew, done, _ = self.emulator.step(self.bot(obs))

                self.bot.run_score += rew

                if self.log == True:
                    if self.args.additional_arguments['transfer'] == 'yes':
                        self.bot.episode_score += rew

                nb_obs += 1

                if done:
                    obs, done = self.done_reset(gen_nb)

                if nb_obs == self.args.additional_arguments['steps']:
                    done = True

            self.final_reset(obs)

        if self.args.additional_arguments['transfer'] == 'no':
            return self.bot.run_score

        else: # self.args.additional_arguments['transfer'] in ['yes', 'fit']:
            self.bot.continual_fitness += self.bot.run_score
            return self.bot.continual_fitness