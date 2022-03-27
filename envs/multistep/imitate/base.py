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


class ImitateMultistepEnvBase(MultistepEnvBase):
    """
    Imitate Multistep Env Base class.
    Concrete subclasses need to be named *Env*.
    """
    def __init__(self, args, rank, size, io_path):

        if not hasattr(self, 'seed'):
            raise NotImplementedError("Imitate Multistep Environments "
                                      "require the attribute 'seed': a "
                                      "function that seeds the emulator.")
        if not hasattr(self, 'get_state'):
            raise NotImplementedError("Imitate Multistep Environments "
                                      "require the attribute 'get_state': a "
                                      "function that returns the emulator's "
                                      "state.")
        if not hasattr(self, 'set_state'):
            raise NotImplementedError("Imitate Multistep Environments "
                                      "require the attribute 'set_state': a  "
                                      "function that sets the emulator's "
                                      "state.")
        if not hasattr(self, 'hide_score'):
            raise NotImplementedError("Imitate Multistep Environments "
                                      "require the attribute 'hide_score': a "
                                      "function that hides the emulator's "
                                      "score.")

        if 'merge' not in args.additional_arguments:
            args.additional_arguments['merge'] = 'no'
        elif args.additional_arguments['merge'] not in ['no', 'yes']:
            raise Exception("Additional argument 'merge' invalid. It "
                            "needs be chosen from one of " +
                             str(['no', 'yes']))

        super().__init__(args, rank, size, io_path, nb_pops=2)

    def reset(self, gen_nb: int) -> np.ndarray:
        """
        First reset function called during a match.

        Args:
            gen_nb - Generation number.
        Returns:
            np.ndarray - The initial environment observation.
        """
        if self.args.additional_arguments['transfer'] in ['no', 'fit']:

            if self.log == True:
                if self.generator == self.action_taker:
                    self.generator.run_score = 0

            if self.target == self.action_taker:
                self.target.reset(gen_nb, 0)

            self.seed(self.emulator, gen_nb)
            obs = self.emulator.reset()

            return obs

        else: # self.args.additional_arguments['transfer'] == 'yes':

            if self.target == self.action_taker:
                self.target.reset(
                    self.discriminator.seed, self.discriminator.steps)

            self.seed(self.emulator, self.state_holder.seed)
            obs = self.emulator.reset()

            if gen_nb > 0:
                self.set_state(self.emulator, self.state_holder.state)
                obs = self.state_holder.obs.copy()

            return obs

    def done_reset(self, gen_nb: int) -> Tuple[Any, bool]:
        """
        Reset function called whenever the emulator returns done.

        Args:
            gen_nb - Generation number.
        Returns:
            Any - A new environment observation (np.ndarray) or None.
            bool - Whether the episode is done or not.
        """
        if self.args.additional_arguments['transfer'] in ['no', 'fit']:

            return None, True

        else: # self.args.additional_arguments['transfer'] == 'yes':

            if self.log:
                if self.generator == self.action_taker:
                    print(self.generator.episode_score)
                    self.generator.episode_score = 0

            self.state_holder.seed = gen_nb
            self.state_holder.steps = 0

            self.discriminator.reset()

            if self.generator == self.action_taker:
                self.generator.reset()
            else: # self.target == self.target:
                self.target.reset(
                    self.discriminator.seed, self.discriminator.steps)

            self.seed(self.emulator, self.state_holder.seed)
            obs = self.emulator.reset()

            return obs, False

    def final_reset(self, obs: np.ndarray) -> None:
        """
        Reset function called at the end of every match.

        Args:
            obs - The final environment observation.
        """
        if self.args.additional_arguments['transfer'] in ['no', 'fit']:

            if self.log:
                if self.generator == self.action_taker:
                    print(self.generator.run_score)
                    self.generator.run_score = 0

            self.discriminator.reset()

            if self.generator == self.action_taker:
                self.generator.reset()

        else: # self.args.additional_arguments['transfer'] == 'yes':

            self.discriminator.reset()

            self.state_holder.state = self.get_state(self.emulator)
            self.state_holder.obs = obs.copy()

    def run(self, gen_nb: int) -> float:

        self.log = False

        if not hasattr(self, 'target'):
            raise NotImplementedError("Imitate Multistep Environments "
                                      "require the attribute 'target': the "
                                      "target behaviour/agent to imitate.")
        if not hasattr(self, 'emulators'):
            raise NotImplementedError("Imitate Multistep Environments "
                                      "require the attribute 'emulators': a "
                                      "list made up of 1 or 2 emulators.")

        [self.generator, self.discriminator] = self.bots
        generator_fitness, discriminator_fitness = 0, 0

        for i in range(2): # Generator/Discriminator -> Target/Discriminator

            if i == 0:
                self.action_taker = self.state_holder = self.generator
            else: # i == 1:
                self.action_taker = self.target
                self.state_holder = self.discriminator

            self.emulator = self.emulators[-i] # Works for 1 & 2 emulators

            obs, done, nb_obs, p_target = self.reset(gen_nb), False, 0, 0

            while not done:
                
                if self.generator == self.action_taker:
                    action = self.generator(self.hide_score(obs,
                                      self.args.additional_arguments['task']))
                else: # self.target == self.action_taker:
                    action = self.target(obs)

                obs, rew, done, _ = self.emulator.step(action)
                
                if self.log == True:
                    if self.generator == self.action_taker:
                        if self.args.additional_arguments['transfer'] == 'yes':
                            self.generator.episode_score += rew
                        else: # 'transfer' in ['no', 'fit']:
                            self.generator.run_score += rew

                p_target += self.discriminator(self.hide_score(obs,
                                      self.args.additional_arguments['task']))

                nb_obs += 1

                if self.args.additional_arguments['transfer'] == 'yes':
                    self.state_holder.steps += 1

                if self.target == self.action_taker:
                    done = self.target.is_done

                if done:
                    obs, done = self.done_reset(gen_nb)

                if nb_obs == self.args.additional_arguments['steps']:
                    done = True

            p_target /= nb_obs
            
            if self.generator == self.action_taker:
                generator_fitness += p_target
                discriminator_fitness -= p_target
            else: # self.target == self.action_taker:
                discriminator_fitness += p_target

            self.final_reset(obs)
        
        if self.args.additional_arguments['transfer'] == 'no':

            return [generator_fitness, discriminator_fitness]

        else: # self.args.additional_arguments['transfer'] in ['yes', 'fit']:

            if self.args.additional_arguments['merge'] == 'yes':
                generator_fitness = generator_fitness * 2 - 1 # fair scale

            self.generator.continual_fitness += generator_fitness
            self.discriminator.continual_fitness += discriminator_fitness

            return [self.generator.continual_fitness,
                    self.discriminator.continual_fitness]