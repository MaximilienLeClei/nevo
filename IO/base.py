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
import glob
import os
import pickle

import numpy as np


class IOBase:
    """
    IO Base class. IO objects are used to handle input/output files.
    Concrete subclasses need to be named *IO* (a default subclass is defined
    at the bottom of this file).
    """

    def __init__(self,
                 args: argparse.Namespace,
                 rank: int,
                 size: int,
                 nb_pops: int):
        """
        Constructor.

        Args:
            args - Experiment specific arguments (obtained through argparse).
            rank - MPI rank of process.
            size - Number of MPI processes.
            nb_pops - Total number of populations.
        """
        self.args = args
        self.rank = rank
        self.size = size
        self.nb_pops = nb_pops

        if self.args.population_size % 2 != 0:
            raise Exception("'population_size' must be an even number.")

        if self.args.population_size % self.size != 0:
            raise Exception("'population_size' must be a multiple of the "
                            "number of MPI processes.")

        self.setup_elitism()
        self.setup_save_points()
        self.setup_state_path()

    def setup_elitism(self) -> None:
        """
        Method called upon object initialization. Sets up the 'elitism'
        parameter which manages how many of the best performing bots will not
        be mutated every iteration.
        """
        if self.args.elitism < 0:

            raise Exception("'elitism' must not be < 0.")

        if self.args.elitism < 1:

            if self.args.elitism > 0.5:
                raise Exception("'elitism' must not be in ]0.5,1[.")

            self.args.elitism = int(
                self.args.elitism * self.args.population_size)

        else:

            if self.args.elitism > self.args.population_size // 2:
                raise Exception("'elitism' must be < 'population_size'/2.")

            self.args.elitism = int(self.args.elitism)

    def setup_save_points(self) -> None:
        """
        Method called upon object initialization. Sets up points at which to
        save the experiment's current state.
        """
        if (self.args.save_frequency > self.args.nb_generations
            or self.args.save_frequency < 0):
            raise Exception("'save_frequency' should be in "
                            "[0, nb_generations].")

        self.save_points = [
            self.args.nb_elapsed_generations + self.args.nb_generations]
        
        if self.args.save_frequency == 0:
            return

        for i in range(self.args.nb_generations//self.args.save_frequency):
            self.save_points.append(
                self.args.nb_elapsed_generations + \
                self.args.save_frequency * (i+1))

    def setup_state_path(self) -> None:
        """
        Method called upon object initialization. Sets up the path which
        states will be loaded from and saved into.
        """
        self.path = self.args.states_path + self.args.env_path.replace('/',
                                            '.').replace('.py', '') + '/'

        if self.args.additional_arguments == {}:

            self.path += '~'

        else:

            for key in sorted(self.args.additional_arguments):
                self.path += str(key) + '.' + \
                    str(self.args.additional_arguments[key]) + '~'

            self.path = self.path[:-1] + '/'

        self.path += self.args.bots_path.replace('/',
                     '.').replace('.py', '') + '/'

    def generate_new_seeds(self, gen_nb: int) -> np.ndarray:
        """
        Method that produces new seeds meant to mutate the bots for this
        generation.

        Args:
            gen_nb - Current generation number
        Returns:
            np.ndarray - Array of seeds
        """
        if gen_nb == 0:
            d_0_non_zero = self.args.population_size
            d_0_zero = 0
        else:
            d_0_non_zero = self.args.population_size - self.args.elitism
            d_0_zero = self.args.elitism

        if ('merge' not in self.args.additional_arguments or
            self.args.additional_arguments['merge'] == 'no'):
            d_1 = self.nb_pops
        else: # self.args.additional_arguments['merge'] == 'yes':
            d_1 = 1

        non_zero_seeds = np.random.randint(
            1, 2**32, (d_0_non_zero, d_1, 1), dtype=np.uint32)
        zero_seeds = np.zeros((d_0_zero, d_1, 1), dtype=np.uint32)
        new_seeds = np.concatenate((non_zero_seeds, zero_seeds), axis=0)

        if ('merge' in self.args.additional_arguments and
            self.args.additional_arguments['merge'] == 'yes'):

            new_seeds = np.repeat(new_seeds, 2, axis=1) # Seeds are shared

            if gen_nb == 0:
                new_seeds[:, 1] = new_seeds[:, 1][::-1] # Reverse

        return new_seeds

    def load_state(self) -> list:
        """
        Load a previous experiment's state.

        Returns:
            list - state (seeds, fitnesses, bots, ...) of experiment.
        """
        load_path = self.path + str(self.args.population_size) + '/' + \
                    str(self.args.nb_elapsed_generations) + '/'

        pkl_files = [
            os.path.basename(x) for x in glob.glob(load_path + '*.pkl')]

        state_files = []
        for pkl_file in pkl_files:
            if pkl_file[:-4].isdigit():
                state_files.append(pkl_file)

        if not os.path.isdir(load_path) or len(state_files) == 0:
            raise Exception("No saved state found at " + load_path + ".")

        if (self.args.communication == 'ps'
            or self.args.communication == 'ps_p2p') and len(state_files) > 1:
            raise Exception("'communication' = '" + \
                             self.args.communication + \
                            "while the saved state used 'big_ps_p2p'")

        if (self.args.communication == 'big_ps_p2p'
            and len(state_files) != self.size):
            raise Exception("The current number of MPI processes is " + \
                             str(self.size) + "while the saved state made "
                            "use of " + len(state_files) + ".")
        
        if not os.path.isfile(load_path + str(self.rank) + '.pkl'):
            raise Exception("File " + str(self.rank) + ".pkl missing. "
                            "Unable to load save state.")

        with open(load_path + str(self.rank) + '.pkl', 'rb') as f:
            state = pickle.load(f)

        if self.args.communication == 'ps' and len(state) == 4:
            raise Exception("'communcation' = 'ps' while "
                            "the saved state used 'ps_p2p'")

        if self.args.communication == 'ps_p2p' and len(state) == 3:
            raise Exception("'communcation' = 'ps_p2p' while "
                            "the saved state used 'ps'")

        return state

    def save_state(self, state: list, gen_nb: int) -> None:
        """
        Save the current experiment's state.

        Args:
            state - state (seeds, fitnesses, bots, ...) of experiment.
            genb_nb - Generation number.
        """
        save_path = self.path + str(self.args.population_size) + '/' + \
                                str(gen_nb) + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        with open(save_path + str(self.rank) + '.pkl', 'wb') as f:
            pickle.dump(state, f)

class IO(IOBase):
    pass