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
import random
import sys
import warnings

from mpi4py import MPI
import numpy as np
import torch


sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/..')

warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(2**31-1)

parser = argparse.ArgumentParser()

parser.add_argument('--states_path', '-sp', type=str, required=True,
                    help="Path to the saved states <=> "
                         "data/states/<env_path>/<additional_arguments>/"
                         "<bots_path>/<population_size>/")

parser.add_argument('--nb_tests', '-t', type=int, default=10,
                    help="Number of tests to evaluate the agents on.")

parser.add_argument('--nb_obs_per_test', '-o', type=int, default=2**31-1,
                    help="Number of observations per test.")

parser.add_argument('--seed', '-s', type=int, default=-1,
                    help="Optional seed to evaluate on. If this argument is "
                         "passed, only one test will be ran.")

args = parser.parse_args()

if args.seed != -1:
    args.nb_tests = 1

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

MAX_INT = 2**31-1

# Backward Compatibility for Control Task experiments
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "bots.static.rnn.control":
            renamed_module = "bots.network.static.rnn.control"
        elif module == "bots.dynamic.rnn.control":
            renamed_module = "bots.network.dynamic.rnn.control"

        return super(RenameUnpickler, self).find_class(renamed_module, name)

"""
Process arguments
"""

if args.states_path[-1] == '/':
    args.states_path = args.states_path[:-1]

split_path = args.states_path.split('/')

env_path = split_path[-4]
additional_arguments = split_path[-3]
bots_path = split_path[-2]
pop_size = int(split_path[-1])

split_additional_arguments = additional_arguments.split('~')

if 'score' in env_path:

    steps = split_additional_arguments[0].split('.')[1]
    task = split_additional_arguments[1].split('.')[1]
    transfer = split_additional_arguments[2].split('.')[1]
    trials = split_additional_arguments[3].split('.')[1]

else: # 'imitate' in env_path:

    merge = split_additional_arguments[0].split('.')[1]
    steps = split_additional_arguments[1].split('.')[1]
    task = split_additional_arguments[2].split('.')[1]
    transfer = split_additional_arguments[3].split('.')[1]

"""
Initialize environment
"""

if 'control' in env_path:

    import gym
    from utils.functions.control import get_task_name

    emulator = gym.make(get_task_name(task))

    hide_score = lambda x : x

elif 'atari' in env_path:

    import gym
    import ale_py

    from utils.functions.atari import get_task_name, hide_score, wrap

    emulator = wrap(gym.make(get_task_name(task),
                             frameskip=1,
                             repeat_action_probability=0))

else: # 'gravity' in env_path:

    data = np.load('data/behaviour/gravity/11k.npy')[-1000:]

"""
Import bots
"""

if 'control' in env_path:
    if 'rnn' in bots_path:
        if 'dynamic' in bots_path:
            from bots.network.dynamic.rnn.control import Bot
        else: # 'static' in bots_path:
            from bots.network.static.rnn.control import Bot
    else:
        if 'dynamic' in bots_path:
            raise Exception("There is no dynamic FC net   ")
        else: # 'static' in bots_path:
            from bots.network.static.fc.control import Bot


elif 'atari' in env_path:

    if 'dynamic' in bots_path:
        from bots.network.dynamic.conv_rnn.atari import Bot
    else: # 'static' in bots_path:
        from bots.network.static.conv_rnn.atari import Bot

else: # 'gravity' in env_path:

    if 'dynamic.conv_rnn' in bots_path:
        from bots.network.dynamic.conv_rnn.gravity import Bot
    else: # 'static.conv_rnn' in bots_path:
        from bots.network.static.conv_rnn.gravity import Bot

"""
Distribute workload
"""

files = [os.path.basename(x) for x in glob.glob(args.states_path + '/*')]

gens = []

for file in files:
    if file.isdigit() and os.path.isdir(args.states_path + '/' + file):
        gens.append(int(file))

gens.sort()

process_gens = []

for i in range(len(gens)):
    if i % size == rank:
        process_gens.append(gens[i])

for gen in process_gens:

    print('Gen : ' + str(gen))

    path = args.states_path + '/' + str(gen) + '/'

    if os.path.isfile(path + 'scores.npy'):
        continue

    pkl_files = [os.path.basename(x) for x in glob.glob(path + '*.pkl')]

    state_files = []

    for pkl_file in pkl_files:

        if pkl_file[:-4].isdigit():

            state_files.append(pkl_file)

    if len(state_files) == 0:
        raise Exception("Directory '" + path + "' empty.")

    try:

        with open(path + '0.pkl', 'rb') as f:
            state = RenameUnpickler(f).load()

    except Exception:

        print("File '" + path + "0.pkl' doesn't exist / is corrupted.")

    if len(state) == 3:

        full_seed_list, _, _ = state

    else: # len(state) == 4:

        _, _, latest_fitnesses_and_bot_sizes, bots = state

        for i in range(1, len(state_files)):

            try:

                with open(path + str(i) + '.pkl', 'rb') as f:
                    bots += RenameUnpickler(f).load()[0]

            except Exception:

                print("File '" + path + str(i) + \
                      ".pkl' doesn't exist / is corrupted.")

        fitnesses_sorting_indices = \
            latest_fitnesses_and_bot_sizes[:, :, 0].argsort(axis=0)
        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)
        selected = np.greater_equal(fitnesses_rankings, pop_size//2)
        selected_indices = np.where(selected[:,0] == True)[0]

    if 'gravity' in env_path:
        scores = np.zeros((pop_size//2))
    else:
        scores = np.zeros((pop_size//2, args.nb_tests))

    for i in range(pop_size//2):

        if len(state) == 3:

            bot = Bot(0)
            bot.build(full_seed_list[i][0])

        else: # len(state) == 4:

            bot = bots[selected_indices[i]][0]
        
        bot.setup_to_run()

        for j in range(args.nb_tests):

            if args.seed != -1:
                seed = args.seed
            else:
                seed = MAX_INT-j

            np.random.seed(MAX_INT-j)
            torch.manual_seed(MAX_INT-j)
            random.seed(MAX_INT-j)
            bot.reset()

            if 'gravity' not in env_path:

                emulator.seed(MAX_INT-j)
                obs = emulator.reset()
                done = False

            else: # 'gravity' in env_path:

                data_point = data[j]
                if task == 'predict':
                    nb_obs_fed_to_generator = 3
                else:
                    nb_obs_fed_to_generator = 1 # 'generate'
            
            for k in range(args.nb_obs_per_test):

                if 'gravity' not in env_path:

                    if 'imitate' in env_path:
                        obs = hide_score(obs)

                    obs, rew, done, _ = emulator.step(bot(obs))

                    scores[i][j] += rew

                    if done:
                        break

                else: # 'gravity' in env_path:

                    if k < nb_obs_fed_to_generator:
                        obs = data_point[k]

                    obs = bot(obs)

                    if torch.is_tensor(obs):
                        obs = obs.numpy()

                    scores[i] += np.sum((obs - data_point[k+1]) ** 3)

                    if k == 2:
                        break

    np.save(path + 'scores.npy', scores)