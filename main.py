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

"""
Main script of the Nevo library, executed by as many processes as are provided
through `mpiexec`. Runs an evolutionary algorithm looping over three stages:
variation, evaluation & selection. Processes communicate at regular intervals
using mpi4py.
"""
import argparse
import copy
import pickle
import sys
import time
import warnings

import numpy as np
from mpi4py import MPI

from utils.functions.misc import initialize_environment


np.set_printoptions(suppress=True)
warnings.filterwarnings('ignore')
sys.setrecursionlimit(2**31-1)

parser = argparse.ArgumentParser()

parser.add_argument('--env_path', '-e', type=str, required=True,
                    help="Path to the env class file.")

parser.add_argument('--bots_path', '-b', type=str, required=True,
                    help="Path to the bot class file.")

parser.add_argument('--population_size', '-p', type=int, required=True,
                    help="Number of bots per population. Must be a multiple "
                         "of the number of MPI processes and must remain "
                         "constant across successive experiments.")

parser.add_argument('--nb_elapsed_generations', '-l', type=int, default=0,
                    help="Number of elapsed generations.")

parser.add_argument('--nb_generations', '-g', type=int, required=True,
                    help="Number of generations to run.")

parser.add_argument('--elitism', '-t', type=float, default=0,
                    help="Proportion (if float in [0, 0.5]) or number "
                         "(if int in [0, 0.5*pop_size]) of the best "
                         "performing bots to not mutate every iteration.")

parser.add_argument('--data_path', '-d', type=str, default='data/',
                    help="Path to the data folder.")

parser.add_argument('--states_path', '-s', type=str, default='data/states/',
                    help="Path to the states folder.")

parser.add_argument('--save_frequency', '-f', type=int, default=0,
                    help="Frequency (int in [0, nb_generations]) at which to "
                         "save the experiment's state.")

parser.add_argument('--communication', '-c',
                    choices=['ps', 'ps_p2p', 'big_ps_p2p'], default='ps_p2p',
                    help="ps : A primary process scatters/gathers data "
                              "to/from secondary processes.\n"
                         "ps_p2p : ps *plus* peer-to-peer data exchange "
                                  "between all processes.\n"
                         "big_ps_p2p : ps_p2p *minus* initial/final bot "
                                      "scatter/gather. Useful in settings "
                                      "where the combined size of bots > 2 GB "
                                      "(the number of MPI processes must "
                                      "remain constant in successive "
                                      "experiments).\n"
                         "All protocols must remain constant across "
                         "successive experiments.")

parser.add_argument('--enable_gpu_use', '-u', type=int, default=0,
                    help="Makes use of GPUs if available (for static nets).")

parser.add_argument('--additional_arguments', '-a', type=str, default='{}',
                    help="JSON string or path to a JSON file of additional "
                         "arguments.")

args = parser.parse_args()

"""
Initialization of variables and objects.
"""
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

env = initialize_environment(args, rank, size)
old_nb_gen = args.nb_elapsed_generations
new_nb_gen = args.nb_generations
pop_size = args.population_size
batch_size = pop_size // size
nb_pops = env.nb_pops

ps_comm = args.communication == 'ps'
ps_p2p_comm = args.communication == 'ps_p2p'
big_ps_p2p_comm = args.communication == 'big_ps_p2p'
p2p_comm = args.communication in ['ps_p2p', 'big_ps_p2p']

full_seed_list = None
fitnesses = None

full_seed_list_batch = np.empty((batch_size, nb_pops, 1), dtype=np.uint32)
fitnesses_batch = np.empty((batch_size, nb_pops), dtype=np.float32)

if p2p_comm:

    bots = None
    pairing_and_seeds = None
    fitnesses_and_bot_sizes = None

    bots_batch = []

    # [MPI buffer size, pair position, sending, seed]
    pairing_and_seeds_batch = np.empty(
        (batch_size, nb_pops, 4), dtype=np.uint32)

    # [fitness, pickled bot size]
    fitnesses_and_bot_sizes_batch = np.empty(
        (batch_size, nb_pops, 2), dtype=np.float32) 

if rank == 0:

    fitnesses = np.empty((pop_size, nb_pops), dtype=np.float32) 

    pairing_and_seeds = np.empty((pop_size, nb_pops, 4), dtype=np.uint32) 

    fitnesses_and_bot_sizes = np.empty(
        (pop_size, nb_pops, 2), dtype=np.float32)

    if old_nb_gen > 0:
        """
        Primary process loads previous experiment state.
        """
        state = env.io.load_state()

        if ps_comm:

            full_seed_list, full_fitness_list, latest_fitnesses = state

            fitnesses_sorting_indices = latest_fitnesses.argsort(axis=0)

        else: # p2p_comm:

            if ps_p2p_comm:
                full_seed_list, full_fitness_list, \
                    latest_fitnesses_and_bot_sizes, bots = state
            else: # big_ps_p2p_comm:
                full_seed_list, full_fitness_list, \
                    latest_fitnesses_and_bot_sizes, bots_batch = state

            fitnesses_sorting_indices = \
                latest_fitnesses_and_bot_sizes[:, :, 0].argsort(axis=0)

        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

    else: # old_nb_gen == 0:

        full_seed_list = np.empty((pop_size, nb_pops, 0), dtype=np.uint32)
        full_fitness_list = np.empty((pop_size, nb_pops, 0), dtype=np.float32)

if old_nb_gen > 0:

    if ps_p2p_comm:
        """
        (ps_p2p) Primary process scatters bots from the previous experiment
        to secondary processes.
        """
        if rank == 0:

            for i in range(pop_size):
                for j in range(nb_pops):
                    fitnesses_and_bot_sizes[i, j, 1] = len(
                        pickle.dumps(bots[i][j]))

            bots = [
                bots[i * batch_size: (i+1) * batch_size] for i in range(size)]

        bots_batch = comm.scatter(bots, root=0)

    elif big_ps_p2p_comm:
        """
        (big_ps_p2p) Secondary processes load bots from previous experiments.
        """
        if rank != 0:
            [bots_batch] = env.io.load_state()

        for i in range(batch_size):
            for j in range(nb_pops):
                fitnesses_and_bot_sizes_batch[i, j, 1] = len(
                    pickle.dumps(bots_batch[i][j]))

        comm.Gather(
            fitnesses_and_bot_sizes_batch, fitnesses_and_bot_sizes, root=0)

"""
Start of the evolutionary algorithm.
"""
for gen_nb in range(old_nb_gen, old_nb_gen + new_nb_gen):

    np.random.seed(gen_nb)

    if rank == 0:

        start = time.time()
        """
        Primary processes generates seeds to encode variation operations.
        """
        new_seeds = env.io.generate_new_seeds(gen_nb)
        
        full_seed_list = np.concatenate((full_seed_list, new_seeds), 2)

        if gen_nb != 0:
            for j in range(nb_pops):
                full_seed_list[:, j] = full_seed_list[:, j][
                    fitnesses_rankings[:, j]]

    if ps_comm or gen_nb == 0:
        """
        (ps / gen 0) Primary processes scatters seeds to secondary processes
        """
        full_seed_list_batch = np.empty(
            (batch_size, nb_pops, gen_nb + 1), dtype=np.uint32)

        comm.Scatter(full_seed_list, full_seed_list_batch, root=0)

    else: # p2p_comm and gen > 0:
        """
        (p2p) Primary process generates peer-to-peer and seed information to
        scatter to secondary processes. 
        """
        if rank == 0:

            pairing_and_seeds[:, :, 0] = np.max(
                fitnesses_and_bot_sizes[:, :, 1]) # MPI buffer size

            for j in range(nb_pops):

                pair_ranking = (fitnesses_rankings[:, j] + pop_size // 2) \
                    % pop_size

                pairing_and_seeds[:, j, 1] = fitnesses_sorting_indices[:, j][
                    pair_ranking] # pair position

            pairing_and_seeds[:, :, 2] = np.greater_equal(
                fitnesses_rankings, pop_size // 2) # sending

            pairing_and_seeds[:, :, 3] = full_seed_list[:, :, -1] # seed

        comm.Scatter(pairing_and_seeds, pairing_and_seeds_batch, root=0)
        """
        (p2p) Processes exchange bots 
        """
        req = []

        for i in range(batch_size):

            for j in range(nb_pops):

                pair = int(pairing_and_seeds_batch[i, j, 1] // batch_size)

                if pairing_and_seeds_batch[i, j, 2] == 1: # sending

                    tag = int(pop_size * j + batch_size * rank + i)

                    req.append(comm.isend(bots_batch[i][j],
                                          dest=pair, tag=tag))

                else: # pairing_and_seeds_batch[i, j, 2] == 0: # receiving

                    tag = int(pop_size * j + pairing_and_seeds_batch[i, j, 1])

                    req.append(comm.irecv(pairing_and_seeds_batch[i, j, 0],
                                          source=pair, tag=tag))

        received_bots = MPI.Request.waitall(req)

        for i, bot in enumerate(received_bots):
            if bot is not None:
                bots_batch[i // nb_pops][i % nb_pops] = bot

    for i in range(batch_size):
        """
        Variation 
        """
        if ps_comm or gen_nb == 0:

            env.build_bots(full_seed_list_batch[i])

        else:

            env.bots = bots_batch[i]
            env.extend_bots(pairing_and_seeds_batch[i, :, 3])
        """
        Evaluation 
        """
        fitnesses_batch[i] = env.evaluate_bots(gen_nb)

        if p2p_comm:

            if gen_nb == 0:
                bots_batch.append(copy.deepcopy(env.bots))

            fitnesses_and_bot_sizes_batch[i, :, 0] = fitnesses_batch[i]

            for j in range(nb_pops):
                fitnesses_and_bot_sizes_batch[i, j, 1] = len(
                    pickle.dumps(bots_batch[i][j]))
    """
    Primary process gathers fitness + pickled bot size information (p2p only)
    """
    if ps_comm:

        comm.Gather(fitnesses_batch, fitnesses, root=0)

    else: # p2p_comm:

        comm.Gather(
            fitnesses_and_bot_sizes_batch, fitnesses_and_bot_sizes, root=0)

    if rank == 0:
        """
        Primary process extracts fitness informations
        """
        if p2p_comm:
            fitnesses = fitnesses_and_bot_sizes[:, :, 0]

        if 'merge' in args.additional_arguments:
            if args.additional_arguments['merge'] == 'yes':
                fitnesses[:, 0] += fitnesses[:, 1][::-1]
                fitnesses[:, 1] = fitnesses[:, 0][::-1]

        fitnesses_sorting_indices = fitnesses.argsort(axis=0)
        fitnesses_rankings = fitnesses_sorting_indices.argsort(axis=0)

        for j in range(nb_pops):
            full_seed_list[:, j] = full_seed_list[:, j][
                fitnesses_sorting_indices[:, j]]
        """
        Selection 
        """
        full_seed_list[:pop_size // 2] = full_seed_list[pop_size // 2:]

        print(gen_nb + 1, ':', int(time.time() - start),
              '\n', np.mean(fitnesses, 0), '\n', np.max(fitnesses, 0))

        full_fitness_list = np.concatenate(
            (full_fitness_list, fitnesses[:, :, None]), 2)

    if gen_nb + 1 in env.io.save_points or gen_nb == 0:
        """
        State saving 
        """
        if ps_comm:

            if rank == 0:
                env.io.save_state([full_seed_list, full_fitness_list,
                                   fitnesses], gen_nb + 1)

        if ps_p2p_comm:
            
            batched_bots = comm.gather(bots_batch, root=0)

            if rank == 0:
                
                bots = []

                for bot_batch in batched_bots:
                    bots = bots + bot_batch

                env.io.save_state([full_seed_list, full_fitness_list,
                                   fitnesses_and_bot_sizes, bots], gen_nb + 1)

        if big_ps_p2p_comm:

            if rank == 0:
                env.io.save_state([full_seed_list, full_fitness_list,
                                   fitnesses_and_bot_sizes, bots_batch],
                                  gen_nb + 1)
            else: # rank != 0:
                env.io.save_state([bots_batch], gen_nb + 1)