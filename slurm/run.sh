#!/usr/bin/env bash

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

module purge &>/dev/null
module load StdEnv/2020 python/3.8.10 scipy-stack/2021a gcc/9.3.0 openmpi/4.0.3 mpi4py/3.0.3 glfw/3.3.2 &>/dev/null

. $SLURM_TMPDIR/envo/bin/activate &>/dev/null

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin &>/dev/null

if [ ${#} == 10 ]; then

    env_path=${1}
    bots_path=${2}
    population_size=${3}
    nb_elapsed_generations=${4}
    nb_generations=${5}
    data_path=${6}
    states_path=${7}
    save_frequency=${8}
    communication=${9}
    additional_arguments=${10}

    python3 ${SCRATCH}/nevo/main.py --env_path ${env_path} \
                                    --bots_path ${bots_path} \
                                    --population_size ${population_size} \
                                    --nb_elapsed_generations ${nb_elapsed_generations} \
                                    --nb_generations ${nb_generations} \
                                    --data_path ${data_path} \
                                    --states_path ${states_path} \
                                    --save_frequency ${save_frequency} \
                                    --communication ${communication} \
                                    --additional_arguments "${additional_arguments}"
else # [ ${#} == 3 ]

    states_path=${1}
    nb_tests=${2}
    nb_obs_per_test=${3}
    seed=${4}

    python3 ${SCRATCH}/nevo/utils/evaluate.py --states_path ${states_path} \
                                              --nb_tests ${nb_tests} \
                                              --nb_obs_per_test ${nb_obs_per_test} \
                                              --seed ${seed}

fi
