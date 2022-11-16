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

#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --output=slurm/logs/%j.out
#SBATCH --cpus-per-task=1

if [ ! ${1} == "-s" ] && [ ! ${1} == "--states_path" ]; then

	nb_elapsed_generations=0
	data_path=data/
	states_path=data/states/
	save_frequency=100
	communication=ps_p2p
	additional_arguments={}

	while [ ${#} -gt 0 ]; do

		case "$1" in
			-e|--env_path)
				env_path=${2}; shift 2;;
			-b|--bots_path)
				bots_path=${2}; shift 2;;
			-p|--population_size)
				population_size=${2}; shift 2;;
			-l|--nb_elapsed_generations)
				nb_elapsed_generations=${2}; shift 2;;
			-g|--nb_generations)
				nb_generations=${2}; shift 2;;
			-d|--data_path)
				data_path=${2}; shift 2;;
			-s|--states_path)
				states_path=${2}; shift 2;;
			-f|--save_frequency)
				save_frequency=${2}; shift 2;;
			-c|--communication)
				communication=${2}; shift 2;;
			-a|--additional_arguments)
				additional_arguments=${2}; shift 2;;
			*)
				echo "Unknown argument : ${1}"; exit 1 ;;
		esac
		
	done

	if [ -z ${env_path} ] || [ -z ${bots_path} ] || [ -z ${nb_generations} ] || [ -z ${population_size} ]; then
		echo "Missing arguments"
		exit 1
	fi

else

	nb_tests=10
	nb_obs_per_test=1000000000
	seed=-1

	while [ ${#} -gt 0 ]; do

		case "$1" in
			-s|--states_path)
				states_path=${2}; shift 2;;
			-t|--nb_tests)
				nb_tests=${2}; shift 2;;
			-o|--nb_obs_per_test)
				nb_obs_per_test=${2}; shift 2;;
			-d|--seed)
				seed=${2}; shift 2;;
			*)
				echo "Unknown argument : ${1}"; exit 1 ;;
		esac
		
	done

	if [ -z ${states_path} ]; then
		echo "Missing arguments"
		exit 1
	fi

fi

while
    srun ${SCRATCH}/nevo/slurm/setup.sh ${env_path}
    [ ${?} != 0 ]
do true; done

if [ -z ${nb_tests} ]; then

	while
		srun ${SCRATCH}/nevo/slurm/run.sh ${env_path} \
										  ${bots_path} \
										  ${population_size} \
										  ${nb_elapsed_generations} \
										  ${nb_generations} \
										  ${data_path} \
										  ${states_path} \
										  ${save_frequency} \
										  ${communication} \
										  "${additional_arguments}"
		[ ${?} != 0 ]
	do true; done

else

	while
		srun ${SCRATCH}/nevo/slurm/run.sh ${states_path} \
										  ${nb_tests} \
										  ${nb_obs_per_test} \
										  ${seed}
		[ ${?} != 0 ]
	do true; done

fi
