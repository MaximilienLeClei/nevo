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

# Create the virtual environments and install packages

module purge &>/dev/null
module load StdEnv/2020 python/3.8.10 scipy-stack/2021a gcc/9.3.0 openmpi/4.0.3 mpi4py/3.0.3 &>/dev/null

if [ ${SLURM_LOCALID} != 0 ]
then 
    exit 0
fi

virtualenv --no-download ${SLURM_TMPDIR}/envo &>/dev/null
. ${SLURM_TMPDIR}/envo/bin/activate &>/dev/null

env_path=${1}

pip install torch==1.10.0 --no-index &>/dev/null
pip install opencv_python==4.5.1.48 --no-index &>/dev/null

if [[ $env_path == *"gravity"* ]]
then
    exit 0

elif [[ $env_path == *"control"* ]]
then

    pip install ${SCRATCH}/wheels/gym-0.21.0-py3-none-any.whl --no-deps &>/dev/null
    pip install ${SCRATCH}/wheels/box2d_py-2.3.8-cp38-cp38-linux_x86_64.whl --no-deps &>/dev/null
    pip install cloudpickle==2.0.0 --no-index &>/dev/null
    pip install ${SCRATCH}/wheels/stable_baselines3-1.4.0-py3-none-any.whl --no-deps &>/dev/null
    pip install ${SCRATCH}/wheels/sb3_contrib-1.4.0-py3-none-any.whl --no-deps &>/dev/null

    cp -r ${SCRATCH}/mujoco_and_roms/lib/python3.8/site-packages/mujoco_py/ \
          ${SLURM_TMPDIR}/envo/lib/python3.8/site-packages/mujoco_py/

    exit 0
    
elif [[ $env_path == *"atari"* ]]
then

    pip install ${SCRATCH}/wheels/gym-0.21.0-py3-none-any.whl --no-deps &>/dev/null
    pip install ale-py==0.7+computecanada --no-index &>/dev/null
    pip install ${SCRATCH}/wheels/stable_baselines3-1.4.0-py3-none-any.whl --no-deps &>/dev/null
    pip install ${SCRATCH}/wheels/sb3_contrib-1.4.0-py3-none-any.whl --no-deps &>/dev/null

    cp ${SCRATCH}/mujoco_and_roms/lib/python3.8/site-packages/AutoROM/roms/*.bin \
       ${SLURM_TMPDIR}/envo/lib/python3.8/site-packages/ale_py/roms/.
 
else

    exit 0

fi