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

# Warning : Takes a little while to complete

echo "Downloading wheel builder ..."
cd ${SCRATCH}
mkdir -p wheels/
git clone https://github.com/ComputeCanada/wheels_builder.git &>/dev/null
cd wheels_builder/

echo "Building wheels ..."
./build_wheel.sh --recursive=0 --package gym==0.21.0 --python=3.8 &>/dev/null
./build_wheel.sh --recursive=0 --package AutoROM==0.4.0 --python=3.8 &>/dev/null
./build_wheel.sh --recursive=0 --package box2d-py==2.3.8 --python=3.8 &>/dev/null
./build_wheel.sh --recursive=0 --package stable-baselines3==1.4.0 --python=3.8 &>/dev/null
./build_wheel.sh --recursive=0 --package sb3_contrib==1.4.0 --python=3.8 &>/dev/null

mv gym-0.21.0-py3-none-any.whl ../wheels/.
mv AutoROM-0.4.0-py3-none-any.whl ../wheels/.
mv box2d_py-2.3.8-cp38-cp38-linux_x86_64.whl ../wheels/.
mv stable_baselines3-1.4.0-py3-none-any.whl ../wheels/.
mv sb3_contrib-1.4.0-py3-none-any.whl ../wheels/.

echo "Downloading MuJoCo & ROMs ..."

module ()
{
    eval $($LMOD_CMD bash "$@") && eval $(${LMOD_SETTARG_CMD:-:} -s sh)
}

module load python/3.8.10 &>/dev/null
virtualenv --no-download ${SCRATCH}/mujoco_and_roms &>/dev/null
. ${SCRATCH}/mujoco_and_roms/bin/activate &>/dev/null
mkdir -p ${HOME}/.mujoco/ &>/dev/null
cd ${HOME}/.mujoco/ &>/dev/null
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz &>/dev/null
tar xvfz mujoco210-linux-x86_64.tar.gz &>/dev/null
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/.mujoco/mujoco210/bin &>/dev/null
cd ${SCRATCH} &>/dev/null
pip install git+https://github.com/openai/mujoco-py &>/dev/null
pip install AutoRom==0.4.0 &>/dev/null
AutoROM --accept-license &>/dev/null
deactivate &>/dev/null

git clone https://github.com/MaximilienLC/nevo &>/dev/null
git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo &>/dev/null
cp -r ${SCRATCH}/rl-baselines3-zoo/rl-trained-agents/ ${SCRATCH}/nevo/data/rl-trained-agents/

echo "Done!"