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
import copy

def seed(emulator, seed):
    emulator.seed(seed)

def set_state(emulator, state):
    emulator = state

def get_state(emulator):
    return copy.deepcopy(emulator)

def get_score_tasks():

    return ['acrobot', 'cart_pole', 'mountain_car', 'mountain_car_continuous',
            'pendulum', 'bipedal_walker', 'bipedal_walker_hardcore',
            'lunar_lander', 'lunar_lander_continuous', 'ant', 'half_cheetah',
            'hopper', 'humanoid', 'humanoid_standup',
            'inverted_double_pendulum', 'inverted_pendulum', 'reacher',
            'swimmer', 'walker_2d']

def get_imitation_tasks():

    return ['acrobot', 'cart_pole', 'mountain_car', 'mountain_car_continuous',
            'pendulum', 'lunar_lander', 'lunar_lander_continuous', 'swimmer']

def get_all_potential_imitation_tasks():

    return ['acrobot', 'cart_pole', 'mountain_car', 'mountain_car_continuous',
            'pendulum', 'bipedal_walker', 'bipedal_walker_hardcore',
            'lunar_lander', 'lunar_lander_continuous', 'ant', 'half_cheetah',
            'hopper', 'humanoid', 'swimmer', 'walker_2d']

# Early published experiments did not use standardization.
def standardize(env_path, task):

    non_standardized_tasks = ['acrobot', 'cart_pole', 'mountain_car',
                              'mountain_car_continuous', 'bipedal_walker',
                              'bipedal_walker_hardcore', 'lunar_lander',
                              'lunar_lander_continuous', 'humanoid_standup',
                              'inverted_double_pendulum', 'inverted_pendulum',
                              'swimmer']

    if task in non_standardized_tasks and 'score' in env_path:
        return False
    else:
        return True

def get_info(task):

    discrete_output = False
    output_range = None

    if task == 'acrobot': #v1
        d_input = 6
        d_output = 3
        discrete_output = True

    elif task == 'cart_pole': #v1
        d_input = 4
        d_output = 2
        discrete_output = True

    elif task == 'mountain_car': #v0
        d_input = 2
        d_output = 3
        discrete_output = True

    elif task == 'mountain_car_continuous': #v0
        d_input = 2
        d_output = 1
        output_range = 1

    elif task == 'pendulum': #v1
        d_input = 3
        d_output = 1
        output_range = 2

    elif task == 'bipedal_walker': # v3
        d_input = 24
        d_output = 4
        output_range = 1

    elif task == 'bipedal_walker_hardcore': #v3
        d_input = 24
        d_output = 4
        output_range = 1

    elif task == 'lunar_lander': #v2
        d_input = 8
        d_output = 4
        discrete_output = True

    elif task == 'lunar_lander_continuous': #v2
        d_input = 8
        d_output = 2
        output_range = 1

    elif task == 'ant': #v3
        d_input = 111
        d_output = 8
        output_range = 1

    elif task == 'half_cheetah': #v3
        d_input = 17
        d_output = 6
        output_range = 1

    elif task == 'hopper': #v3
        d_input = 11
        d_output = 3
        output_range = 1

    elif task == 'humanoid': #v3
        d_input = 376
        d_output = 17
        output_range = .4

    elif task == 'humanoid_standup': #v2
        d_input = 376
        d_output = 17
        output_range = .4

    elif task == 'inverted_double_pendulum': #v2
        d_input = 11
        d_output = 1
        output_range = 1

    elif task == 'inverted_pendulum': #v2
        d_input = 4
        d_output = 1
        output_range = 3
        
    elif task == 'reacher': #v2
        d_input = 11
        d_output = 2
        output_range = 1

    elif task == 'swimmer': #v3
        d_input = 8
        d_output = 2
        output_range = 1

    elif task == 'walker_2d': #v3
        d_input = 17
        d_output = 6
        output_range = 1

    else:
        raise Exception('Task: ' + task + ' not supported.')

    return d_input, d_output, discrete_output, output_range

def get_task_name(task):

    if task == 'acrobot':
        return 'Acrobot-v1'
    elif task == 'cart_pole':
        return 'CartPole-v1'
    elif task == 'mountain_car':
        return 'MountainCar-v0'
    elif task == 'mountain_car_continuous':
        return 'MountainCarContinuous-v0'
    elif task == 'pendulum':
        return 'Pendulum-v1'
    elif task == 'bipedal_walker':
        return 'BipedalWalker-v3'
    elif task == 'bipedal_walker_hardcore':
        return 'BipedalWalkerHardcore-v3'
    elif task == 'lunar_lander':
        return 'LunarLander-v2'
    elif task == 'lunar_lander_continuous':
        return 'LunarLanderContinuous-v2'
    elif task == 'ant':
        return 'Ant-v3'
    elif task == 'half_cheetah':
        return 'HalfCheetah-v3'
    elif task == 'hopper':
        return 'Hopper-v3'
    elif task == 'humanoid':
        return 'Humanoid-v3'
    elif task == 'humanoid_standup':
        return 'HumanoidStandup-v2'
    elif task == 'inverted_double_pendulum':
        return 'InvertedDoublePendulum-v2'
    elif task == 'inverted_pendulum':
        return 'InvertedPendulum-v2'
    elif task == 'reacher':
        return 'Reacher-v2'
    elif task == 'swimmer':
        return 'Swimmer-v3'
    elif task == 'walker_2d':
        return 'Walker2d-v3'
    else:
        raise Exception('Task: ' + task + ' not supported.')