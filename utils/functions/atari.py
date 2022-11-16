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

from collections import deque
import re

import cv2
import gym
from gym import spaces
import numpy as np

# emulator.seed(seed) creates unstable behaviour & doesn't alter the randomness
def seed(emulator, seed):
    return 

def set_state(emulator, state):
    emulator.unwrapped.restore_full_state(state)

def get_state(emulator):
    return emulator.unwrapped.clone_full_state()

def get_task_name(task):
    return 'ALE/' + re.sub(r'(?:^|_)([a-z])',
                           lambda x: x.group(1).upper(),
                           task) + '-v5'

def get_score_tasks():

    return ['alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
            'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling',
            'boxing', 'breakout', 'centipede', 'chopper_command',
            'crazy_climber', 'defender', 'demon_attack', 'double_dunk',
            'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher',
            'gravitar', 'hero', 'ice_hockey', 'jamesbond', 'kangaroo',
            'krull', 'kung_fu_master', 'montezuma_revenge', 'ms_pacman',
            'name_this_game', 'phoenix', 'pitfall', 'pong', 'private_eye',
            'qbert', 'riverraid', 'road_runner' ,'robotank', 'seaquest',
            'skiing', 'solaris',  'space_invaders', 'star_gunner', 'surround',
            'tennis', 'time_pilot', 'tutankham', 'up_n_down', 'venture',
            'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']

def get_all_potential_imitation_tasks():

    return ['asteroids', 'beam_rider', 'breakout', 'enduro', 'pong', 'qbert',
            'road_runner', 'seaquest', 'space_invaders']

def get_imitation_tasks():
    
    return ['breakout', 'enduro', 'pong']

def get_action(x):

    # FIRE, UP, DOWN, LEFT, RIGHT

    if x[0] == 0:                    #          ¬ FIRE   =  ∅
        f = ''
    else: # x[0] == 1:               #            FIRE   = FIRE
        f = 'f'
        
    if x[1] == 0 and x[2] == 0:      # ¬ UP   & ¬ DOWN   =  ∅
        ud = ''
    elif x[1] == 1 and x[2] == 0:    #   UP   & ¬ DOWN   =  UP
        ud = 'u'
    elif x[1] == 0 and x[2] == 1:    # ¬ UP   &   DOWN   =  DOWN
        ud = 'd'
    else: # x[1] == 1 and x[2] == 1: #   UP   &   DOWN   =  ∅
        ud = ''

    if x[3] == 0 and x[4] == 0:      # ¬ LEFT & ¬ RIGHT  =  ∅
        lr = ''
    elif x[3] == 1 and x[4] == 0:    #   LEFT & ¬ RIGHT  =  LEFT
        lr = 'l'
    elif x[3] == 0 and x[4] == 1:    # ¬ LEFT &   RIGHT  =  RIGHT
        lr = 'r'
    else: # x[3] == 1 and x[4] == 1: #   LEFT &   RIGHT  =  ∅
        lr = ''

    if f == '' and lr == '' and ud == '':
        x = 0 # NOOP

    elif f == 'f' and lr == '' and ud == '':
        x = 1 # FIRE

    elif f == '' and lr == '' and ud == 'u':
        x = 2 # UP

    elif f == '' and lr == 'r' and ud == '':
        x = 3 # RIGHT

    elif f == '' and lr == 'l' and ud == '':
        x = 4 # LEFT

    elif f == '' and lr == '' and ud == 'd':
        x = 5 # DOWN

    elif f == '' and lr == 'r' and ud == 'u':
        x = 6 # UPRIGHT

    elif f == '' and lr == 'l' and ud == 'u':
        x = 7 # UPLEFT

    elif f == '' and lr == 'r' and ud == 'd':
        x = 8 # DOWNRIGHT

    elif f == '' and lr == 'l' and ud == 'd':
        x = 9 # DOWNLEFT

    elif f == 'f' and lr == '' and ud == 'u':
        x = 10 # UPFIRE

    elif f == 'f' and lr == 'r' and ud == '':
        x = 11 # RIGHTFIRE

    elif f == 'f' and lr == 'l' and ud == '':
        x = 12 # LEFTFIRE

    elif f == 'f' and lr == '' and ud == 'd':
        x = 13 # DOWNFIRE

    elif f == 'f' and lr == 'r' and ud == 'u':
        x = 14 # UPRIGHTFIRE

    elif f == 'f' and lr == 'l' and ud == 'u':
        x = 15 # UPLEFTFIRE

    elif f == 'f' and lr == 'r' and ud == 'd':
        x = 16 # DOWNRIGHTFIRE

    else: # f == 'f' and lr == 'l' and ud == 'd':
        x = 17 # DOWNLEFTFIRE

    return x

def hide_score(obs, task):

    new_obs = np.array(obs, copy=True)

    if 'breakout' in task:
        new_obs[:10] = 0

    elif 'enduro' in task:
        new_obs[-21:] = 0

    elif 'pong' in task:
        new_obs[:12] = 0

    return new_obs

# From https://bit.ly/2Q4fAih
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

# From https://bit.ly/2Q4fAih
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros(
            (2,)+env.observation_space.shape, dtype=np.uint8)
        self._skip       = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

# From https://bit.ly/2Q4fAih
class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing.
        """
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset(**kwargs)
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        return self.env.step(ac)

# From https://bit.ly/2Q4fAih
class WarpFrame(gym.ObservationWrapper):
    def __init__(self,
                 env,
                 width=84,
                 height=84,
                 grayscale=True,
                 dict_space_key=None):
        """
        Warp frames to 84x84 as done in the Nature paper and later work.
        If the environment uses dictionary observations, `dict_space_key`
        can be specified which indicates which observation should be warped.
        """
        super().__init__(env)
        self._width = width
        self._height = height
        self._grayscale = grayscale
        self._key = dict_space_key
        if self._grayscale:
            num_colors = 1
        else:
            num_colors = 3

        new_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width, num_colors),
            dtype=np.uint8,
        )
        if self._key is None:
            original_space = self.observation_space
            self.observation_space = new_space
        else:
            original_space = self.observation_space.spaces[self._key]
            self.observation_space.spaces[self._key] = new_space
        assert (original_space.dtype == np.uint8 
                and len(original_space.shape) == 3)

    def observation(self, obs):
        if self._key is None:
            frame = obs
        else:
            frame = obs[self._key]

        if self._grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(
            frame, (self._width, self._height), interpolation=cv2.INTER_AREA
        )
        if self._grayscale:
            frame = np.expand_dims(frame, -1)

        if self._key is None:
            obs = frame
        else:
            obs = obs.copy()
            obs[self._key] = frame
        return obs

# Modified from https://bit.ly/2Q4fAih
class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames"""
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(shp[:-1] + (shp[-1] * k,)),
                             dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return np.array(self.frames).squeeze()

class SkiingNoFireEnv(gym.Wrapper):
    """No shooting while skiing"""
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.spec._env_name == 'Skiing'
        
    def step(self, ac):
        if ac in range(1, 10):
            ac -= 1
        elif ac in range(10, 18):
            ac -= 9
        return self.env.step(ac)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

def wrap(env):

    env = NoopResetEnv(env, noop_max=30)

    env = MaxAndSkipEnv(env, skip=4)
    
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    if 'Skiing' in env.spec.id:
        env = SkiingNoFireEnv(env)

    env = WarpFrame(env)
    env = FrameStack(env, k=4)

    return env