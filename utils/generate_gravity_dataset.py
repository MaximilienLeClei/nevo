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

import cv2
import numpy as np
import phyre # requires python==3.6

simulator = phyre.initialize_simulator(('00003:058', '00003:059'), 'ball')
actions = simulator.build_discrete_action_space(max_actions=100_000)

simulation_images = np.empty((11_000, 4, 16, 16))
nb_valid_simulations = 0

for i in range(100_000):

    simulation = simulator.simulate_action(
        task_index=0, action=actions[i], stride=30)
    
    if simulation.status.is_invalid():
        continue

    x = simulation.images[0]
    x = phyre.observations_to_float_rgb(x)
    x = x.astype(np.float32)
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    x = cv2.resize(x, (76, 76), interpolation=cv2.INTER_AREA)
    x[:,65:] = 1

    if (x[:,64:] != 1).sum() == 0:
        if (x[:12] != 1).sum() == 0:
            if x.sum() < 5700:

                for nb_obs in range(4):

                    x = simulation.images[nb_obs]
                    x = phyre.observations_to_float_rgb(x)
                    x = x.astype(np.float32)
                    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
                    x = cv2.resize(x, (19, 19), interpolation=cv2.INTER_AREA)
                    x = x[3:,:16]

                    simulation_images[nb_valid_simulations][nb_obs] = x

                nb_valid_simulations += 1
                print(nb_valid_simulations)
                if nb_valid_simulations == 11_000:
                    print(i)
                    break

np.save('../data/behaviour/gravity/11k.npy', simulation_images)