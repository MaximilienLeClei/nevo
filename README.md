![nevo image](https://i.imgur.com/29TNXP9.png)

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
**Evolving Artificial Neural Networks (a.k.a. [Neuroevolution](https://en.wikipedia.org/wiki/Neuroevolution)) in parallel using [MPI for Python](https://mpi4py.readthedocs.io/en/stable/).**

**Warning:** This library is no longer maintained. Development efforts have moved to a new library aiming called [Gran](https://github.com/MaximilienLC/gran)

## What does this library do ?

At its core, this library runs a simple [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm) wherein fixed-sized populations of artificial agents are iterated upon. Every iteration, during the **variation** stage, all agents receive random mutations. They are then **evaluated** on some defined task. Finally the top 50% scoring agents are **selected** and duplicated over the bottom 50% agents (truncation selection). This process loops onto itself for a chosen amount of iterations.

![ea image](https://i.imgur.com/ZVktPG9.png)
**<p align="center">Evolutionary Algorithm Example</p>**

In addition to this base case, we offer an **elitism** mechanism, in which the highest scoring agents do not receive variations, but do not implement other common genetic algorithm operations like **crossovers** and **speciation**.

There are many potential types of artificial agents that can be used in combination with evolutionary algorithms, however, the focus of this library is on evolving agents composed of [Artificial Neural Networks](https://en.wikipedia.org/wiki/Artificial_neural_network), a.k.a. **Neuroevolution**.

![ne image](https://i.imgur.com/vulBaCY.png)
**<p align="center">Neuroevolution Example</p>**

<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17">  For a more in-depth introduction to neuroevolution you can read [this thesis excerpt](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=9) (pages 9 to 13).  
<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17">  If you want a broader overview of the field you will most likely be interested to read [this review article](https://www.researchgate.net/profile/Jeff-Clune/publication/330203191_Designing_neural_networks_through_neuroevolution/links/5e7243fc92851c93e0ac18ea/Designing-neural-networks-through-neuroevolution.pdf).

## How do I install it ?

### On a personal computer (tested on Ubuntu 20.04):

```
# Debian packages                       ----------mpi4py---------- ~~~~~~~~~~~~~~~~~Gym~~~~~~~~~~~~~~~~~~~
sudo apt install git python3-virtualenv python3-dev libopenmpi-dev g++ swig libosmesa6-dev patchelf ffmpeg

# MuJoCo
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -P ~/Downloads/
mkdir -p ~/.mujoco/ && tar -zxf ~/Downloads/mujoco210-linux-x86_64.tar.gz -C ~/.mujoco/
echo -e "\n# MuJoCo\nMUJOCO_PATH=~/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=\$MUJOCO_PATH:\$LD_LIBRARY_PATH" >> ~/.bashrc
source ~/.bashrc

# Library & Dependencies
git clone https://github.com/MaximilienLC/nevo && cd nevo/
virtualenv venv && source venv/bin/activate && pip3 install -r requirements.txt

# Atari ROMs
AutoROM --accept-license --install-dir venv/lib/python3.8/site-packages/ale_py/roms/
```

### On a Slurm-powered cluster:

```
git clone https://github.com/MaximilienLC/nevo.git ${SCRATCH}/nevo/ && cd ${SCRATCH}/nevo/
sh slurm/download.sh
```
*Caution : You will probably need change certain files in* `slurm/` *for use on clusters outside Compute Canada.*

## How is the code structured ?

This library is structured around 4 types of classes: 
- **nets**
- **bots**
- **envs**
- **IO**

### Nets

**Nets** is short for Artificial Neural Networks. You'll find here two classes of such networks:
- Static-sized deep neural networks, which we call **static nets**
- Neural networks of dynamic complexity, which we call **dynamic nets**

To define a new **static net**, we suggest you base yourself on `nets.static.base`.  
To define a new **dynamic net**, we suggest you base yourself on `nets.dynamic.base`.

<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17"> You can read about two types of networks that we developped in [this thesis excerpt](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=18) (pages 18 to 28).  
These networks are available at `nets.dynamic.convolutional` and `nets.dynamic.recurrent`.

### Bots

**Bots** is short for Artificial Agents. **Bots** ought to encapsulate one or more **nets**.

To define a new **static network bot**, we suggest you base yourself on `bots.network.static.base`.  
To define a new **dynamic network bot**, we suggest you base yourself on `bots.network.dynamic.base`.  
If neither of these formats fit your needs, we suggest you then base yourself on `bots.network.base`.  
If you do not want bots to be composed of neural networks, we suggest you then base yourself on `bots.base`.

### Envs

**Envs** is short for Environment. **Envs** are virtual playgrounds for **bots** to evolve and produce behaviour.

To define a new **env** for reinforcement learning, we suggest you base yourself on `envs.multistep.score.base`.  
To define a new **env** for imitation learning, we suggest you base yourself on `envs.multistep.imitate.base`.  
If neither of these formats fit your needs, we suggest you then base yourself on `envs.multistep.base`.  
If you do not want agents to perform multiple steps in the environment, we suggest you then base yourself on `envs.base`.

### IO

IO classes handle anything related to input/output (loading & saving data, bots or seeds). 

To simply load targets (pre-trained agents or recorded behaviour) for imitation learning, we suggest you base yourself on `io.imitate.base`.  
If your task requires performing other types of IO operations, we suggest you instead base yourself on `io.base`.  
The path to the new IO class needs to be specified through the **env** argument `io_path`.

### Other folders

The **data/** folder is for any experiment-related data.  
The **slurm/** folder is made up of scripts for slurm-based execution.  
Finally, the **utils/** folder is composed of various helper functions, python scripts and notebooks.

## How does this library scale with computation ?

We split the computational workload across every process allocated with MPI **(succesfully tested on 1000+ cores)**.  
Each process maintains one **env** through which it mutates and evaluates artificial agents.  
The selection mechanism, however, is handled solely by the main process.

<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17">  Every iteration, information about mutations and evaluation scores is therefore communicated between the main processes and the side processes. We use a seed-based agent-building technique as described in [Such et al., 2017](https://arxiv.org/pdf/1712.06567.pdf) (section 3.1) and propose three different communication protocols:
- a simple protocol wherein a main process scatters seeds and gathers fitnesses. In this case, processes rebuild artificial agents from scratch with an increasingly large list of seeds.  
*To use this communication protocol, include argument* `--communication ps`*.*
- a protocol with better scaling properties (provided bandwidth between processes is not an issue), supplemented with peer-to-peer communication wherein processes exchange fully-formed agents.  
*To use this communication protocol, include argument* `--communication ps_p2p` *(default).*
- a protocol almost identical to the previous one, the difference being that artificial agents do not get broadcasted/gathered to start/finalize experiments (processes load/save agents independently), for experiments with massive population and/or artificial agent sizes.  
*To use this communication protocol, include argument* `--communication big_ps_p2p`*.*

<img src="https://i.imgur.com/8MiXwUN.png" alt="YT" width="20" height="17"> If you want to learn more about these communication protocols, you can read [this thesis excerpt](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=28) (pages 28 to 30).

## What is the template for running experiments ?

#### On a personal computer:

Run the experiment:
```
mpiexec -n <n> python3 main.py --env_path <env_path> \
                               --bots_path <bots_path> \
                               --nb_generations <nb_generations> \
                               --population_size <population_size> \
                               --additional_arguments <additional_arguments>
```

Evaluate the results:
`mpiexec -n <n> python3 utils/evaluate.py --states_path data/states/<env_path>/<additional_arguments>/<bots_path>/<population_size>`

Record the elite:
`python3 utils/record.py --state_path data/states/<env_path>/<additional_arguments>/<bots_path>/<population_size>/<nb_generations>`

For the two commands above:  
1) replace every `/` with a `.` in the paths and remove the file extension.  
Example: `envs/multistep/score/control.py` turns into `envs.multistep.score.control`  
2) <additional_arguments> is a JSON string, it is formatted into a path alphabetically.  
Example: `{"arg_b": 5, "arg_a": "hi"}` turns into `arg_a.hi~arg_b.5`

#### On a Slurm-powered cluster:
Run the experiment:
```
sbatch --account=<account> \
       --mail-user=<mail-user> \
       --nodes=<nodes> \
       --ntasks-per-node=<ntasks-per-node> \
       --mem=<mem> \
       --time=<time> \
       ${SCRATCH}/nevo/slurm/main.sh --env_path <env_path> \
                                     --bots_path <bots_path> \
                                     --nb_generations <nb_generations> \
                                     --population_size <population_size>
```

Evaluate the results:
```
sbatch --account=<account> \
       --mail-user=<mail-user> \
       --nodes=<nodes> \
       --ntasks-per-node=<ntasks-per-node> \
       --mem=<mem> \
       --time=<time> \
       ${SCRATCH}/nevo/slurm/main.sh --states_path ${SCRATCH}/nevo/data/states/<env_path>/<additional_arguments>/<bots_path>/<population_size>
```

## Any pre-built tasks I can try ?

***

### <p align="center">Control Tasks</p>

***

#### Template

```
mpiexec -n <n> python3 main.py --env_path envs/multistep/<goal>/control.py \
                               --bots_path bots/network/<net>/control.py \
                               --nb_generations <nb_generations> \
                               --population_size <population_size> \
                               --additional_arguments '{"task" : "<task>"}'
```

##### Reinforcement Learning
`<goal>` = `score`  
`<task>` = `acrobot`, `cart_pole`, `mountain_car`, `mountain_car_continuous`, `pendulum`, `bipedal_walker`, `bipedal_walker_hardcore`, `lunar_lander`, `lunar_lander_continuous`, `ant`, `half_cheetah`, `hopper`, `humanoid`, `humanoid_standup`, `inverted_double_pendulum`, `inverted_pendulum`, `reacher`, `swimmer` or `walker_2d`  
`<net>` = `static/rnn` or `dynamic/rnn`  

Example:
```
mpiexec -n 2 python3 main.py --env_path envs/multistep/score/control.py \
                             --bots_path bots/network/dynamic/rnn/control.py \
                             --nb_generations 100 \
                             --population_size 16 \
                             --additional_arguments '{"task" : "acrobot"}'
```

##### Imitation Learning

Download the pre-trained agents to imitate:  
`git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo ~/rl-baselines3-zoo`

`<goal>` = `imitate`  
`<task>` = `acrobot`, `cart_pole`, `mountain_car`, `mountain_car_continuous`, `pendulum`, `bipedal_walker`, `bipedal_walker_hardcore`, `lunar_lander`, `lunar_lander_continuous`, `ant`, `half_cheetah`, `hopper`, `humanoid`, `swimmer` or `walker_2d`  
`<net>` = `static/rnn` or `dynamic/rnn`

Example:
```
mpiexec -n 2 python3 main.py --env_path envs/multistep/imitate/control.py \
                             --bots_path bots/network/static/rnn/control.py \
                             --nb_generations 200 \
                             --population_size 64 \
                             --additional_arguments '{"task" : "mountain_car_continuous"}' \
                             --data_path ~/rl-baselines3-zoo/
```

<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17"> You can learn more about how we evolve agents to imitate behaviour in [this thesis excerpt](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=16) (pages 16 to 17).  
<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17"> You can find results of our early Gym Control experiments through [this thesis excerpt](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=31) (pages 31 to 32).

***

### <p align="center">Atari Tasks</p>

***

#### Template

```
mpiexec -n <n> python3 main.py --env_path envs/multistep/<goal>/atari.py \
                               --bots_path bots/network/<net>/atari.py \
                               --nb_generations <nb_generations> \
                               --population_size <population_size> \
                               --additional_arguments '{"task" : "<task>"}'
```

##### Reinforcement Learning:

`<goal>` = `score`  
`<task>` = Any of the 57 Atari games listed [here](https://arxiv.org/pdf/2003.13350.pdf#page=27) (replace spaces with underscores)  
`<net>` = `static/conv_rnn` or `dynamic/conv_rnn`

```
mpiexec -n 2 python3 main.py --env_path envs/multistep/score/atari.py \
                             --bots_path bots/network/static/conv_rnn/atari.py \
                             --nb_generations 100 \
                             --population_size 8 \
                             --additional_arguments '{"task" : "pong"}'
```

##### Imitation Learning:

Download the pre-trained agents to imitate:  
`git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo ~/rl-baselines3-zoo`

`<goal>` = `imitate`  
`<task>` = `asteroids`, `beam_rider`, `breakout`, `enduro`, `pong`, `qbert`, `road_runner`, `seaquest` or `space_invaders`  
`<net>` = `static/conv_rnn` or `dynamic/conv_rnn`

```
mpiexec -n 2 python3 main.py --env_path envs/multistep/imitate/atari.py \
                             --bots_path bots/network/dynamic/conv_rnn/atari.py \
                             --nb_generations 100 \
                             --population_size 8 \
                             --additional_arguments '{"task" : "space_invaders"}' \
                             --data_path ~/rl-baselines3-zoo/
```

<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17"> You can find results of our early Atari experiments through [this slide deck](https://docs.google.com/presentation/d/1s-xtB7cP1ZvxklnRqgjYg8-o-xxAV8hjuSSr7V86F4w/) and [this thesis excerpt](https://papyrus.bib.umontreal.ca/xmlui/bitstream/handle/1866/26072/Le_Clei_Maximilien_2021_memoire.pdf#page=33) (page 33).  
<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17"> You might also be interested in the additional argument `"transfer" : "yes"`, which allows agents to pass on their internal and environment states to their offsprings. You can learn more about it through [this slide deck](https://docs.google.com/presentation/d/1a5EDwKOaonJFNWet9E7oXG_oEYxIpEh-16561inNigI).

***

### <p align="center">Gravity Tasks</p>

***

The goal of these tasks is to get agents to predict/generate frames of a falling ball after seeing its original position.

Download the dataset:  
`wget https://www.dropbox.com/s/1d2a88h9mv31crp/11k.npy -P data/behaviour/gravity/`

#### Template:

```
mpiexec -n <n> python3 main.py --env_path envs/multistep/imitate/gravity.py \
                               --bots_path bots/network/<net>/gravity.py \
                               --nb_generations <nb_generations> \
                               --population_size <population_size> \
                               --additional_arguments '{"task" : "<task>"}'
```

##### Predict & Generate:
`<task>` = `predict` or `generate`  
`<net>` = `static/conv_rnn` or `dynamic/conv_rnn`

Example:
```
mpiexec -n 2 python3 main.py --env_path envs/multistep/imitate/gravity.py \
                             --bots_path bots/network/dynamic/conv_rnn/gravity.py \
                             --nb_generations 100 \
                             --population_size 64 \
                             --additional_arguments '{"task" : "predict"}'
```

<img src="https://i.imgur.com/8MiXwUN.png" alt="x" width="20" height="17"> You can find results (videos) of our experiments in [this google drive folder](https://drive.google.com/drive/folders/1bj3meDEEJqewgZfnG3OB6y1ajV46io_Y?usp=sharing).  
(For very large population sizes, `--communication_protocol big_ps_p2p` is necessary.)


## Contributors

This library was developped with the help of [François Paugam](francois.paugam@laposte.net) during the [CNeuromod project](https://www.cneuromod.ca/) led by [Pierre Bellec](pierre.bellec@gmail.com).
