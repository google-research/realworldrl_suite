# Real-World Reinforcement Learning Challenge Framework

"A Task Suite for Benchmarking Progress on Real World Reinforcement Learning"
describes an evaluation framework and a set of environments that can provide
a good evaluation of an RL algorithm’s potential applicability to real-world systems.

This is the codebase for the RealWorld RL challenge framework.

Currently the challenge is to be comprised of five environments:

* Cartpole
* Walker
* Quadriped
* Manipulator
* Humanoid

The codebase is currently structured as:

* environments/ -- the extended environments
* utils/ -- wrapper classes for logging and standardized evaluations

Questions can be directed to the Real-World RL group e-mail
[RETRACTED].

## Challenges

### Safety
Adds a set of constraints on the task.
Returns an additional entry in the observations ('constraints') in the
length of the number of the constraints, where each entry is True if the
constraint is satisfied and False otherwise.

### Delays
Adds actions, observations, and rewards delays.
Actions delay is the number of steps between passing the action to the
environment to when it is actually performed, and observations (rewards)
delay is the offset of freshness of the returned observation (reward) after
performing a step.

### Noise
Adds action or observation noise.
Different noise include: white Gaussian actions/observations,
dropped actions/observations values, stuck actions/observations values,
or repetitive actions.

### Perturbations
Perturbs physical quantities of the environment. These perturbations are
non-stationary and are governed by a scheduler.

### Dimensionality
Adds extra dummy features to observations to increase dimensionality of the
state space.

Multi-Objective Reward:
Adds additional objectives and specifies objectives interaction (e.g., sum).



## Installation
- Extract zip file to the relevant directory.

- Install pip3: go to website https://pip.pypa.io/en/stable/installing/
Then run the following commands:
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py

- Make sure pip is up to date.

```
pip3 install --upgrade pip
```

- Create a
[Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
to manage your dependencies, so as not to clobber your system installation:

```bash
pip3 install virtualenv
virtualenv realworldrl_venv
source ./realworldrl_venv/bin/activate
```

- Install MuJoCo (see dm_control - https://github.com/deepmind/dm_control).

- To install `realworldrl`,

a) ensure you are in the parent directory of realworldrl_suite

b) run the command

```bash
pip3 install realworldrl_suite/
```
**If you get a Permission denied error, then run
```bash
pip3 install --user <package_name>
```

## Running examples

- Running the examples requires installing the following packages:

```bash
pip3 install tensorflow==1.15.0 dm2gym
pip3 install git+git://github.com/openai/baselines.git
```

- The PPO example can then be run with


```bash
cd realworldrl_suite/examples
mkdir /tmp/rwrl/
python3 run_ppo.py
```

The examples (e.g. run_ppo.py) by default look for the MuJoCo licence key in `~/.mujoco/mjkey.txt`
