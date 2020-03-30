# Real-World Reinforcement Learning (RWRL) Challenge Framework

The ["Challenges of Real-World RL"](https://arxiv.org/abs/1904.12901) paper
describes an evaluation framework and a set of environments that can provide
a good evaluation of an RL algorithm’s potential applicability to real-world 
systems.

This is the codebase for the RWRL challenge framework and is referred to as the
`realworldrl-suite` (Real World Reinforcement Learning Suite).

Currently the suite is to be comprised of five environments:

* Cartpole
* Walker
* Quadriped
* Manipulator
* Humanoid

The codebase is currently structured as:

* environments/ -- the extended environments
* utils/ -- wrapper classes for logging and standardized evaluations
* analysis/ -- Notebook for training an agent and generating plots
* examples/ -- Random policy and PPO agent example implementations

Questions can be directed to the Real-World RL group e-mail
[realworldrl@google.com].

Note: If you wish to test your agent in a principled fashion on related 
challenges in low-dimensional domains, we highly recommend 
using `bsuite` (https://github.com/deepmind/bsuite).

## Challenges

### Safety
Adds a set of constraints on the task. Returns an additional entry in the
observations ('constraints') in the length of the number of the contraints,
where each entry is True if the constraint is satisfied and False otherwise.

### Delays
Action, observation and reward delays.

- Action delay is the number of steps between passing the action to the
  environment to when it is actually performed.
- Observation delay is the offset of freshness of the returned observation 
  after performing a step. 
- Reward delay indicates the number of steps before receiving a reward after 
  taking an action.

### Noise
Action and observation noise. Different noise include:

- White Gaussian action/observation noise
- Dropped actions/observations
- Stuck actions/observations
- Repetitive actions 

The noise specifications can be parameterized in the noise_spec dictionary.

### Perturbations
Perturbs physical quantities of the environment. These perturbations are
non-stationary and are governed by a scheduler.

### Dimensionality
Adds extra dummy features to observations to increase dimensionality of the
state space.

### Multi-Objective Reward:
Adds additional objectives and specifies objectives interaction (e.g., sum).

### RWRL Combined Challenge Benchmarks:
Combines multiple challenges into the same environment. The challenges are
divided into 'Easy', 'Medium' and 'Hard' which depend on the magnitude of the
challenge effects applied along each challenge dimension.


## Installation

- Install pip:
- Run the following commands:

  ```bash
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python get-pip.py
  ```

- Make sure pip is up to date.

  ```bash
  pip3 install --upgrade pip
  ```

- (Optional) You may wish to create a
  [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
  to manage your dependencies, so as not to clobber your system installation:

  ```bash
  sudo pip3 install virtualenv
  /usr/local/bin/virtualenv realworldrl_suite
  source ./realworldrl/bin/activate
  ```
- Install MuJoCo (see dm_control - https://github.com/deepmind/dm_control).

- To install `realworldrl_suite`, run the command

  ```bash
  pip3 install git+git://github.com/google_research/realworldrl_suite.git
  ```

or clone the repository and:
  
- Ensure you are in the parent directory of realworldrl_suite

- Run the command:

  ```bash
  pip3 install realworldrl_suite/
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

- The examples (e.g. run_ppo.py) by default look for the MuJoCo licence key in 
`~/.mujoco/mjkey.txt`

## RWRL Combined Challenge Benchmark Instantiation:
As mentioned above, these benchmark challenges are
divided into 'Easy', 'Medium' and 'Hard' difficulty levels. For the current
state-of-the-art performance on these benchmarks, please see
<a href="https://arxiv.org/abs/2003.11881">this</a> paper.

Instantiating a combined challenge environment with 'Easy' difficulty is done
as follows:

```python
import realworldrl_suite.environments as rwrl
env = rwrl.load(
    domain_name='cartpole',
    task_name='realworld_swingup',
    combined_challenge='easy',
    log_output='/tmp/path/to/results.npz',
    environment_kwargs=dict(log_safety_vars=True, flat_observation=True))
```

## Acknowledgements

  If you use `realworldrl_suite` in your work, please cite:

  ```
  @article{dulacarnold2020realworldrlempirical,
           title={An empirical investigation of the challenges of real-world reinforcement learning},
           author={Dulac-Arnold, Gabriel and
                   Levine, Nir and
                   Mankowitz, Daniel J. and
                   Li, Jerry and
                   Paduraru, Cosmin and
                   Gowal, Sven and
                   Hester, Todd
                   },
           year={2020},
  }
  ```

## Paper links
- <a href="https://arxiv.org/abs/1904.12901">Challenges of real-world reinforcement learning</a>

- <a href="https://arxiv.org/abs/2003.11881">An empirical investigation of the challenges of real-world reinforcement learning</a>

