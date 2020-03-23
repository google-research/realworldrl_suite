# Real-World Reinforcement Learning Challenge Framework

The ["Challenges of Real-World RL"](https://arxiv.org/abs/1904.12901) paper
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

Questions can be directed to the Real-World RL group e-mail:
[realworldrl@google.com]

## Challenges

### Safety
Adds a set of constraints on the task. Returns an additional entry in the
observations ('constraints') in the length of the number of the contraints,
where each entry is True if the constraint is satisfied and False otherwise.

### Delays
Adds actions and observations delays. Actions delay is the number of steps
between passing the action to the environment to when it is actually
performed, and observation delay is the offset of freshness of the returned
observation after performing a step.

### Noise
Adds action or observation noise.  Noise is either: white gaussian, dropped
values, or stuck values. The noise specifications  can be parameterized in
the noise_spec dictionary.

### Non-Stationary Perturbations
Perturbs physical quantities of the environment. These perturbations are
non-stationary and are governed by a scheduler.

### High-dimensionality
Adds extra dummy features to observations to increase dimensionality of the
state space.


## Installation

- (Optional) You may wish to create a
  [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
  to manage your dependencies, so as not to clobber your system installation:

  ```bash
  sudo pip3 install virtualenv
  /usr/local/bin/virtualenv realworldrl_suite
  source ./realworldrl/bin/activate
  ```

- To install `realworldrl_suite`, run the command

  ```bash
  pip3 install git+git://github.com/google_research/realworldrl_suite.git
  ```

  or clone the repository and run

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

If you use `realworldrl_suite` in your work, please cite:

```bash
@article{dulacarnold2020realworldrl,
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
