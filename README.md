# Real-World Reinforcement Learning (RWRL) Challenge Framework

<p align="center">
  <img src="docs/img/angular_velocity.gif" height="150px"/><img src="docs/img/humanoid_perturbations.gif" height="150px">
</p>

The ["Challenges of Real-World RL"](https://arxiv.org/abs/1904.12901) paper
identifies and describes a set of nine challenges that are currently preventing
Reinforcement Learning (RL) agents from being utilized on real-world
applications and products. It also describes an evaluation framework and a set
of environments that can provide an evaluation of an RL algorithm’s potential
applicability to real-world systems. It has since then been followed up with
["An Empirical Investigation of the challenges of real-world reinforcement
learning"](https://arxiv.org/pdf/2003.11881.pdf) which implements eight of the
nine described challenges (excluding explainability) and analyses their effects
on various state-of-the-art RL algorithms. This is the codebase used to perform
this analysis, and is also intended as a common platform for easily reproducible
experimentation around these challenges, it is referred to as the
`realworldrl-suite` (Real-World Reinforcement Learning (RWRL) Suite).

Currently the suite is to comprised of five environments:

*   Cartpole
*   Walker
*   Quadriped
*   Manipulator (less tested)
*   Humanoid

The codebase is currently structured as:

*   environments/ -- the extended environments
*   utils/ -- wrapper classes for logging and standardized evaluations
*   analysis/ -- Notebook for training an agent and generating plots
*   examples/ -- Random policy and PPO agent example implementations
*   docs/ -- Documentation

Questions can be directed to the Real-World RL group e-mail
[realworldrl@google.com].

> :information_source: If you wish to test your agent in a principled fashion on
> related challenges in low-dimensional domains, we highly recommend using
> [bsuite](https://github.com/deepmind/bsuite).

## Documentation

We overview the challenges here, but more thorough documentation on how to
configure each challenge can be found [here](docs/README.md). 

Starter examples are presented in the [examples](#running-examples) section.

## Challenges

### Safety

Adds a set of constraints on the task. Returns an additional entry in the
observations ('constraints') in the length of the number of the contraints,
where each entry is True if the constraint is satisfied and False otherwise.

### Delays

Action, observation and reward delays.

-   Action delay is the number of steps between passing the action to the
    environment to when it is actually performed.
-   Observation delay is the offset of freshness of the returned observation
    after performing a step.
-   Reward delay indicates the number of steps before receiving a reward after
    taking an action.

### Noise

Action and observation noise. Different noise include:

-   White Gaussian action/observation noise
-   Dropped actions/observations
-   Stuck actions/observations
-   Repetitive actions

The noise specifications can be parameterized in the noise_spec dictionary.

### Perturbations

Perturbs physical quantities of the environment. These perturbations are
non-stationary and are governed by a scheduler.

### Dimensionality

Adds extra dummy features to observations to increase dimensionality of the
state space.

### Multi-Objective Rewards:

Adds additional objectives and specifies objectives interaction (e.g., sum).

### Offline Learning

We provide our offline datasets through the
[RL Unplugged](https://github.com/deepmind/deepmind-research/blob/master/rl_unplugged/)
library. There is an
[example](https://github.com/deepmind/deepmind-research/blob/master/rl_unplugged/rwrl_example.py)
and an associated
[colab](https://github.com/deepmind/deepmind-research/blob/master/rl_unplugged/rwrl_d4pg.ipynb).

### RWRL Combined Challenge Benchmarks:

Combines multiple challenges into the same environment. The challenges are
divided into 'Easy', 'Medium' and 'Hard' which depend on the magnitude of the
challenge effects applied along each challenge dimension.

## Installation

-   Install pip:
-   Run the following commands:

    ```bash
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    python get-pip.py
    ```

-   Make sure pip is up to date.

    ```bash
    pip3 install --upgrade pip
    ```

-   (Optional) You may wish to create a
    [Python virtual environment](https://docs.python.org/3/tutorial/venv.html)
    to manage your dependencies, so as not to clobber your system installation:

    ```bash
    sudo pip3 install virtualenv
    /usr/local/bin/virtualenv realworldrl_suite
    source ./realworldrl/bin/activate
    ```

-   Install MuJoCo (see dm_control - https://github.com/deepmind/dm_control).

-   To install `realworldrl_suite`:

    -   Clone the repository by running:

    ```bash
    git clone https://github.com/google-research/realworldrl_suite.git
    ```

    -   Ensure you are in the parent directory of realworldrl_suite
    -   Run the command:

    ```bash
    pip3 install realworldrl_suite/
    ```

## Running examples

We provide three example agents: a random agent, a PPO agent, and an
[ACME](https://github.com/deepmind/acme)-based DMPO agent.

-   For PPO, running the examples requires installing the following packages:

    ```bash
    pip3 install tensorflow==1.15.0 dm2gym
    pip3 install git+git://github.com/openai/baselines.git
    ```

-   The PPO example can then be run with

    ```bash
    cd realworldrl_suite/examples
    mkdir /tmp/rwrl/
    python3 run_ppo.py
    ```

-   For DMPO, one can run the example by installing the following packages:

    ```bash
    pip install dm-acme
    pip install dm-acme[reverb]
    pip install dm-acme[tf]
    ```

    You *may* also have to install the following:

    ```bash
    pip install gym
    pip install jax
    pip install dm-sonnet
    ```

-   The examples look for the MuJoCo licence key in `~/.mujoco/mjkey.txt` by
    default.

## RWRL Combined Challenge Benchmark Instantiation:

As mentioned above, these benchmark challenges are divided into 'Easy', 'Medium'
and 'Hard' difficulty levels. For the current state-of-the-art performance on
these benchmarks, please see <a href="https://arxiv.org/abs/2003.11881">this</a>
paper.

Instantiating a combined challenge environment with 'Easy' difficulty is done as
follows:

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

-   <a href="https://arxiv.org/abs/1904.12901">Challenges of real-world
    reinforcement learning</a>

-   <a href="https://arxiv.org/abs/2003.11881">An empirical investigation of the
    challenges of real-world reinforcement learning</a>
