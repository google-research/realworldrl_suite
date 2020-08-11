# RWRL Suite Challenge Spec

This document outlines the specification dictionaries for each of the supported
challenges.  It provides an overview of each challenge's parameters and their
effects.


### Safety

Adds a set of constraints on the task. Returns an additional entry in the
observations under the `constraints` key. The observation is a binary vectory
and is the length of the number of the contraints, where each entry is True if
the constraint is satisfied and False otherwise. The following dictionary is fed
as an argument into the RWRL environment load function ot intialize safety
constraints:

```
safety_spec = {
  'enable': bool, # Whether to enable safety constraints.
  'observations': bool, # Whether to add the constraint violations observation.
  'safety_coeff': float, # Safety coefficient that regulates the difficulty of the constraint.  1 is the baseline, and 0 is impossible (always violated).
  'constraints' : list # Optional list of additional safety constraints.  Can only operate on variables returned by the `safety_vars` method.
}
```

Each of the built-in constraints has been tuned to be just at the border of
nominal operation. For tasks such as Cartpole it is tuned to be violated only
during the swingup phase, but not violated during balancing. Different
constraints will be more or less difficult to satisfy.

The built-in constraints are as follows:

* Cartpole Swing-Up:
  * `slider_pos_constraint` : Constrain cart to be within a specific region on track.
  * `balance_velocity_constraint` : Constrain pole angular velocity to be below a certain threshold when arriving near the top.  This provides a more subtle constraint than a standard box constraint on a variable.
  * `slider_accel_constraint` : Constrain cart acceleration to be below a certain value.
* Walker:
  * `joint_angle_constraint` : Constrain joint angles to specific ranges.  This is joint-specific.
  * `joint_velocity_constraint` : Constrain joint velocities to a certain range. This is a global value.
  * `dangerous_fall_constraint` : Discourage dangerous falls by ensuring the torso stays positioned forwards.
  * `torso_upright_constraint` : Discourage dangerous operation by ensuring that the torso stays upright.
* Quadruped:
  * `joint_angle_constraint` : Constrain joint angles to specific ranges.  This is joint-specific.
  * `joint_velocity_constraint` : Constrain joint velocities to a certain range. This is a global value.
  * `upright_constraint` : Constrain the Quadruped's torso's z-axis to be oriented upwards.
  * `foot_force_constraint` : Limits foot contact forces when touching the ground.
* Humanoid
  * `joint_angle_constraint` : Constrain joint angles to specific ranges.  This is joint-specific.
  * `joint_velocity_constraint` : Constrain joint velocities to a certain range. This is a global value.
  * `upright_constraint` : Constrain the Humanoid's pelvis's z-axis to be oriented upwards.
  * `foot_force_constraint` : Limits foot contact forces when touching the ground.
  * `dangerous_fall_constraint` : Discourages dangerous falls by limiting head and torso contact.

It is also possible to add in arbitrary constraints by passing in a list of methods to the `observations` key, which will receive the values returned by the task's `safety_vars` method.


### Delays
This challenge provides delays on actions, observations and rewards.  For each of these, the delayed element is placed into a buffer and either applied to the environment, or sent back to the agent after the specified number of steps.

This can be configured by passing in a `delay_spec` dictionary of the following form:

```
delay_spec = {
  'enable': bool, # Whether to enable this challenge
  'actions': int, # The delay on actions in # of steps
  'observations': int, # The delay on observations in # of steps
  'rewards': int, # The delay on actions in # of steps
}
```

### Noise

Real-world systems often have action and observation noise, and this can come in
many flavors, from simply noisy values, to dropped and stuck signals. This
challenge allows you to experiment with different types of noise:

- White Gaussian action/observation noise
- Dropped actions/observations
- Stuck actions/observations
- Repetitive actions 

The noise specifications can be parameterized in the noise_spec dictionary.

```
noise_spec = {
        'gaussian': { # Specifies the white Gaussian additive noise.
            'enable': bool, # Whether to enable Gaussian noise.
            'actions': float, # Standard deviation of noise added to actions.
            'observations': float, # Standard deviation of noise added to observations.
        },
        'dropped': { # Specifies dropped value noise.
            'enable': bool, # Whether to enable dropped values.
            'observations_prob': float, # Value in [0,1] indicating the probability of dropping each observation independently.
            'observations_steps': int, # Value > 0 specifying the number of steps to drop an observation for if dropped.
            'action_prob': float # Value in [0,1] indicating the probability of dropping each action independently.
            'action_steps': int, # Value > 0 specifying the number of steps to drop an action for if dropped.
        },
        'stuck': { # Specifies stuck values noise.
            'enable': bool, # Whether to enable stuck values.
            'observations_prob': float, # Value in [0,1] indicating the probability of an observation component becoming stuck.
            'observations_steps': int, # Value > 0 specifying the number of steps an observation remains stuck.
            'action_prob': float, # Value in [0,1] indicating the probability of an action component becoming stuck.
            'action_steps': int # Value > 0 specifying the number of steps an action remains stuck.
        },
        'repetition': { # Specifies repetition statistics.
            'enable': bool, # Whether to enable repeating values.
            'actions_prob': float, # Value in [0,1] indicating the probability of an action repeating.
            'actions_steps': int # Value > 0 specifying the number of steps an action repeats.
        },
    }

```

### Perturbations

Real systems are imperfect and degrade or change over time. This means the
controller needs to understand these changes and update its control policy
accordingly. The RWRL suite can simulate various system perturbations, with
varying ways in which a perturbation evolves over time (which we call its
'schedule').

These challenges can be configured by passing in a `pertub_spec` dictionary with the
following format:

```
perturb_spec = {
    'enable': bool, # Whether to enable perturbations
    'period': int, # Number of episodes between perturbation changes.
    'param': str, # Specifies which parameter to perturb. Specified below.
    'scheduler': str, # Specifies which scheduler to apply. Specified below.
    'start': float, # Indicates initial value of perturbed parameter.
    'min': float, # Indicates the minimal value of the perturbed parameter.
    'max': float, # Indicates the maximal value of the perturbed parameter.
    'std': float # Indicates the standard deviation of white noise used in scheduling.
}
```

The various scheduler choices are as follows:

* `constant` : keeps the perturbation constant.
* `random_walk`: change the perturbation by a random amount (defined by the 'std' key).
* `drift_pos` : change the perturbation by a random positive amount.
* `drift_neg` : change the perturbation by a random negative amount.
* `cyclic_pos` : change the perturbation by a random positive amount and cycle back when 'max' is attained.
* `cyclic_neg` : change the perturbation by a random negative amount and cycle back when 'min' is attained.
* `uniform` : set the perturbation to a uniform random value within [min, max].
* `saw_wave` : cycle between `drift_pos` and `drift_neg` when [min, max] bounds are reached.

Each environment has a set of parameters which can be perturbed:

* Cartpole
  * `pole_length`
  * `pole_mass`
  * `joint_damping` : adds a damping factor to the pole joint.
  * `slider_damping` : adds a damping factor to the slider (cart).
* Walker
  * `thigh_length`
  * `torso_length`
  * `joint_damping` : adds a damping factor to all joints.
  * `contact_friction` : alters contact friction with ground.
* Quadruped
  * `shin_length`
  * `torso_density`: alters torso density, therefore changing weight with constant volume.
  * `joint_damping` : adds a damping factor to all joints.
  * `contact_friction` : alters contact friction with ground.
* Humanoid
  * `joint_damping` : adds a damping factor to all joints.
  * `contact_friction` : alters contact friction with ground.
  * `head_size` : alters head size (and therefore weight).


### Dimensionality
Adds extra dummy features to observations to increase dimensionality of the
state space.

```
dimensionality_spec = {
    'enable': bool, # Whether to enable dimensionality challenges.
    'num_random_state_observations': int, # Number of random observation dimension to add.
}
```

### Multi-Objective Reward
This challenge looks at multi-objective rewards.  There is a default multi-objective reward included which allows a safety objective to be defined from the set of constraints, but new objectives are easy to implement by adding them to the `utils.multiobj_objectives.OBJECTIVES` dict.

```
multiobj_spec = {
  'enable': bool, # Whether to enable the multi-objective challenge.
  'objective': str or object, # Either a string which will load an `Objective` class from
                              # utils.multiobj_objectives.OBJECTIVES or an Objective object
                              # which subclasses utils.multiobj_objectives.Objective.
  'reward': bool, # Whether to add the multiobj objective's reward to the environment's returned reward.
  'coeff': float, # A number in [0,1] used as a reward mixing ratio by the Objective object.
  'observed': bool # Whether the defined objectives should be added to the observation.
}
```

### RWRL Combined Challenge Benchmarks:
The RWRL suite allows you to combine multiple challenges into the same environment. The challenges are
divided into 'Easy', 'Medium' and 'Hard' which depend on the magnitude of the
challenge effects applied along each challenge dimension.

* The 'Easy' challenge:

```
delay_spec = {
        'enable': True,
        'actions': 3,
        'observations': 3,
        'rewards': 10
    }
    noise_spec = {
        'gaussian': {
            'enable': True,
            'actions': 0.1,
            'observations': 0.1
        },
        'dropped': {
            'enable': True,
            'observations_prob': 0.01,
            'observations_steps': 1,
        },
        'stuck': {
            'enable': True,
            'observations_prob': 0.01,
            'observations_steps': 1,
        },
        'repetition': {
            'enable': True,
            'actions_prob': 1.0,
            'actions_steps': 1
        }
    }
    perturb_spec = {
        'enable': True,
        'period': 1,
        'scheduler': 'uniform'
    }
    dimensionality_spec = {
        'enable': True,
        'num_random_state_observations': 10
    }
```

* The 'Medium' challenge:

```
    delay_spec = {
        'enable': True,
        'actions': 6,
        'observations': 6,
        'rewards': 20
    }
    noise_spec = {
        'gaussian': {
            'enable': True,
            'actions': 0.3,
            'observations': 0.3
        },
        'dropped': {
            'enable': True,
            'observations_prob': 0.05,
            'observations_steps': 5,
        },
        'stuck': {
            'enable': True,
            'observations_prob': 0.05,
            'observations_steps': 5,
        },
        'repetition': {
            'enable': True,
            'actions_prob': 1.0,
            'actions_steps': 2
        }
    }
    perturb_spec = {
        'enable': True,
        'period': 1,
        'scheduler': 'uniform'
    }
    dimensionality_spec = {
        'enable': True,
        'num_random_state_observations': 20
    }
```

 * The 'Hard' challenge:
 
```
    delay_spec = {
        'enable': True,
        'actions': 9,
        'observations': 9,
        'rewards': 40
    }
    noise_spec = {
        'gaussian': {
            'enable': True,
            'actions': 1.0,
            'observations': 1.0
        },
        'dropped': {
            'enable': True,
            'observations_prob': 0.1,
            'observations_steps': 10,
        },
        'stuck': {
            'enable': True,
            'observations_prob': 0.1,
            'observations_steps': 10,
        },
        'repetition': {
            'enable': True,
            'actions_prob': 1.0,
            'actions_steps': 3
        }
    }
    perturb_spec = {
        'enable': True,
        'period': 1,
        'scheduler': 'uniform'
    }
    dimensionality_spec = {
        'enable': True,
        'num_random_state_observations': 50
    }
```
