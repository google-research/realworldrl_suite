# coding=utf-8
# Copyright 2020 The Real-World RL Suite Authors.
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

"""Real-World Planar Manipulator domain."""

import collections

from dm_control.suite import common
from dm_control.suite import manipulator
from lxml import etree
import numpy as np

from realworldrl_suite.environments import realworld_env
from realworldrl_suite.utils import loggers
from realworldrl_suite.utils import wrappers

_CONTROL_TIMESTEP = .01  # (Seconds)
_TIME_LIMIT = 10  # (Seconds)
_ARM_JOINTS = ['arm_shoulder', 'arm_elbow']  # Used for constraints

PERTURB_PARAMS = ['lower_arm_length', 'root_damping', 'shoulder_damping']


def load(task_name, **task_kwargs):
  return globals()[task_name](**task_kwargs)


# Task Constraints
def joint_angle_constraint(env, safety_vars):
  """Joint angles must be within a certain area of the track."""
  joint_pos = safety_vars['joint_angle']
  return (np.greater(joint_pos,
                     env.limits['joint_angle_constraint'][0]).all() and
          np.less(joint_pos, env.limits['joint_angle_constraint'][1])).all()


def joint_velocity_constraint(env, safety_vars):
  """Joint angle velocities must stay below a certain limit."""
  joint_vels = safety_vars['joint_vels']
  return np.less(np.max(joint_vels), env.limits['joint_velocity_constraint'])


def joint_accel_constraint(env, safety_vars):
  """Joint angle velocities must stay below a certain limit."""
  joint_accels = safety_vars['joint_accels']
  return np.less(np.max(joint_accels), env.limits['joint_accel_constraint'])


def grasp_force_constraint(env, safety_vars):
  """Limit gripper contact forces."""
  return np.less(
      np.max(safety_vars['grasp_force']), env.limits['grasp_force_constraint'])


# Action rate of change constraint.
action_roc_constraint = realworld_env.action_roc_constraint


def gen_task(use_peg,
             insert,
             fully_observable=True,
             time_limit=_TIME_LIMIT,
             random=None,
             log_output=None,
             environment_kwargs=None,
             safety_spec=None,
             delay_spec=None,
             noise_spec=None,
             perturb_spec=None,
             dimensionality_spec=None,
             multiobj_spec=None,
             combined_challenge=None):
  """Returns the Manipulator Bring task with specified real world attributes.

  Args:
    use_peg: A `bool`, whether to replace the ball prop with the peg prop.
    insert: A `bool`, whether to insert the prop in a receptacle.
    fully_observable: A `bool`, whether the observation should contain the
      position and velocity of the object being manipulated and the target
      location.
    time_limit: Integer length of task
    random: Optional, either a `numpy.random.RandomState` instance, an integer
      seed for creating a new `RandomState`, or None to select a seed
      automatically (default).
    log_output: String of path for pickle data logging, None disables logging
    environment_kwargs: additional kwargs for environment
    safety_spec: dictionary that specifies the safety specifications.
    delay_spec: dictionary that specifies the delay.
    noise_spec: dictionary that specifies the noise specifications.
    perturb_spec: dictionary that specifies the perturbations specifications.
    dimensionality_spec: dictionary that specifies extra observation features.
    multiobj_spec: dictionary that specifies complementary objectives.
    combined_challenge: string that can be 'easy', 'medium', or 'hard'.
      Specifying the combined challenge (can't be used with any other spec).
  """
  physics = manipulator.Physics.from_xml_string(
      *manipulator.make_model(use_peg, insert))
  safety_spec = safety_spec or {}
  delay_spec = delay_spec or {}
  noise_spec = noise_spec or {}
  perturb_spec = perturb_spec or {}
  dimensionality_spec = dimensionality_spec or {}
  multiobj_spec = multiobj_spec or {}
  # Check and update for combined challenge.
  (delay_spec, noise_spec,
   perturb_spec, dimensionality_spec) = (
       realworld_env.get_combined_challenge(
           combined_challenge, delay_spec, noise_spec, perturb_spec,
           dimensionality_spec))

  task = RealWorldBring(
      use_peg=use_peg,
      insert=insert,
      fully_observable=fully_observable,
      safety_spec=safety_spec,
      delay_spec=delay_spec,
      noise_spec=noise_spec,
      perturb_spec=perturb_spec,
      dimensionality_spec=dimensionality_spec,
      multiobj_spec=multiobj_spec,
      random=random)

  environment_kwargs = environment_kwargs or {}
  if log_output:
    logger = loggers.PickleLogger(path=log_output)
  else:
    logger = None
  return wrappers.LoggingEnv(
      physics,
      task,
      logger=logger,
      control_timestep=_CONTROL_TIMESTEP,
      time_limit=time_limit,
      **environment_kwargs)


def realworld_bring_ball(fully_observable=True,
                         time_limit=_TIME_LIMIT,
                         random=None,
                         log_output=None,
                         environment_kwargs=None,
                         safety_spec=None,
                         delay_spec=None,
                         noise_spec=None,
                         perturb_spec=None,
                         dimensionality_spec=None,
                         multiobj_spec=None,
                         combined_challenge=None):
  """Returns manipulator bring task with the ball prop."""
  use_peg = False
  insert = False
  return gen_task(use_peg, insert, fully_observable, time_limit, random,
                  log_output, environment_kwargs, safety_spec, delay_spec,
                  noise_spec, perturb_spec, dimensionality_spec, multiobj_spec,
                  combined_challenge)


def realworld_bring_peg(fully_observable=True,
                        time_limit=_TIME_LIMIT,
                        random=None,
                        log_output=None,
                        environment_kwargs=None,
                        safety_spec=None,
                        delay_spec=None,
                        noise_spec=None,
                        perturb_spec=None,
                        dimensionality_spec=None,
                        multiobj_spec=None,
                        combined_challenge=None):
  """Returns manipulator bring task with the peg prop."""
  use_peg = True
  insert = False
  return gen_task(use_peg, insert, fully_observable, time_limit, random,
                  log_output, environment_kwargs, safety_spec, delay_spec,
                  noise_spec, perturb_spec, dimensionality_spec, multiobj_spec,
                  combined_challenge)


def realworld_insert_ball(fully_observable=True,
                          time_limit=_TIME_LIMIT,
                          random=None,
                          log_output=None,
                          environment_kwargs=None,
                          safety_spec=None,
                          delay_spec=None,
                          noise_spec=None,
                          perturb_spec=None,
                          dimensionality_spec=None,
                          multiobj_spec=None,
                          combined_challenge=None):
  """Returns manipulator insert task with the ball prop."""
  use_peg = False
  insert = True
  return gen_task(use_peg, insert, fully_observable, time_limit, random,
                  log_output, environment_kwargs, safety_spec, delay_spec,
                  noise_spec, perturb_spec, dimensionality_spec, multiobj_spec,
                  combined_challenge)


def realworld_insert_peg(fully_observable=True,
                         time_limit=_TIME_LIMIT,
                         random=None,
                         log_output=None,
                         environment_kwargs=None,
                         safety_spec=None,
                         delay_spec=None,
                         noise_spec=None,
                         perturb_spec=None,
                         dimensionality_spec=None,
                         multiobj_spec=None,
                         combined_challenge=None):
  """Returns manipulator insert task with the peg prop."""
  use_peg = True
  insert = True
  return gen_task(use_peg, insert, fully_observable, time_limit, random,
                  log_output, environment_kwargs, safety_spec, delay_spec,
                  noise_spec, perturb_spec, dimensionality_spec, multiobj_spec,
                  combined_challenge)


class Physics(manipulator.Physics):
  """Inherits from manipulator.Physics."""


class RealWorldBring(realworld_env.Base, manipulator.Bring):
  """A Manipulator task with real-world specifications.

  Subclasses dm_control.suite.manipulator.

  Safety:
    Adds a set of constraints on the task.
    Returns an additional entry in the observations ('constraints') in the
    length of the number of the constraints, where each entry is True if the
    constraint is satisfied and False otherwise.

  Delays:
    Adds actions, observations, and rewards delays.
    Actions delay is the number of steps between passing the action to the
    environment to when it is actually performed, and observations (rewards)
    delay is the offset of freshness of the returned observation (reward) after
    performing a step.

  Noise:
    Adds action or observation noise.
    Different noise include: white Gaussian actions/observations,
    dropped actions/observations values, stuck actions/observations values,
    or repetitive actions.

  Perturbations:
    Perturbs physical quantities of the environment. These perturbations are
    non-stationary and are governed by a scheduler.

  Dimensionality:
    Adds extra dummy features to observations to increase dimensionality of the
    state space.

  Multi-Objective Reward:
    Adds additional objectives and specifies objectives interaction (e.g., sum).
  """

  def __init__(self,
               use_peg,
               insert,
               fully_observable,
               safety_spec,
               delay_spec,
               noise_spec,
               perturb_spec,
               dimensionality_spec,
               multiobj_spec,
               random=None,
               **kwargs):
    """Initialize the RealWorldBring task.

    Args:
      use_peg: A `bool`, whether to replace the ball prop with the peg prop.
      insert: A `bool`, whether to insert the prop in a receptacle.
      fully_observable: A `bool`, whether the observation should contain the
        position and velocity of the object being manipulated and the target
        location.
      safety_spec: dictionary that specifies the safety specifications of the
        task. It may contain the following fields:
        enable- bool that represents whether safety specifications are enabled.
        constraints- list of class methods returning boolean constraint
          satisfactions.
        limits- dictionary of constants used by the functions in 'constraints'.
        safety_coeff - a scalar between 1 and 0 that scales safety constraints,
          1 producing the base constraints, and 0 likely producing an
          unsolveable task.
        observations- a default-True boolean that toggles the whether a vector
          of satisfied constraints is added to observations.
      delay_spec: dictionary that specifies the delay specifications of the
        task. It may contain the following fields:
        enable- bool that represents whether delay specifications are enabled.
        actions- integer indicating the number of steps actions are being
          delayed.
        observations- integer indicating the number of steps observations are
          being delayed.
        rewards- integer indicating the number of steps observations are being
          delayed.
      noise_spec: dictionary that specifies the noise specifications of the
        task. It may contains the following fields:
        gaussian- dictionary that specifies the white Gaussian additive noise.
          It may contain the following fields:
          enable- bool that represents whether noise specifications are enabled.
          actions- float inidcating the standard deviation of a white Gaussian
            noise added to each action.
          observations- similarly, additive white Gaussian noise to each
            returned observation.
        dropped- dictionary that specifies the dropped values noise.
          It may contain the following fields:
          enable- bool that represents whether dropped values specifications are
            enabled.
          observations_prob- float in [0,1] indicating the probability of
            dropping each observation component independently.
          observations_steps- positive integer indicating the number of time
            steps of dropping a value (setting to zero) if dropped.
          actions_prob- float in [0,1] indicating the probability of dropping
            each action component independently.
          actions_steps- positive integer indicating the number of time steps of
            dropping a value (setting to zero) if dropped.
        stuck- dictionary that specifies the stuck values noise.
          It may contain the following fields:
          enable- bool that represents whether stuck values specifications are
            enabled.
          observations_prob- float in [0,1] indicating the probability of each
            observation component becoming stuck.
          observations_steps- positive integer indicating the number of time
            steps an observation (or components of) stays stuck.
          actions_prob- float in [0,1] indicating the probability of each
            action component becoming stuck.
          actions_steps- positive integer indicating the number of time
            steps an action (or components of) stays stuck.
        repetition- dictionary that specifies the repetition statistics.
          It may contain the following fields:
          enable- bool that represents whether repetition specifications are
            enabled.
          actions_prob- float in [0,1] indicating the probability of the actions
            to be repeated in the following steps.
          actions_steps- positive integer indicating the number of time steps of
            repeating the same action if it to be repeated.
      perturb_spec: dictionary that specifies the perturbation specifications
        of the task. It may contain the following fields:
        enable- bool that represents whether perturbation specifications are
          enabled.
        period- int, number of episodes between updates perturbation updates.
        param- string indicating which parameter to perturb (currently
          supporting lower_arm_length, root_damping, shoulder_damping).
        scheduler- string inidcating the scheduler to apply to the perturbed
          parameter (currently supporting constant, random_walk, drift_pos,
          drift_neg, cyclic_pos, cyclic_neg, uniform, and saw_wave).
        start - float indicating the initial value of the perturbed parameter.
        min - float indicating the minimal value the perturbed parameter may be.
        max - float indicating the maximal value the perturbed parameter may be.
        std - float indicating the standard deviation of the white noise for the
          scheduling process.
      dimensionality_spec: dictionary that specifies the added dimensions to the
        state space. It may contain the following fields:
        enable- bool that represents whether dimensionality specifications are
          enabled.
        num_random_state_observations - num of random (unit Gaussian)
          observations to add.
      multiobj_spec: dictionary that sets up the multi-objective challenge.
        The challenge works by providing an `Objective` object which describes
        both numerical objectives and a reward-merging method that allow to both
        observe the various objectives in the observation and affect the
        returned reward in a manner defined by the Objective object.
        enable- bool that represents whether delay multi-objective
          specifications are enabled.
        objective - either a string which will load an `Objective` class from
          utils.multiobj_objectives.OBJECTIVES, or an Objective object which
          subclasses utils.multiobj_objectives.Objective.
        reward - boolean indicating whether to add the multiobj objective's
          reward to the environment's returned reward.
        coeff - a number in [0,1] that is passed into the Objective object to
          change the mix between the original reward and the Objective's
          rewards.
        observed - boolean indicating whether the defined objectives should be
          added to the observation.
      random: Optional, either a `numpy.random.RandomState` instance, an integer
        seed for creating a new `RandomState`, or None to select a seed
        automatically (default).
      **kwargs: extra parameters passed to parent class (manipulator.Bring)
    """
    # Initialize parent classes.
    realworld_env.Base.__init__(self)
    manipulator.Bring.__init__(
        self, use_peg, insert, fully_observable, random=random, **kwargs)

    # Safety setup.
    self._setup_safety(safety_spec)

    # Delay setup.
    realworld_env.Base._setup_delay(self, delay_spec)

    # Noise setup.
    realworld_env.Base._setup_noise(self, noise_spec)

    # Perturb setup.
    self._setup_perturb(perturb_spec)

    # Dimensionality setup
    realworld_env.Base._setup_dimensionality(self, dimensionality_spec)

    # Multi-objective setup
    realworld_env.Base._setup_multiobj(self, multiobj_spec)

    self._use_peg = use_peg
    self._insert = insert

  # Safety methods.
  def _setup_safety(self, safety_spec):
    """Setup for the safety specifications of the task."""
    self._safety_enabled = safety_spec.get('enable', False)
    self._safety_observed = safety_spec.get('observations', True)

    if self._safety_enabled:
      # Add safety specifications.
      if 'constraints' in safety_spec:
        self.constraints = safety_spec['constraints']
      else:
        self.constraints = collections.OrderedDict([
            ('joint_angle_constraint', joint_angle_constraint),
            ('joint_velocity_constraint', joint_velocity_constraint),
            ('joint_accel_constraint', joint_accel_constraint),
            ('grasp_force_constraint', grasp_force_constraint)
        ])
      if 'limits' in safety_spec:
        self.limits = safety_spec['limits']
      else:
        if 'safety_coeff' in safety_spec:
          if safety_spec['safety_coeff'] < 0 or safety_spec['safety_coeff'] > 1:
            raise ValueError(
                'safety_coeff should be in [0,1], but got {}'.format(
                    safety_spec['safety_coeff']))
          safety_coeff = safety_spec['safety_coeff']
        else:
          safety_coeff = 1
        self.limits = {
            'joint_angle_constraint':
                safety_coeff * np.array([[-160, -140], [160, 140]]) *
                np.pi / 180.,  # rad
            'joint_velocity_constraint':
                safety_coeff * 10,  #  rad/s
            'joint_accel_constraint':
                safety_coeff * 3600,  # rad/s^2
            'grasp_force_constraint':
                safety_coeff * 5,  # newtons
            'action_roc_constraint': safety_coeff * 1.5
        }
      self._constraints_obs = np.ones(len(self.constraints), dtype=bool)

  def safety_vars(self, physics):
    """Centralized retrieval of safety-related variables to simplify logging."""
    safety_vars = collections.OrderedDict(
        joint_angle=physics.named.data.qpos[_ARM_JOINTS].copy(),  # rad
        joint_vels=np.abs(physics.named.data.qvel[_ARM_JOINTS]).copy(),  # rad/s
        joint_accels=np.abs(
            physics.named.data.qacc[_ARM_JOINTS]).copy(),  # rad/s^2
        grasp_force=physics.touch(),  # log(1+newtons)
        actions=physics.control(),
    )
    return safety_vars

  def _setup_perturb(self, perturb_spec):
    """Setup for the perturbations specification of the task."""
    self._perturb_enabled = perturb_spec.get('enable', False)
    self._perturb_period = perturb_spec.get('period', 1)

    if self._perturb_enabled:
      # Add perturbations specifications.
      self._perturb_param = perturb_spec.get('param', 'lower_arm_length')
      # Making sure object to perturb is supported.
      if self._perturb_param not in PERTURB_PARAMS:
        raise ValueError("""param was: {}. Currently only supporting {}.
        """.format(self._perturb_param, PERTURB_PARAMS))

      # Setting perturbation function.
      self._perturb_scheduler = perturb_spec.get('scheduler', 'constant')
      if self._perturb_scheduler not in realworld_env.PERTURB_SCHEDULERS:
        raise ValueError("""scheduler was: {}. Currently only supporting {}.
        """.format(self._perturb_scheduler, realworld_env.PERTURB_SCHEDULERS))

      # Setting perturbation process parameters.
      if self._perturb_param == 'lower_arm_length':
        self._perturb_cur = perturb_spec.get('start', 0.12)
        self._perturb_start = perturb_spec.get('start', 0.12)
        self._perturb_min = perturb_spec.get('min', 0.1)
        self._perturb_max = perturb_spec.get('max', 0.25)
        self._perturb_std = perturb_spec.get('std', 0.01)
      elif self._perturb_param == 'root_damping':
        self._perturb_cur = perturb_spec.get('start', 2.0)
        self._perturb_start = perturb_spec.get('start', 2.0)
        self._perturb_min = perturb_spec.get('min', 0.1)
        self._perturb_max = perturb_spec.get('max', 10.0)
        self._perturb_std = perturb_spec.get('std', 0.1)
      elif self._perturb_param == 'shoulder_damping':
        self._perturb_cur = perturb_spec.get('start', 1.5)
        self._perturb_start = perturb_spec.get('start', 1.5)
        self._perturb_min = perturb_spec.get('min', 0.1)
        self._perturb_max = perturb_spec.get('max', 10.0)
        self._perturb_std = perturb_spec.get('std', 0.1)

  def update_physics(self):
    """Returns a new Physics object with perturbed parameter."""
    # Generate the new perturbed parameter.
    realworld_env.Base._generate_parameter(self)

    # Create new physics object with the perturb parameter.
    xml_string = manipulator.make_model(self._use_peg, self._insert)[0]
    mjcf = etree.fromstring(xml_string)

    if self._perturb_param == 'lower_arm_length':
      lower_arm = mjcf.find('./worldbody/body/body/body/geom')
      lower_arm.set('fromto', '0 0 0 0 0 {}'.format(self._perturb_cur))
      hand = mjcf.find('./worldbody/body/body/body/body')
      hand.set('pos', '0 0 {}'.format(self._perturb_cur))
    elif self._perturb_param == 'root_damping':
      joints = mjcf.findall('./worldbody/body/joint')
      for joint in joints:
        if joint.get('name') == 'arm_root':
          joint.set('damping', str(self._perturb_cur))
    elif self._perturb_param == 'shoulder_damping':
      shoulder_joint = mjcf.find('./worldbody/body/body/joint')
      shoulder_joint.set('damping', str(self._perturb_cur))

    xml_string_modified = etree.tostring(mjcf, pretty_print=True)
    physics = Physics.from_xml_string(xml_string_modified, common.ASSETS)
    return physics

  def before_step(self, action, physics):
    """Updates the environment using the action and returns a `TimeStep`."""
    self._last_action = physics.control()
    action_min = self.action_spec(physics).minimum[:]
    action_max = self.action_spec(physics).maximum[:]
    action = realworld_env.Base.before_step(self, action, action_min,
                                            action_max)
    manipulator.Bring.before_step(self, action, physics)

  def after_step(self, physics):
    realworld_env.Base.after_step(self, physics)
    manipulator.Bring.after_step(self, physics)
    self._last_action = None
