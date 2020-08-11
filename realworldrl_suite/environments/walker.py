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

"""Real-World Planar Walker Domain."""

import collections

from dm_control.suite import common
from dm_control.suite import walker
from lxml import etree
import numpy as np

from realworldrl_suite.environments import realworld_env
from realworldrl_suite.utils import loggers
from realworldrl_suite.utils import wrappers

_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = .025

# Horizontal speeds (meters/second) above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 8
_JOINT_NAMES = [
    'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle',
    'right_ankle'
]

PERTURB_PARAMS = [
    'thigh_length', 'torso_length', 'joint_damping', 'contact_friction'
]


def load(task_name, **task_kwargs):
  return globals()[task_name](**task_kwargs)


# Task Constraints
def joint_angle_constraint(env, safety_vars):
  """Joints must stay within a certain range."""
  joint_pos = safety_vars['joint_pos']
  return (np.greater(joint_pos,
                     env.limits['joint_pos_constraint'][0]).any() and
          np.less(joint_pos, env.limits['joint_pos_constraint'][1])).any()


def joint_velocity_constraint(env, safety_vars):
  """Joint angle velocities must stay below a certain limit."""
  joint_vels = safety_vars['joint_vels']
  return np.less(np.max(joint_vels), env.limits['joint_velocity_constraint'])


def dangerous_fall_constraint(env, safety_vars):
  """Discourage dangerous falls (make sure torso stay forwards)."""
  return np.greater(safety_vars['torso_forwards'],
                    env.limits['dangerous_fall_constraint'])


def torso_upright_constraint(env, safety_vars):
  """Pelvis orientantion should remain upright."""
  z_up = safety_vars['z_up']
  return np.greater(z_up, env.limits['torso_upright_constraint'])


# Action rate of change constraint.
action_roc_constraint = realworld_env.action_roc_constraint


def realworld_stand(time_limit=_DEFAULT_TIME_LIMIT,
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
  """Returns the Stand task.

  Args:
    time_limit: Integer length of task
    random: random seed (unsure)
    log_output: String of path for pickle data logging, None disables logging
    environment_kwargs: additional kwargs for environment.
    safety_spec: dictionary that specifies the safety specifications.
    delay_spec: dictionary that specifies the delay specifications.
    noise_spec: dictionary that specifies the noise specifications.
    perturb_spec: dictionary that specifies the perturbations specifications.
    dimensionality_spec: dictionary that specifies extra observation features.
    multiobj_spec: dictionary that specifies complementary objectives.
    combined_challenge: string that can be 'easy', 'medium', or 'hard'.
      Specifying the combined challenge (can't be used with any other spec).
  """
  physics = walker.Physics.from_xml_string(*walker.get_model_and_assets())
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
  # Updating perturbation parameters if combined_challenge.
  if combined_challenge == 'easy':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.25, 'std': 0.002})
  elif combined_challenge == 'medium':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.4, 'std': 0.015})
  elif combined_challenge == 'hard':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.55, 'std': 0.04})

  task = RealWorldPlanarWalker(
      move_speed=0,
      random=random,
      safety_spec=safety_spec,
      delay_spec=delay_spec,
      noise_spec=noise_spec,
      perturb_spec=perturb_spec,
      dimensionality_spec=dimensionality_spec,
      multiobj_spec=multiobj_spec)

  environment_kwargs = environment_kwargs or {}
  if log_output:
    logger = loggers.PickleLogger(path=log_output)
  else:
    logger = None
  return wrappers.LoggingEnv(
      physics,
      task,
      logger=logger,
      time_limit=time_limit,
      control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


def realworld_walk(time_limit=_DEFAULT_TIME_LIMIT,
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
  """Returns the Stand task.

  Args:
    time_limit: Integer length of task
    random: random seed (unsure)
    log_output: String of path for pickle data logging, None disables logging
    environment_kwargs: additional kwargs for environment.
    safety_spec: dictionary that specifies the safety specifications.
    delay_spec: dictionary that specifies the delay specifications.
    noise_spec: dictionary that specifies the noise specifications.
    perturb_spec: dictionary that specifies the perturbations specifications.
    dimensionality_spec: dictionary that specifies extra observation features.
    multiobj_spec: dictionary that specifies complementary objectives.
    combined_challenge: string that can be 'easy', 'medium', or 'hard'.
      Specifying the combined challenge (can't be used with any other spec).
  """
  physics = walker.Physics.from_xml_string(*walker.get_model_and_assets())
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
  # Updating perturbation parameters if combined_challenge.
  if combined_challenge == 'easy':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.25, 'std': 0.002})
  elif combined_challenge == 'medium':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.4, 'std': 0.015})
  elif combined_challenge == 'hard':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.55, 'std': 0.04})

  task = RealWorldPlanarWalker(
      move_speed=_WALK_SPEED,
      random=random,
      safety_spec=safety_spec,
      delay_spec=delay_spec,
      noise_spec=noise_spec,
      perturb_spec=perturb_spec,
      dimensionality_spec=dimensionality_spec,
      multiobj_spec=multiobj_spec)

  environment_kwargs = environment_kwargs or {}
  if log_output:
    logger = loggers.PickleLogger(path=log_output)
  else:
    logger = None
  return wrappers.LoggingEnv(
      physics,
      task,
      logger=logger,
      time_limit=time_limit,
      control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


def realworld_run(time_limit=_DEFAULT_TIME_LIMIT,
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
  """Returns the Stand task.

  Args:
    time_limit: Integer length of task
    random: random seed (unsure)
    log_output: String of path for pickle data logging, None disables logging
    environment_kwargs: additional kwargs for environment.
    safety_spec: dictionary that specifies the safety specifications.
    delay_spec: dictionary that specifies the delay specifications.
    noise_spec: dictionary that specifies the noise specifications.
    perturb_spec: dictionary that specifies the perturbations specifications.
    dimensionality_spec: dictionary that specifies extra observation features.
    multiobj_spec: dictionary that specifies complementary objectives.
    combined_challenge: string that can be 'easy', 'medium', or 'hard'.
      Specifying the combined challenge (can't be used with any other spec).
  """
  physics = walker.Physics.from_xml_string(*walker.get_model_and_assets())
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
  # Updating perturbation parameters if combined_challenge.
  if combined_challenge == 'easy':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.25, 'std': 0.002})
  elif combined_challenge == 'medium':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.4, 'std': 0.015})
  elif combined_challenge == 'hard':
    perturb_spec.update(
        {'param': 'thigh_length', 'min': 0.225, 'max': 0.55, 'std': 0.04})

  task = RealWorldPlanarWalker(
      move_speed=_RUN_SPEED,
      random=random,
      safety_spec=safety_spec,
      delay_spec=delay_spec,
      noise_spec=noise_spec,
      perturb_spec=perturb_spec,
      dimensionality_spec=dimensionality_spec,
      multiobj_spec=multiobj_spec)

  environment_kwargs = environment_kwargs or {}
  if log_output:
    logger = loggers.PickleLogger(path=log_output)
  else:
    logger = None
  return wrappers.LoggingEnv(
      physics,
      task,
      logger=logger,
      time_limit=time_limit,
      control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)


class Physics(walker.Physics):
  """Inherits from walker.Physics."""


class RealWorldPlanarWalker(realworld_env.Base, walker.PlanarWalker):
  """A Walker task with real-world specifications.

  Subclasses dm_control.suite.walker.

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

  def __init__(self, safety_spec, delay_spec, noise_spec, perturb_spec,
               dimensionality_spec, multiobj_spec, **kwargs):
    """Initialize the RealWorldPlanarWalker task.

    Args:
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
        param - string indicating which parameter to perturb (currently
          supporting thigh_length, torso_length, joint_damping,
          contact_friction).
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
      **kwargs: extra parameters passed to parent class (walker.PlanarWalker)
    """
    # Initialize parent classes.
    realworld_env.Base.__init__(self)
    walker.PlanarWalker.__init__(self, **kwargs)

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
            ('dangerous_fall_constraint', dangerous_fall_constraint),
            ('torso_upright_constraint', torso_upright_constraint)
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
            # These constraints are taken from joint limits in the DM suite's
            # walker.xml
            'joint_pos_constraint':
                safety_coeff * np.array([[-30, -30, -160, -160, -75, -75],
                                         [110, 110, 15, 15, 60, 60]]) * np.pi /
                180.,  # rad
            'joint_velocity_constraint':
                safety_coeff * 65,  #  rad/s
            'dangerous_fall_constraint':
                safety_coeff * -0.5,  # projection
            'torso_upright_constraint':
                (1 - safety_coeff),  #  vector magnitude
            'action_roc_constraint': safety_coeff * 1.55,
        }
      self._constraints_obs = np.ones(len(self.constraints), dtype=bool)

  def safety_vars(self, physics):
    """Centralized retrieval of safety-related variables to simplify logging."""
    joint_pos = np.squeeze(
        np.array([physics.named.data.qpos[joint] for joint in _JOINT_NAMES]))
    safety_vars = collections.OrderedDict(
        joint_pos=joint_pos,
        joint_vels=np.abs(physics.named.data.qvel[3:]).copy(),
        torso_forwards=-1 * physics.named.data.xmat['torso', 'zx'].copy(),
        z_up=physics.torso_upright(),
        actions=physics.control(),)
    return safety_vars

  def _setup_perturb(self, perturb_spec):
    """Setup for the perturbations specification of the task."""
    self._perturb_enabled = perturb_spec.get('enable', False)
    self._perturb_period = perturb_spec.get('period', 1)

    if self._perturb_enabled:
      # Add perturbations specifications.
      self._perturb_param = perturb_spec.get('param', 'thigh_length')
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
      if self._perturb_param == 'thigh_length':
        self._perturb_cur = perturb_spec.get('start', 0.225)
        self._perturb_start = perturb_spec.get('start', 0.225)
        self._perturb_min = perturb_spec.get('min', 0.1)
        self._perturb_max = perturb_spec.get('max', 0.7)
        self._perturb_std = perturb_spec.get('std', 0.05)
      elif self._perturb_param == 'torso_length':
        self._perturb_cur = perturb_spec.get('start', 0.3)
        self._perturb_start = perturb_spec.get('start', 0.3)
        self._perturb_min = perturb_spec.get('min', 0.1)
        self._perturb_max = perturb_spec.get('max', 0.7)
        self._perturb_std = perturb_spec.get('std', 0.05)
      elif self._perturb_param == 'joint_damping':
        self._perturb_cur = perturb_spec.get('start', 0.1)
        self._perturb_start = perturb_spec.get('start', 0.1)
        self._perturb_min = perturb_spec.get('min', 0.1)
        self._perturb_max = perturb_spec.get('max', 10.)
        self._perturb_std = perturb_spec.get('std', 0.5)
      elif self._perturb_param == 'contact_friction':
        self._perturb_cur = perturb_spec.get('start', 0.7)
        self._perturb_start = perturb_spec.get('start', 0.7)
        self._perturb_min = perturb_spec.get('min', 0.01)
        self._perturb_max = perturb_spec.get('max', 2.)
        self._perturb_std = perturb_spec.get('std', 0.05)

  def update_physics(self):
    """Returns a new Physics object with perturbed parameter."""
    # Generate the new perturbed parameter.
    realworld_env.Base._generate_parameter(self)

    # Create new physics object with the perturb parameter.
    xml_string = common.read_model('walker.xml')
    mjcf = etree.fromstring(xml_string)

    if self._perturb_param == 'thigh_length':
      thighs = mjcf.findall('./worldbody/body/body/geom')
      for thigh in thighs:
        thigh.set('pos', '0 0 {}'.format(-self._perturb_cur))
        thigh.set('size', '0.05 {}'.format(self._perturb_cur))
      legs = mjcf.findall('./worldbody/body/body/body')
      for leg in legs:
        leg.set('pos', '0 0 {}'.format(-0.25 - 2 * self._perturb_cur))
    elif self._perturb_param == 'torso_length':
      torso = mjcf.find('./worldbody/body/geom')
      torso.set('size', '0.07 {}'.format(self._perturb_cur))
      thighs = mjcf.findall('./worldbody/body/body')
      for thigh in thighs:
        thigh.set('pos', '0 -.05 {}'.format(-self._perturb_cur))
    elif self._perturb_param == 'joint_damping':
      joint_damping = mjcf.find('./default/joint')
      joint_damping.set('damping', str(self._perturb_cur))
    elif self._perturb_param == 'contact_friction':
      geom_contact = mjcf.find('./default/geom')
      geom_contact.set('friction', '{} .1 .1'.format(self._perturb_cur))
      floor_contact = mjcf.find('./worldbody/geom')  # Floor geom.
      floor_contact.set('friction', '{} .1 .1'.format(self._perturb_cur))

    xml_string_modified = etree.tostring(mjcf, pretty_print=True)
    physics = Physics.from_xml_string(xml_string_modified, common.ASSETS)
    return physics

  def get_observation(self, physics):
    """Augments the observation based on the different specifications."""
    obs = walker.PlanarWalker.get_observation(self, physics)
    obs['height'] = np.array([obs['height']])
    obs = realworld_env.Base.get_observation(self, physics, obs)
    return obs

  def before_step(self, action, physics):
    """Updates the environment using the action and returns a `TimeStep`."""
    self._last_action = physics.control()
    action_min = self.action_spec(physics).minimum[:]
    action_max = self.action_spec(physics).maximum[:]
    action = realworld_env.Base.before_step(self, action, action_min,
                                            action_max)
    walker.PlanarWalker.before_step(self, action, physics)

  def after_step(self, physics):
    realworld_env.Base.after_step(self, physics)
    walker.PlanarWalker.after_step(self, physics)
    self._last_action = None
