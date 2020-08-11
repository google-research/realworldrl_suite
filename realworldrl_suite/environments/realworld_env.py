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

"""Base class for the real-world environments."""

import collections
import copy
import inspect

import numpy as np
from realworldrl_suite.utils import multiobj_objectives

PERTURB_SCHEDULERS = [
    'constant', 'random_walk', 'drift_pos', 'drift_neg', 'cyclic_pos',
    'cyclic_neg', 'uniform', 'saw_wave'
]


def action_roc_constraint(env, safety_vars):
  """Limits the rate of change of the input action.

  This is imported directly by each task environment and is inactive by default.
  To use, set it as one of the entries to safety_spec['constraints'].
  Args:
    env: The RWRL Env.
    safety_vars: The safety_vars from calling env.safety_vars(physics).
  Returns:
    A boolean, where True represents the constraint was not violated.
  """
  if env._last_action is None:  # pylint:disable=protected-access
    # May happen at the very first step.
    return True
  return np.all(np.less(
      np.abs(env._last_action - safety_vars['actions']),  # pylint:disable=protected-access
      env.limits['action_roc_constraint']))


# Utility functions.
def delay_buffer_spec(delay_spec, delay_name):
  """Returns the length (time steps) of a specified delay."""
  b_len = delay_spec.get(delay_name, 0)
  return b_len + 1, None


def delayed_buffer_item(buffer_item, buffer_item_len, item):
  """Maintains delays using lists."""
  item_copy = copy.copy(item)
  if buffer_item is None:
    buffer_item = buffer_item_len * [item_copy]
  else:
    buffer_item.append(item_copy)
  item_cur = copy.copy(buffer_item.pop(0))

  return buffer_item, item_cur


def noise_value(noise_spec, noise_name, default_value=0.0, int_val=False):
  """Returns the value of a specified noise."""
  val = noise_spec.get(noise_name, default_value)
  if int_val:
    val = int(val)
  return val


def get_combined_challenge(combined_challenge, delay_spec, noise_spec,
                           perturb_spec, dimensionality_spec):
  """Returns the specs that define the combined challenge (if applicable)."""
  # Verify combined_challenge value is legal.
  if (combined_challenge is not None) and (
      combined_challenge not in ['easy', 'medium', 'hard']):
    raise ValueError('combined_challenge must be easy, medium, or hard.')
  # Verify no other spec is defined if combined_challenge is specified.
  if combined_challenge is not None:
    if (bool(delay_spec)) or (bool(noise_spec)) or (bool(perturb_spec)) or (
        bool(dimensionality_spec)):
      raise ValueError('combined_challenge is specified.'
                       'delay_spec, noise_spec, perturb_spec, or '
                       'dimensionality_spec may not be specified.')
  # Define the specs according to the combined challenge.
  if combined_challenge == 'easy':
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
  elif combined_challenge == 'medium':
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
  elif combined_challenge == 'hard':
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
  # Return the updated specs.
  return delay_spec, noise_spec, perturb_spec, dimensionality_spec


class Base(object):
  """Base class for the different real-world environments.

  This class is used for common code sharing between the different environments.
  """

  def __init__(self):
    """Initalizes the base class for realworld environments.

    The following attributes must be set explicitly by any subclass.
    """
    # Safety related.
    # If subclass sets self._safety_enabled to True, also must set
    # self.constraints and self._constraints_obs. self.constraints takes two
    # arguments, the 'self' object and the safety_vars.
    self._safety_enabled = False
    self.constraints = None
    self._constraints_obs = None

    # Delay related.
    self._delay_enabled = False
    self._buffer_observations_len = None
    self._buffer_observations = None
    self._buffer_actions_len = None
    self._buffer_actions = None
    self._buffer_rewards_len = None
    self._buffer_rewards = None

    # Noise Gaussian related.
    self._noise_guassian_enabled = False
    self._noise_gaussian_observations = None
    self._noise_gaussian_actions = None

    # Noise dropped related.
    self._noise_dropped_enabled = False
    self._noise_dropped_obs_prob = None
    self._noise_dropped_obs_steps = None
    self._noise_dropped_obs_dict = None
    self._noise_dropped_action_prob = None
    self._noise_dropped_action_steps = None
    self._noise_dropped_action_arr = None

    # Noise stuck related.
    self._noise_stuck_enabled = False
    self._noise_stuck_obs_prob = None
    self._noise_stuck_obs_steps = None
    self._noise_stuck_obs_dict = None
    self._noise_stuck_obs = None
    self._noise_stuck_action_prob = None
    self._noise_stuck_action_steps = None
    self._noise_stuck_action_arr = None
    self._noise_stuck_action = None

    # Noise repetition related.
    self._noise_repetition_enabled = None
    self._noise_repetition_actions_prob = None
    self._noise_repetition_actions_steps = None
    self._noise_repetition_action = None
    self._noise_repetition_action_counter = None

    # Perturbation related.
    # If subclass sets self._perturb_enabled to True, also must set
    # self._perturb_param, self._perturb_scheduler, self._perturb_cur,
    # self._perturb_start, self._perturb_min, self._perturb_max,
    # self._perturb_std, and self._perturb_period
    self._perturb_enabled = False
    self._perturb_period = None
    self._perturb_param = None
    self._perturb_scheduler = None
    self._perturb_saw_wave_sign = 1.  # initial direction - for saw_wave.
    self._perturb_cur = None
    self._perturb_start = None
    self._perturb_min = None
    self._perturb_max = None
    self._perturb_std = None

    # State and action dimensions related.
    self._dimensionality_enabled = False
    self._num_random_state_observations = 0

    # Multi-objective related.
    self._multiobj_enabled = False
    self._multiobj_objective = None
    self._multiobj_reward = False
    self._multiobj_coeff = 0
    self._multiobj_observed = False

    # Constraint related.
    self._last_action = None

  def _setup_delay(self, delay_spec):
    """Setup for the delay specifications of the task."""
    self._delay_enabled = delay_spec.get('enable', False)

    if self._delay_enabled:
      # Add delay specifications.
      (self._buffer_actions_len,
       self._buffer_actions) = delay_buffer_spec(delay_spec, 'actions')

      (self._buffer_observations_len,
       self._buffer_observations) = delay_buffer_spec(delay_spec,
                                                      'observations')

      (self._buffer_rewards_len,
       self._buffer_rewards) = delay_buffer_spec(delay_spec, 'rewards')

  def _setup_noise(self, noise_spec):
    """Setup for the noise specifications of the task."""
    # White Gaussian noise.
    self._noise_guassian_enabled = noise_spec.get('gaussian',
                                                  {}).get('enable', False)
    if self._noise_guassian_enabled:
      self._noise_gaussian_observations = noise_value(noise_spec['gaussian'],
                                                      'observations')
      self._noise_gaussian_actions = noise_value(noise_spec['gaussian'],
                                                 'actions')

    # Dropped noise.
    self._noise_dropped_enabled = noise_spec.get('dropped',
                                                 {}).get('enable', False)
    if self._noise_dropped_enabled:
      self._noise_dropped_obs_prob = noise_value(noise_spec['dropped'],
                                                 'observations_prob')
      self._noise_dropped_obs_steps = noise_value(
          noise_spec['dropped'],
          'observations_steps',
          default_value=1,
          int_val=True)
      self._noise_dropped_action_prob = noise_value(noise_spec['dropped'],
                                                    'actions_prob')
      self._noise_dropped_action_steps = noise_value(
          noise_spec['dropped'], 'actions_steps', default_value=1, int_val=True)

    # Stuck noise.
    self._noise_stuck_enabled = noise_spec.get('stuck', {}).get('enable', False)
    if self._noise_stuck_enabled:
      self._noise_stuck_obs_prob = noise_value(noise_spec['stuck'],
                                               'observations_prob')
      self._noise_stuck_obs_steps = noise_value(
          noise_spec['stuck'],
          'observations_steps',
          default_value=1,
          int_val=True)
      self._noise_stuck_action_prob = noise_value(noise_spec['stuck'],
                                                  'actions_prob')
      self._noise_stuck_action_steps = noise_value(
          noise_spec['stuck'], 'actions_steps', default_value=1, int_val=True)

    # Repetition noise.
    self._noise_repetition_enabled = noise_spec.get('repetition',
                                                    {}).get('enable', False)
    if self._noise_repetition_enabled:
      self._noise_repetition_actions_prob = noise_value(
          noise_spec['repetition'], 'actions_prob')
      self._noise_repetition_actions_steps = noise_value(
          noise_spec['repetition'],
          'actions_steps',
          default_value=1,
          int_val=True)
      self._noise_repetition_action_counter = 0

  def _setup_dimensionality(self, dimensionality_spec):
    """Setup for the noise specifications of the task."""
    # Dummy variables of white Gaussian noise.
    self._dimensionality_enabled = dimensionality_spec.get('enable', False)

    if self._dimensionality_enabled:
      self._num_random_state_observations = dimensionality_spec.get(
          'num_random_state_observations', 0)

  def _setup_multiobj(self, multiobj_spec):
    """Setup for the multi-objective reward task."""
    self._multiobj_enabled = multiobj_spec.get('enable', False)

    if self._multiobj_enabled:
      self._multiobj_reward = multiobj_spec.get('reward', False)
      self._multiobj_coeff = multiobj_spec.get('coeff', 0.0)
      self._multiobj_observed = multiobj_spec.get('observed', False)

      # Load either from internal library or accept passing in class.
      multiobj_objective = multiobj_spec['objective']

      if isinstance(multiobj_objective, str):
        self._multiobj_objective = multiobj_objectives.OBJECTIVES[
            multiobj_objective]()
      elif inspect.isclass(multiobj_objective) or callable(multiobj_objective):
        self._multiobj_objective = multiobj_objective()

  def _generate_parameter(self):
    """Generates a new value for the physics perturbed parameter."""
    delta = np.random.normal(scale=self._perturb_std)

    if self._perturb_scheduler == 'constant':
      pass
    elif self._perturb_scheduler == 'random_walk':
      self._perturb_cur += delta
    elif self._perturb_scheduler == 'drift_pos':
      self._perturb_cur += abs(delta)
    elif self._perturb_scheduler == 'drift_neg':
      self._perturb_cur -= abs(delta)
    elif self._perturb_scheduler == 'cyclic_pos':
      self._perturb_cur += abs(delta)
      if self._perturb_cur >= self._perturb_max:
        self._perturb_cur = self._perturb_start
    elif self._perturb_scheduler == 'cyclic_neg':
      self._perturb_cur -= abs(delta)
      if self._perturb_cur <= self._perturb_min:
        self._perturb_cur = self._perturb_start
    elif self._perturb_scheduler == 'uniform':
      self._perturb_cur = np.random.uniform(
          low=self._perturb_min, high=self._perturb_max)
    elif self._perturb_scheduler == 'saw_wave':
      self._perturb_cur = self._perturb_saw_wave_sign * abs(delta)
      if ((self._perturb_cur >= self._perturb_max) or
          (self._perturb_cur <= self._perturb_min)):
        self._perturb_saw_wave_sign *= -1.

    # Clip the value to be in the defined support
    self._perturb_cur = np.clip(self._perturb_cur, self._perturb_min,
                                self._perturb_max)

  def get_observation(self, physics, obs=None):
    """Augments the observation based on the different specifications."""

    # This will get the task-specific observation.
    if not obs:
      obs = super(Base, self).get_observation(physics)  # pytype: disable=attribute-error

    if self._noise_guassian_enabled:
      # Add white Gaussian noise to observations.
      for k, v in obs.items():
        obs[k] = np.random.normal(v, self._noise_gaussian_observations)
        if not isinstance(v, np.ndarray):
          obs[k] = np.float64(obs[k]).astype(v.dtype)

    if self._noise_dropped_enabled:
      # Drop observation values with some probability.
      if not self._noise_dropped_obs_dict:
        # First observation - need to initialize dictionary.
        self._noise_dropped_obs_dict = collections.OrderedDict([
            (k, np.zeros(v.shape)) for k, v in obs.items()
        ])
      for k, v in self._noise_dropped_obs_dict.items():
        # Updating identities and length of dropped values.
        identities = np.random.binomial(
            n=1, p=self._noise_dropped_obs_prob, size=v.shape)
        self._noise_dropped_obs_dict[k][
            (v == 0) & (identities == 1)] = self._noise_dropped_obs_steps
        # Dropping values.
        if isinstance(obs[k], np.ndarray):
          obs[k][self._noise_dropped_obs_dict[k] > 0] = 0.
        else:
          obs[k] = np.float64(0.).astype(
              obs[k].dtype) if self._noise_dropped_obs_dict[k] > 0 else obs[k]
        update_indices = self._noise_dropped_obs_dict[k] > 0
        self._noise_dropped_obs_dict[k][update_indices] -= 1.

    if self._noise_stuck_enabled:
      # Stuck observation values with some probability.
      if not self._noise_stuck_obs_dict:
        # First observation - need to initialize dictionary and previous obs.
        self._noise_stuck_obs_dict = collections.OrderedDict([
            (k, np.zeros(v.shape)) for k, v in obs.items()
        ])
        self._noise_stuck_obs = copy.deepcopy(obs)
      for k, v in self._noise_stuck_obs_dict.items():
        # Updating identities and length of stuck values.
        identities = np.random.binomial(
            n=1, p=self._noise_stuck_obs_prob, size=v.shape)
        self._noise_stuck_obs_dict[k][
            (v == 0) & (identities == 1)] = self._noise_stuck_obs_steps
        # Stick values.
        if isinstance(obs[k], np.ndarray):
          stuck_indices = self._noise_stuck_obs_dict[k] > 0
          obs[k][stuck_indices] = self._noise_stuck_obs[k][stuck_indices]
        else:
          obs[k] = (
              self._noise_stuck_obs[k]
              if self._noise_stuck_obs_dict[k] > 0 else obs[k])
        update_indices = self._noise_stuck_obs_dict[k] > 0
        self._noise_stuck_obs_dict[k][update_indices] -= 1.
      # Storing observation as previous observation for next step.
      self._noise_stuck_obs = copy.deepcopy(obs)

    if self._safety_enabled:
      if self._safety_observed:
        obs['constraints'] = self._constraints_obs

    if self._delay_enabled and self._buffer_observations_len > 1:
      # Delay the observations.
      self._buffer_observations, obs = delayed_buffer_item(
          self._buffer_observations, self._buffer_observations_len, obs)

    if self._dimensionality_enabled and self._num_random_state_observations > 0:
      for i in range(self._num_random_state_observations):
        obs['dummy-{}'.format(i)] = np.array(np.random.normal())

    if self._multiobj_enabled and self._multiobj_observed:
      obs['multiobj'] = self.get_multiobj_obs(physics)

    return obs

  def get_reward(self, physics):
    # This will call the 2nd element of the mixin's `get_reward` method.
    # e.g. for a mixin with cartpole.Balance, this effectively calls
    # cartpole.Balance.get_reward
    reward = super(Base, self).get_reward(physics)  # pytype: disable=attribute-error
    reward = self.get_multiobj_reward(physics, reward)
    reward = self.delay_reward(reward)
    return reward

  def delay_reward(self, reward):
    """Augments the reward based on the different specifications."""
    if self._delay_enabled and self._buffer_rewards_len > 1:
      # Delay the reward.
      self._buffer_rewards, reward = delayed_buffer_item(
          self._buffer_rewards, self._buffer_rewards_len, reward)

    return reward

  def get_multiobj_obs(self, physics):
    base_reward = super(Base, self).get_reward(physics)  # pytype: disable=attribute-error
    objectives = self._multiobj_objective.get_objectives(self)
    return np.append(base_reward, objectives)

  def get_multiobj_reward(self, physics, reward):
    """Adds a multi-objective reward to the current reward."""
    if self._multiobj_enabled and self._multiobj_reward:
      return self._multiobj_objective.merge_reward(self, physics, reward,
                                                   self._multiobj_coeff)
    else:
      return reward

  def before_step(self, action, action_min, action_max):
    """Returns an actions according to the different specifications."""
    if self._delay_enabled and self._buffer_actions_len > 1:
      # Delay the actions.
      self._buffer_actions, action = delayed_buffer_item(
          self._buffer_actions, self._buffer_actions_len, action)

    if self._noise_guassian_enabled:
      # Add white Gaussian noise to actions.
      action = np.random.normal(action, self._noise_gaussian_actions)
      action = np.clip(action, action_min, action_max)

    if self._noise_dropped_enabled:
      # Drop action values with some probability.
      if self._noise_dropped_action_arr is None:
        # First action - need to initialize array.
        self._noise_dropped_action_arr = np.zeros(action.shape)
      # Updating identities and length of dropped values.
      identities = np.random.binomial(
          n=1, p=self._noise_dropped_action_prob, size=action.shape)
      dropped_indices = ((self._noise_dropped_action_arr == 0) &
                         (identities == 1))
      self._noise_dropped_action_arr[dropped_indices] = (
          self._noise_dropped_action_steps)
      # Dropping values.
      action[self._noise_dropped_action_arr > 0] = 0.
      update_indices = self._noise_dropped_action_arr > 0
      self._noise_dropped_action_arr[update_indices] -= 1.

    if self._noise_stuck_enabled:
      # Stuck action values with some probability.
      if self._noise_stuck_action_arr is None:
        # First action - need to initialize array.
        self._noise_stuck_action_arr = np.zeros(action.shape)
        self._noise_stuck_action = copy.deepcopy(action)
      # Updating identities and length of stuck values.
      identities = np.random.binomial(
          n=1, p=self._noise_stuck_action_prob, size=action.shape)
      stuck_indices = ((self._noise_stuck_action_arr == 0) & (identities == 1))
      self._noise_stuck_action_arr[stuck_indices] = (
          self._noise_stuck_action_steps)
      # Stick values.
      if isinstance(action, np.ndarray):
        stuck_indices = self._noise_stuck_action_arr > 0
        action[stuck_indices] = self._noise_stuck_action[stuck_indices]
      else:
        action = (
            self._noise_stuck_action
            if self._noise_stuck_action_arr > 0 else action)
      update_indices = self._noise_stuck_action_arr > 0
      self._noise_stuck_action_arr[update_indices] -= 1.
      # Storing action as previous action for next step.
      self._noise_stuck_action = copy.deepcopy(action)

    if self._noise_repetition_enabled:
      # Repeat previous actions if relevant.
      if self._noise_repetition_action is None:
        # First action - need to store reference.
        self._noise_repetition_action = copy.deepcopy(action)
      if self._noise_repetition_action_counter == 0:
        # Finished previous repetition.
        if np.random.uniform() < self._noise_repetition_actions_prob:
          # Action is to be repeated.
          self._noise_repetition_action_counter = (
              self._noise_repetition_actions_steps)
          # Setting the action to be the previous one.
          action = copy.deepcopy(action)
          # Decreasing the repetition counter by one.
          self._noise_repetition_action_counter -= 1
      else:
        # Still repeating previous action.
        action = copy.deepcopy(self._noise_repetition_action)
        self._noise_repetition_action_counter -= 1
      # Storing the action to serve as the next step's reference action.
      self._noise_repetition_action = copy.deepcopy(action)

    return action

  def safety_vars(self, physics):
    raise NotImplementedError

  def _populate_constraints_obs(self, physics):
    """Copies over the safety vars and populates the contraints observation."""
    safety_vars = self.safety_vars(physics)
    for idx, constraint in enumerate(self.constraints):
      self._constraints_obs[idx] = self.constraints[constraint](self,
                                                                safety_vars)

  def after_step(self, physics):
    # Populate safety observations here so it can be used by a multi-objective
    # reward function, which will be called before get_observation.
    if self._safety_enabled:
      self._populate_constraints_obs(physics)

  @property
  def constraints_obs(self):
    # The cached constraint observation
    return self._constraints_obs

  @property
  def safety_enabled(self):
    return self._safety_enabled

  @property
  def delay_enabled(self):
    return self._delay_enabled

  @property
  def perturb_enabled(self):
    return self._perturb_enabled

  @property
  def perturb_period(self):
    return self._perturb_period

  @property
  def multiobj_enabled(self):
    return self._multiobj_enabled
