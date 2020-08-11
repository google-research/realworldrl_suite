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

"""RealWorld RL env logging wrappers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy

from dm_control.rl import control
import dm_env
from dm_env import specs
from realworldrl_suite.utils import accumulators
import six


class LoggingEnv(control.Environment):
  """Subclass of control.Environment which adds logging."""

  def __init__(self,
               physics,
               task,
               logger=None,
               log_safety_vars=False,
               time_limit=float('inf'),
               control_timestep=None,
               n_sub_steps=None,
               log_every=100,
               flat_observation=False):
    """A subclass of `Environment` with logging hooks.

    Args:
      physics: Instance of `Physics`.
      task: Instance of `Task`.
      logger: Instance of 'realworldrl.utils.loggers.LoggerEnv', if specified
        will be used to log necessary data for realworld eval.
      log_safety_vars: If we should also log vars in self._task.safety_vars(),
        generally used for debugging or to find pertinent values for vars, will
        increase size of files on disk
      time_limit: Optional `int`, maximum time for each episode in seconds. By
        default this is set to infinite.
      control_timestep: Optional control time-step, in seconds.
      n_sub_steps: Optional number of physical time-steps in one control
        time-step, aka "action repeats". Can only be supplied if
        `control_timestep` is not specified.
      log_every: How many episodes between each log write.
      flat_observation: If True, observations will be flattened and concatenated
        into a single numpy array.

    Raises:
      ValueError: If both `n_sub_steps` and `control_timestep` are supplied.
    """
    super(LoggingEnv, self).__init__(
        physics,
        task,
        time_limit,
        control_timestep,
        n_sub_steps,
        flat_observation=False)
    self._flat_observation_ = flat_observation
    self._logger = logger
    self._buffer = []
    self._counter = 0
    self._log_every = log_every
    self._ep_counter = 0
    self._log_safety_vars = self._task.safety_enabled and log_safety_vars
    if self._logger:
      meta_dict = dict(task_name=type(self._task).__name__)
      if self._task.safety_enabled:
        meta_dict['safety_constraints'] = list(self._task.constraints.keys())
        if self._log_safety_vars:
          meta_dict['safety_vars'] = list(
              list(self._task.safety_vars(self._physics).keys()))
      self._logger.set_meta(meta_dict)

      self._stats_acc = accumulators.StatisticsAccumulator(
          acc_safety=self._task.safety_enabled,
          acc_safety_vars=self._log_safety_vars,
          acc_multiobj=self._task.multiobj_enabled)
    else:
      self._stats_acc = None

  def reset(self):
    """Starts a new episode and returns the first `TimeStep`."""
    if self._stats_acc:
      self._stats_acc.clear_buffer()
    if self._task.perturb_enabled:
      if self._counter % self._task.perturb_period == 0:
        self._physics = self._task.update_physics()
      self._counter += 1
    timestep = super(LoggingEnv, self).reset()
    self._track(timestep)
    if self._flat_observation_:
      timestep = dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=None,
          discount=None,
          observation=control.flatten_observation(
              timestep.observation)['observations'])
    return timestep

  def observation_spec(self):
    """Returns the observation specification for this environment.

    Infers the spec from the observation, unless the Task implements the
    `observation_spec` method.

    Returns:
      An dict mapping observation name to `ArraySpec` containing observation
      shape and dtype.
    """
    self._flat_observation = self._flat_observation_
    obs_spec = super(LoggingEnv, self).observation_spec()
    self._flat_observation = False
    if self._flat_observation_:
      return obs_spec['observations']
    return obs_spec

  def step(self, action):
    """Updates the environment using the action and returns a `TimeStep`."""
    do_track = not self._reset_next_step
    timestep = super(LoggingEnv, self).step(action)
    if do_track:
      self._track(timestep)
    if timestep.last():
      self._ep_counter += 1
      if self._ep_counter % self._log_every == 0:
        self.write_logs()
    # Only flatten observation if we're not forwarding one from a reset(),
    # as it will already be flattened.
    if self._flat_observation_ and not timestep.first():
      timestep = dm_env.TimeStep(
          step_type=timestep.step_type,
          reward=timestep.reward,
          discount=timestep.discount,
          observation=control.flatten_observation(
              timestep.observation)['observations'])
    return timestep

  def _track(self, timestep):
    if self._logger is None:
      return
    ts = copy.deepcopy(timestep)
    # Augment the timestep with unobserved variables for logging purposes.
    # Add safety-related observations.
    if self._task.safety_enabled and 'constraints' not in ts.observation:
      ts.observation['constraints'] = copy.copy(self._task.constraints_obs)
    if self._log_safety_vars:
      ts.observation['safety_vars'] = copy.deepcopy(
          self._task.safety_vars(self._physics))
    if self._task.multiobj_enabled and 'multiobj' not in ts.observation:
      ts.observation['multiobj'] = self._task.get_multiobj_obs(self._physics)
    self._stats_acc.push(ts)

  def get_logs(self):
    return self._logger.logs

  def write_logs(self):
    if self._logger is None:
      return
    self._logger.save(data=self._stats_acc.to_ndarray_dict())

  @property
  def stats_acc(self):
    return self._stats_acc

  @property
  def logs_path(self):
    if self._logger is None:
      return None
    return self._logger.logs_path


def _spec_from_observation(observation):
  result = collections.OrderedDict()
  for key, value in six.iteritems(observation):
    result[key] = specs.Array(value.shape, value.dtype, name=key)
  return result
