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

"""Tests for realworldrl.utils.wrappers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import realworldrl_suite.environments as rwrl


class RandomAgent(object):

  def __init__(self, action_spec):
    self.action_spec = action_spec

  def action(self):
    return np.random.uniform(
        self.action_spec.minimum,
        self.action_spec.maximum,
        size=self.action_spec.shape)


class WrappersTest(parameterized.TestCase):

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_init(self, domain_name, task_name):
    temp_file = self.create_tempfile()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        log_output=temp_file.full_path,
        environment_kwargs=dict(log_safety_vars=True))
    env.write_logs()

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_no_logger(self, domain_name, task_name):
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        log_output=None,
        environment_kwargs=dict(log_safety_vars=True))
    env.write_logs()

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_disabled_safety_obs(self, domain_name, task_name):
    temp_file = self.create_tempfile()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True, 'observations': False},
        log_output=temp_file.full_path,
        environment_kwargs=dict(log_safety_vars=True))
    env.reset()
    timestep = env.step(0)
    self.assertNotIn('constraints', timestep.observation.keys())
    self.assertIn('constraints', env._stats_acc._buffer[-1].observation)
    env.write_logs()

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_custom_safety_constraint(self, domain_name, task_name):
    const_constraint = (domain_name == rwrl.ALL_TASKS[0][1])
    def custom_constraint(unused_env_obj, unused_safety_vars):
      return const_constraint
    constraints = {'custom_constraint': custom_constraint}
    temp_file = self.create_tempfile()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True, 'observations': True,
                     'constraints': constraints},
        log_output=temp_file.full_path,
        environment_kwargs=dict(log_safety_vars=True))
    env.reset()
    timestep = env.step(0)
    self.assertIn('constraints', timestep.observation.keys())
    self.assertEqual(timestep.observation['constraints'],
                     np.array([const_constraint]))
    env.write_logs()

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_action_roc_constraint(self, domain_name, task_name):
    is_within_constraint = (domain_name == rwrl.ALL_TASKS[0][1])
    constraints = {
        'action_roc_constraint': rwrl.DOMAINS[domain_name].action_roc_constraint
    }
    temp_file = self.create_tempfile()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True, 'observations': True,
                     'constraints': constraints},
        log_output=temp_file.full_path,
        environment_kwargs=dict(log_safety_vars=True))
    env.reset()
    _ = env.step(-100)
    timestep_2 = env.step(-100 if is_within_constraint else 100)
    self.assertIn('constraints', timestep_2.observation.keys())
    self.assertEqual(timestep_2.observation['constraints'],
                     np.array([is_within_constraint]))
    env.write_logs()

  # @parameterized.parameters(*(5,))
  def test_log_every_n(self, every_n=5):
    domain_name = 'cartpole'
    task_name = 'realworld_balance'
    temp_dir = self.create_tempdir()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True, 'observations': False},
        log_output=os.path.join(temp_dir.full_path, 'test.pickle'),
        environment_kwargs=dict(log_safety_vars=True, log_every=every_n))
    env.reset()
    n = 0
    while True:
      timestep = env.step(0)
      if timestep.last():
        n += 1
        if n % every_n == 0:
          self.assertTrue(os.path.exists(env.logs_path))
          os.remove(env.logs_path)
          self.assertFalse(os.path.exists(env.logs_path))
        if n > 20:
          break

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_flat_obs(self, domain_name, task_name):
    temp_file = self.create_tempfile()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        log_output=temp_file.full_path,
        environment_kwargs=dict(log_safety_vars=True, flat_observation=True))
    self.assertLen(env.observation_spec().shape, 1)
    if domain_name == 'cartpole' and task_name in ['swingup', 'balance']:
      self.assertEqual(env.observation_spec().shape[0], 8)
    timestep = env.step(0)
    self.assertLen(timestep.observation.shape, 1)
    if domain_name == 'cartpole' and task_name in ['swingup', 'balance']:
      self.assertEqual(timestep.observation.shape[0], 8)
    env.write_logs()


if __name__ == '__main__':
  absltest.main()
