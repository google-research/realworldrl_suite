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

"""Tests for accumulators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import numpy.testing as npt
import realworldrl_suite.environments as rwrl




class RandomAgent(object):

  def __init__(self, action_spec):
    self.action_spec = action_spec

  def action(self):
    return np.random.uniform(
        self.action_spec.minimum,
        self.action_spec.maximum,
        size=self.action_spec.shape)


class AccumulatorsTest(parameterized.TestCase):

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_logging(self, domain_name, task_name):
    temp_dir = self.create_tempdir()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        multiobj_spec={
            'enable': True,
            'objective': 'safety',
            'observed': False,
        },
        log_output=os.path.join(temp_dir.full_path, 'test.pickle'),
        environment_kwargs=dict(log_safety_vars=True))
    random_policy = RandomAgent(env.action_spec()).action
    n_steps = 0
    for _ in range(3):
      timestep = env.step(random_policy())
      constraints = (~timestep.observation['constraints']).astype('int')
      n_steps += 1
      while not timestep.last():
        timestep = env.step(random_policy())
        constraints += (~timestep.observation['constraints']).astype('int')
      npt.assert_equal(
          env.stats_acc.stat_buffers['safety_stats']['total_violations'][-1],
          constraints)
    env.write_logs()
    with open(env.logs_path, 'rb') as f:
      read_data = np.load(f, allow_pickle=True)
      data = read_data['data'].item()
      self.assertLen(data.keys(), 4)
      self.assertIn('safety_vars_stats', data)
      self.assertIn('total_violations', data['safety_stats'])
      self.assertIn('per_step_violations', data['safety_stats'])
      self.assertIn('episode_totals', data['multiobj_stats'])
      self.assertIn('episode_totals', data['return_stats'])
      self.assertLen(data['safety_stats']['total_violations'], n_steps)
      self.assertLen(data['safety_vars_stats'], n_steps)
      self.assertLen(data['multiobj_stats']['episode_totals'], n_steps)
      self.assertLen(data['return_stats']['episode_totals'], n_steps)


if __name__ == '__main__':
  absltest.main()
