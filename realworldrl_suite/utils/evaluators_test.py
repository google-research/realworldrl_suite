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

"""Tests for evaluators."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import realworldrl_suite.environments as rwrl
from realworldrl_suite.utils import evaluators


class RandomAgent(object):

  def __init__(self, action_spec):
    self.action_spec = action_spec

  def action(self):
    return np.random.uniform(
        self.action_spec.minimum,
        self.action_spec.maximum,
        size=self.action_spec.shape)


class EvaluatorsTest(parameterized.TestCase):

  def _gen_stats(self, domain_name, task_name):
    temp_dir = self.create_tempdir()
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        log_output=os.path.join(temp_dir.full_path, 'test.pickle'),
        environment_kwargs=dict(log_safety_vars=True, flat_observation=True))
    random_policy = RandomAgent(env.action_spec()).action
    for _ in range(3):
      timestep = env.step(random_policy())
      while not timestep.last():
        timestep = env.step(random_policy())
    env.write_logs()
    return env.logs_path

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_loading(self, domain_name, task_name):
    temp_path = self._gen_stats(domain_name, task_name)
    data_in = np.load(temp_path, allow_pickle=True)
    evaluators.Evaluators(data_in)

  def test_safety_evaluator(self):
    # TODO(dulacarnold): Make this test general to all envs.
    temp_path = self._gen_stats(
        domain_name='cartpole', task_name='realworld_balance')
    data_in = np.load(temp_path, allow_pickle=True)
    ev = evaluators.Evaluators(data_in)
    self.assertLen(ev.get_safety_evaluator(), 3)

  def test_standard_evaluators(self):
    # TODO(dulacarnold): Make this test general to all envs.
    temp_path = self._gen_stats(
        domain_name='cartpole', task_name='realworld_balance')
    data_in = np.load(temp_path, allow_pickle=True)
    ev = evaluators.Evaluators(data_in)
    self.assertLen(ev.get_standard_evaluators(), 5)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def test_safety_plot(self, domain_name, task_name):
    temp_path = self._gen_stats(domain_name, task_name)
    data_in = np.load(temp_path, allow_pickle=True)
    ev = evaluators.Evaluators(data_in)
    ev.get_safety_plot()


if __name__ == '__main__':
  absltest.main()
