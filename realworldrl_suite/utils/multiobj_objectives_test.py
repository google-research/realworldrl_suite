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

"""Tests for multi-objective reward."""

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import realworldrl_suite.environments as rwrl
from realworldrl_suite.utils import multiobj_objectives


class MultiObjTest(parameterized.TestCase):

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testMultiObjNoSafety(self, domain_name, task_name):
    """Ensure multi-objective safety reward can be loaded."""
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': False},
        multiobj_spec={
            'enable': True,
            'objective': 'safety',
            'observed': True,
            'coeff': 0.5
        })
    with self.assertRaises(Exception):
      env.reset()
      env.step(0)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testMultiObjPassedObjective(self, domain_name, task_name):
    """Ensure objective class can be passed directly."""
    multiobj_class = lambda: multiobj_objectives.SafetyObjective()  # pylint: disable=unnecessary-lambda
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        multiobj_spec={
            'enable': True,
            'objective': multiobj_class,
            'observed': True,
            'coeff': 0.5
        })
    env.reset()
    env.step(0)

    multiobj_class = multiobj_objectives.SafetyObjective
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        multiobj_spec={
            'enable': True,
            'objective': multiobj_class,
            'observed': True,
            'coeff': 0.5
        })
    env.reset()
    env.step(0)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testMultiObjSafetyNoRewardObs(self, domain_name, task_name):
    """Ensure multi-objective safety reward can be loaded."""
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        multiobj_spec={
            'enable': True,
            'objective': 'safety',
            'reward': False,
            'observed': True,
            'coeff': 0.5
        })
    env.reset()
    env.step(0)
    ts = env.step(0)
    self.assertIn('multiobj', ts.observation)
    # Make sure we see a 1 in normalized violations
    env.task._constraints_obs = np.ones(
        env.task._constraints_obs.shape).astype(np.bool)
    obs = env.task.get_observation(env.physics)
    self.assertEqual(obs['multiobj'][1], 1)
    # And that there is no effect on rewards
    env.task._multiobj_coeff = 0
    r1 = env.task.get_reward(env.physics)
    env.task._multiobj_coeff = 1
    r2 = env.task.get_reward(env.physics)
    self.assertEqual(r1, r2)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testMultiObjSafetyRewardNoObs(self, domain_name, task_name):
    """Ensure multi-objective safety reward can be loaded."""
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        multiobj_spec={
            'enable': True,
            'objective': 'safety',
            'reward': True,
            'observed': False,
            'coeff': 0.5
        })
    env.reset()
    ts = env.step(0)
    self.assertNotIn('multiobj', ts.observation)
    # Make sure the method calls without error.
    env.task.get_multiobj_reward(env._physics, 0)
    env.task._constraints_obs = np.ones(
        env.task._constraints_obs.shape).astype(np.bool)

    # Make sure the mixing is working and global reward calls without error.
    env.task._multiobj_coeff = 1
    max_reward = env.task.get_reward(env.physics)
    env.task._multiobj_coeff = 0.5
    mid_reward = env.task.get_reward(env.physics)
    env.task._multiobj_coeff = 0.0
    min_reward = env.task.get_reward(env.physics)
    self.assertGreaterEqual(max_reward, min_reward)
    self.assertGreaterEqual(mid_reward, min_reward)

    env.task._multiobj_coeff = 0.5
    max_reward = env.task.get_reward(env.physics)
    self.assertEqual(
        env.task._multiobj_objective.merge_reward(env.task, env._physics, 0, 1),
        1)
    self.assertEqual(env.task.get_multiobj_reward(env._physics, 0), 0.5)
    self.assertEqual(env.task.get_multiobj_reward(env._physics, 0.5), 0.75)
    self.assertEqual(env.task.get_multiobj_reward(env._physics, 1), 1)


if __name__ == '__main__':
  absltest.main()
