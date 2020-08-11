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

"""Tests for real-world environments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import operator

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import realworldrl_suite.environments as rwrl
from realworldrl_suite.environments import realworld_env

NUM_DUMMY = 5


class EnvTest(parameterized.TestCase):

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testLoadEnv(self, domain_name, task_name):
    """Ensure it is possible to load the environment."""
    env = rwrl.load(domain_name=domain_name, task_name=task_name)
    env.reset()
    self.assertIsNotNone(env)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testSafetyConstraintsPresent(self, domain_name, task_name):
    """Ensure observations contain 'constraints' when safety is specified."""
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True})
    env.reset()
    step = env.step(0)
    self.assertIn('constraints', step.observation.keys())

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testSafetyCoeff(self, domain_name, task_name):
    """Ensure observations contain 'constraints' when safety is specified."""
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True, 'safety_coeff': 0.1})
    env.reset()
    step = env.step(0)
    self.assertIn('constraints', step.observation.keys())
    for c in [2, -1]:
      with self.assertRaises(ValueError):
        env = rwrl.load(
            domain_name=domain_name,
            task_name=task_name,
            safety_spec={'enable': True, 'safety_coeff': c})

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testSafetyObservationsDisabled(self, domain_name, task_name):
    """Ensure safety observations can be disabled."""
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={
            'enable': True,
            'observations': False
        })
    env.reset()
    step = env.step(0)
    self.assertNotIn('constraints', step.observation.keys())

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testDelayActionsNoDelay(self, domain_name, task_name):
    """Ensure there is no action delay if not specified."""
    env = rwrl.load(domain_name=domain_name, task_name=task_name)
    env.reset()
    action_spec = env.action_spec()

    # Send zero action and make sure it is immediately executed.
    zero_action = np.zeros(shape=action_spec.shape, dtype=action_spec.dtype)
    env.step(copy.copy(zero_action))
    np.testing.assert_array_equal(env.physics.control(), zero_action)

    # Send one action and make sure it is immediately executed.
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)
    if hasattr(action_spec, 'minimum'):
      one_action = np.minimum(action_spec.maximum, one_action)
    env.step(copy.copy(one_action))
    np.testing.assert_array_equal(env.physics.control(), one_action)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testDelayActionsDelay(self, domain_name, task_name):
    """Ensure there is action delay as specified."""
    actions_delay = np.random.randint(low=1, high=10)
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        delay_spec={
            'enable': True,
            'actions': actions_delay
        })
    env.reset()
    action_spec = env.action_spec()

    zero_action = np.zeros(shape=action_spec.shape, dtype=action_spec.dtype)
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)
    if hasattr(action_spec, 'minimum'):
      one_action = np.minimum(action_spec.maximum, one_action)
    # Perfrom first action that fills up the buffer.
    env.step(copy.copy(zero_action))

    # Send one action and make sure zero action is still executed.
    for _ in range(actions_delay):
      env.step(copy.copy(one_action))
      np.testing.assert_array_equal(env.physics.control(), zero_action)

    # Make sure we finally perform the delayed one action.
    env.step(copy.copy(zero_action))
    np.testing.assert_array_equal(env.physics.control(), one_action)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testDelayObservationsNoDelay(self, domain_name, task_name):
    """Ensure there is no observation delay if not specified."""
    env = rwrl.load(domain_name=domain_name, task_name=task_name)
    env.reset()
    action_spec = env.action_spec()
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)
    if hasattr(action_spec, 'minimum'):
      one_action = np.minimum(action_spec.maximum, one_action)
    obs1 = env._task.get_observation(env._physics)

    env.step(copy.copy(one_action))
    obs2 = env._task.get_observation(env._physics)

    # Make sure subsequent observations are different.
    array_equality = []
    for key in obs1:
      array_equality.append((obs1[key] == obs2[key]).all())
    self.assertIn(False, array_equality)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testDelayObservationsDelay(self, domain_name, task_name):
    """Ensure there is observation delay as specified."""
    observations_delay = np.random.randint(low=1, high=10)
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        delay_spec={
            'enable': True,
            'observations': observations_delay
        })
    obs1 = env.reset()[3]
    action_spec = env.action_spec()
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)

    # Make sure subsequent observations are the same (clearing the buffer).
    for _ in range(observations_delay):
      obs2 = env.step(copy.copy(one_action))[3]
    for key in obs1:
      np.testing.assert_array_equal(obs1[key], obs2[key])

    # Make sure we finally observe a different observation.
    obs2 = env.step(copy.copy(one_action))[3]
    array_equality = []
    for key in obs1:
      array_equality.append((obs1[key] == obs2[key]).all())
    self.assertIn(False, array_equality)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseGaussianActions(self, domain_name, task_name):
    """Ensure there is an additive Gaussian noise to the action."""
    noise = 0.5
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={'gaussian': {
            'enable': True,
            'actions': noise
        }})
    env.reset()
    action_spec = env.action_spec()

    zero_action = np.zeros(shape=action_spec.shape, dtype=action_spec.dtype)

    # Perform zero action.
    env.step(copy.copy(zero_action))

    # Verify that a non-zero action was actually performed.
    np.testing.assert_array_compare(operator.__ne__, env.physics.control(),
                                    zero_action)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testAddedDummyObservations(self, domain_name, task_name):
    """Ensure there is an additive Gaussian noise to the observation."""
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        dimensionality_spec={
            'enable': True,
            'num_random_state_observations': 5,
        })
    env.reset()

    # Get observation from realworld task.
    obs = env._task.get_observation(env._physics)
    for i in range(5):
      self.assertIn('dummy-{}'.format(i), obs.keys())
    for i in range(6, 10):
      self.assertNotIn('dummy-{}'.format(i), obs.keys())

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testAddedDummyObservationsFlattened(self, domain_name, task_name):
    """Ensure there is an additive Gaussian noise to the observation."""
    base_env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        safety_spec={'enable': True},
        environment_kwargs=dict(flat_observation=True))
    base_env.reset()
    mod_env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        dimensionality_spec={
            'enable': True,
            'num_random_state_observations': NUM_DUMMY,
        },
        safety_spec={'enable': True},
        environment_kwargs=dict(flat_observation=True))
    mod_env.reset()

    # Get observation from realworld task.
    base_obs = base_env.step(0)
    mod_obs = mod_env.step(0)
    self.assertEqual(mod_obs.observation.shape[0],
                     base_obs.observation.shape[0] + NUM_DUMMY)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseGaussianObservationsFlattening(self, domain_name, task_name):
    """Ensure there is an additive Gaussian noise to the observation."""
    noise = 0.5
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={'gaussian': {
            'enable': True,
            'observations': noise
        }},
        environment_kwargs={'flat_observation': True})
    env.reset()
    env.step(0)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseGaussianObservations(self, domain_name, task_name):
    """Ensure there is an additive Gaussian noise to the observation."""
    noise = 0.5
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={'gaussian': {
            'enable': True,
            'observations': noise
        }})
    env.reset()

    # Get observation from realworld cartpole.
    obs1 = env._task.get_observation(env._physics)

    # Get observation from underlying cartpole.
    obs2 = collections.OrderedDict()
    if domain_name == 'cartpole':
      obs2['position'] = env.physics.bounded_position()
      obs2['velocity'] = env.physics.velocity()
    elif domain_name == 'humanoid':
      obs2['joint_angles'] = env.physics.joint_angles()
      obs2['head_height'] = env.physics.head_height()
      obs2['extremities'] = env.physics.extremities()
      obs2['torso_vertical'] = env.physics.torso_vertical_orientation()
      obs2['com_velocity'] = env.physics.center_of_mass_velocity()
      obs2['velocity'] = env.physics.velocity()
    elif domain_name == 'manipulator':
      arm_joints = [
          'arm_root', 'arm_shoulder', 'arm_elbow', 'arm_wrist', 'finger',
          'fingertip', 'thumb', 'thumbtip'
      ]
      obs2['arm_pos'] = env.physics.bounded_joint_pos(arm_joints)
      obs2['arm_vel'] = env.physics.joint_vel(arm_joints)
      obs2['touch'] = env.physics.touch()
      obs2['hand_pos'] = env.physics.body_2d_pose('hand')
      obs2['object_pos'] = env.physics.body_2d_pose(env._task._object)
      obs2['object_vel'] = env.physics.joint_vel(env._task._object_joints)
      obs2['target_pos'] = env.physics.body_2d_pose(env._task._target)
    elif domain_name == 'quadruped':
      obs2['egocentric_state'] = env.physics.egocentric_state()
      obs2['torso_velocity'] = env.physics.torso_velocity()
      obs2['torso_upright'] = env.physics.torso_upright()
      obs2['imu'] = env.physics.imu()
      obs2['force_torque'] = env.physics.force_torque()
    elif domain_name == 'walker':
      obs2['orientations'] = env.physics.orientations()
      obs2['height'] = env.physics.torso_height()
      obs2['velocity'] = env.physics.velocity()
    else:
      raise ValueError('Unknown environment name: %s' % domain_name)

    # Verify that the observations are different (noise added).
    for key in obs1:
      np.testing.assert_array_compare(operator.__ne__, obs1[key], obs2[key])

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseDroppedObservationsFlattening(self, domain_name, task_name):
    """Ensure there is an additive Gaussian noise to the observation."""
    prob = 1.
    steps = np.random.randint(low=3, high=10)

    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={
            'dropped': {
                'enable': True,
                'observations_prob': prob,
                'observations_steps': steps,
            }
        },
        environment_kwargs={'flat_observation': True}
    )
    env.reset()
    env.step(np.array(0))  # Scalar actions aren't tolerated with noise.

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseDroppedObservationsValues(self, domain_name, task_name):
    """Ensure observations drop values."""
    steps = np.random.randint(low=3, high=10)
    prob = 1.
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={
            'dropped': {
                'enable': True,
                'observations_prob': prob,
                'observations_steps': steps,
            }
        })
    action_spec = env.action_spec()
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)

    for step in range(steps):
      # Verify that values are dropping for the first `steps` steps.
      if step == 1:
        # Cancel the dropped values after the first sequence.
        env._task._noise_dropped_obs_steps = 0.
      obs = env.step(copy.copy(one_action))[3]
      for key in obs:
        if isinstance(obs[key], np.ndarray):
          np.testing.assert_array_equal(obs[key], np.zeros(obs[key].shape))
        else:
          np.testing.assert_array_equal(obs[key], 0.)
    obs = env.step(copy.copy(one_action))[3]
    # Ensure observation is not filled with zeros.
    for key in obs:
      obs[key] += np.random.normal()
    # Pass observation through the base class that in charge of dropping values.
    obs = realworld_env.Base.get_observation(env._task, env._physics, obs)
    for key in obs:
      # Verify that values have stopped dropping.
      if isinstance(obs[key], np.ndarray):
        np.testing.assert_array_compare(operator.__ne__, obs[key],
                                        np.zeros(obs[key].shape))
      else:
        np.testing.assert_array_compare(operator.__ne__, obs[key], 0.)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseDroppedActionsValues(self, domain_name, task_name):
    """Ensure observations drop values."""
    steps = np.random.randint(low=3, high=10)
    prob = 1.
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={
            'dropped': {
                'enable': True,
                'actions_prob': prob,
                'actions_steps': steps,
            }
        })
    env.reset()
    action_spec = env.action_spec()
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)
    if hasattr(action_spec, 'minimum'):
      one_action = np.minimum(action_spec.maximum, one_action)

    for step in range(steps):
      # Verify that values are dropping for the first `steps` steps.
      if step == 1:
        # Cancel the dropped values after the first sequence.
        env._task._noise_dropped_action_steps = 0.
      _ = env.step(copy.copy(one_action))
      action = env.physics.control()
      if isinstance(action, np.ndarray):
        np.testing.assert_array_equal(action, np.zeros(action.shape))
      else:
        np.testing.assert_array_equal(action, 0.)
    # Ensure values are no longer dropping.
    _ = env.step(copy.copy(one_action))
    action = env.physics.control()
    if isinstance(action, np.ndarray):
      np.testing.assert_array_compare(operator.__ne__, action,
                                      np.zeros(action.shape))
    else:
      np.testing.assert_array_compare(operator.__ne__, action, 0.)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseStuckObservationsValues(self, domain_name, task_name):
    """Ensure observations have stuck values."""
    steps = np.random.randint(low=3, high=10)
    prob = 1.
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={
            'stuck': {
                'enable': True,
                'observations_prob': prob,
                'observations_steps': steps,
            }
        })
    action_spec = env.action_spec()
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)

    prev_obs = None
    for step in range(steps):
      # Verify that values are stuck for the first `steps` steps.
      if step == 1:
        # Cancel the stuck values after the first sequence.
        env._task._noise_stuck_obs_steps = 0.
      obs = env.step(copy.copy(one_action))[3]
      if not prev_obs:
        prev_obs = copy.deepcopy(obs)
      for key in obs:
        np.testing.assert_array_equal(obs[key], prev_obs[key])
      prev_obs = copy.deepcopy(obs)
    # Perturb observation.
    for key in obs:
      obs[key] += np.random.normal()
    # Pass observation through the base class that in charge of stuck values.
    obs = realworld_env.Base.get_observation(env._task, env._physics, obs)
    for key in obs:
      # Verify that values have stopped getting stuck.
      np.testing.assert_array_compare(operator.__ne__, obs[key], prev_obs[key])

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseStuckActionsValues(self, domain_name, task_name):
    """Ensure observations have stuck values."""
    steps = np.random.randint(low=3, high=10)
    prob = 1.
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={
            'stuck': {
                'enable': True,
                'actions_prob': prob,
                'actions_steps': steps,
            }
        })
    env.reset()
    action_spec = env.action_spec()
    zero_action = np.zeros(shape=action_spec.shape, dtype=action_spec.dtype)
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)
    if hasattr(action_spec, 'minimum'):
      one_action = np.minimum(action_spec.maximum, one_action)

    # Get the action stuck for the next `steps` steps.
    env.step(copy.copy(one_action))
    # Cancel the stuck values after the first sequence.
    env._task._noise_stuck_action_steps = 0.

    for _ in range(steps):
      # Verify that values are stuck for the first `steps` steps.
      np.testing.assert_array_equal(env.physics.control(), one_action)
      # Apply a different action.
      env.step(copy.copy(zero_action))
    # Verify that zero_action executed after action becoming un-stuck.
    np.testing.assert_array_equal(env.physics.control(), zero_action)

    for step in range(17):
      # Alternate actions and make sure they don't get stuck.
      action = zero_action if step % 2 == 0 else one_action
      env.step(copy.copy(action))
      np.testing.assert_array_equal(env.physics.control(), action)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testNoiseRepetitionActions(self, domain_name, task_name):
    """Ensure actions are being repeated."""
    steps = np.random.randint(low=3, high=10)
    prob = 1.
    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        noise_spec={
            'repetition': {
                'enable': True,
                'actions_prob': prob,
                'actions_steps': steps,
            }
        })
    action_spec = env.action_spec()
    zero_action = np.zeros(shape=action_spec.shape, dtype=action_spec.dtype)
    one_action = np.ones(shape=action_spec.shape, dtype=action_spec.dtype)
    if hasattr(action_spec, 'minimum'):
      one_action = np.minimum(action_spec.maximum, one_action)

    env.reset()
    env.step(copy.copy(zero_action))
    # Verify that all the actions are zero_action for 'steps' time steps.
    for _ in range(steps):
      np.testing.assert_array_equal(env.physics.control(), zero_action)
      env.step(copy.copy(one_action))
    # Verify that all the actions are one_action for 'steps' time steps.
    for _ in range(steps):
      np.testing.assert_array_equal(env.physics.control(), one_action)
      env.step(copy.copy(zero_action))
    np.testing.assert_array_equal(env.physics.control(), zero_action)

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testPerturbRandomWalk(self, domain_name, task_name):
    """Ensure parameter is perturbed on each reset."""
    period = 3

    perturb_scheduler = 'drift_pos'
    if domain_name == 'cartpole':
      perturb_param = 'pole_length'
    elif domain_name == 'walker':
      perturb_param = 'thigh_length'
    elif domain_name == 'humanoid':
      perturb_param = 'contact_friction'
    elif domain_name == 'quadruped':
      perturb_param = 'shin_length'
    elif domain_name == 'manipulator':
      perturb_param = 'lower_arm_length'
    else:
      raise ValueError('Unknown environment name: %s' % domain_name)

    env = rwrl.load(
        domain_name=domain_name,
        task_name=task_name,
        perturb_spec={
            'enable': True,
            'period': period,
            'param': perturb_param,
            'scheduler': perturb_scheduler
        })

    def get_param_val():
      if domain_name == 'cartpole':
        return env._physics.named.model.geom_size['pole_1', 1]
      elif domain_name == 'walker':
        return env._physics.named.model.geom_size['right_thigh', 1]
      elif domain_name == 'humanoid':
        return env._physics.named.model.geom_friction['right_right_foot', 0]
      elif domain_name == 'quadruped':
        return env._physics.named.model.geom_size['shin_front_left', 1]
      elif domain_name == 'manipulator':
        return env._physics.named.model.geom_size['lower_arm', 1]
      else:
        pass

    # Verify that first reset changes value.
    val_old = get_param_val()
    env.reset()
    val_new = get_param_val()
    self.assertNotEqual(val_old, val_new)

    # Verify that the parameter changes only each `period` number of times.
    val_old = val_new
    for unused_count1 in range(1):
      for unused_count2 in range(period - 1):
        env.reset()
        val_new = get_param_val()
        self.assertEqual(val_old, val_new)
      env.reset()
      val_new = get_param_val()
      self.assertNotEqual(val_old, val_new)
      val_old = val_new

  @parameterized.named_parameters(*rwrl.ALL_TASKS)
  def testCombinedChallenges(self, domain_name, task_name):
    """Ensure the combined challenges are properly defined."""
    all_combined_challenges = ['easy', 'medium', 'hard']

    # Verify that a non-specified combined challenge breaks the code.
    with self.assertRaises(ValueError):
      _ = rwrl.load(
          domain_name=domain_name,
          task_name=task_name,
          combined_challenge='random_name')

    # Verify specs can't be specified if combined challenge is specified.
    for combined_challenge in all_combined_challenges:
      with self.assertRaises(ValueError):
        _ = rwrl.load(
            domain_name=domain_name,
            task_name=task_name,
            safety_spec={'enable': True},
            delay_spec={'enable': True},
            noise_spec={'enable': True},
            perturb_spec={'enable': True},
            dimensionality_spec={'enable': True},
            multiobj_spec={'enable': True},
            combined_challenge=combined_challenge)

    # Verify the combined challenges are correctly set.
    for combined_challenge in all_combined_challenges:
      env = rwrl.load(
          domain_name=domain_name,
          task_name=task_name,
          combined_challenge=combined_challenge)
      if combined_challenge == 'easy':
        # Delay.
        self.assertTrue(env._task._delay_enabled)
        self.assertEqual(env._task._buffer_observations_len, 3+1)
        self.assertEqual(env._task._buffer_actions_len, 3+1)
        self.assertEqual(env._task._buffer_rewards_len, 10+1)
        # Noise.
        self.assertTrue(env._task._noise_guassian_enabled)
        self.assertEqual(env._task._noise_gaussian_observations, 0.1)
        self.assertEqual(env._task._noise_gaussian_actions, 0.1)
        self.assertTrue(env._task._noise_dropped_enabled)
        self.assertEqual(env._task._noise_dropped_obs_prob, 0.01)
        self.assertEqual(env._task._noise_dropped_obs_steps, 1)
        self.assertTrue(env._task._noise_stuck_enabled)
        self.assertEqual(env._task._noise_stuck_obs_prob, 0.01)
        self.assertEqual(env._task._noise_stuck_obs_steps, 1)
        self.assertTrue(env._task._noise_repetition_enabled)
        self.assertEqual(env._task._noise_repetition_actions_prob, 1.0)
        self.assertEqual(env._task._noise_repetition_actions_steps, 1)
        # Perturbation.
        self.assertTrue(env._task._perturb_enabled)
        self.assertEqual(env._task._perturb_period, 1)
        self.assertEqual(env._task._perturb_scheduler, 'uniform')
        if domain_name == 'cartpole':
          self.assertEqual(env._task._perturb_param, 'pole_length')
          self.assertEqual(env._task._perturb_min, 0.9)
          self.assertEqual(env._task._perturb_max, 1.1)
          self.assertEqual(env._task._perturb_std, 0.02)
        elif domain_name == 'quadruped':
          self.assertEqual(env._task._perturb_param, 'shin_length')
          self.assertEqual(env._task._perturb_min, 0.25)
          self.assertEqual(env._task._perturb_max, 0.3)
          self.assertEqual(env._task._perturb_std, 0.005)
        elif domain_name == 'walker':
          self.assertEqual(env._task._perturb_param, 'thigh_length')
          self.assertEqual(env._task._perturb_min, 0.225)
          self.assertEqual(env._task._perturb_max, 0.25)
          self.assertEqual(env._task._perturb_std, 0.002)
        elif domain_name == 'humanoid':
          self.assertEqual(env._task._perturb_param, 'contact_friction')
          self.assertEqual(env._task._perturb_min, 0.6)
          self.assertEqual(env._task._perturb_max, 0.8)
          self.assertEqual(env._task._perturb_std, 0.02)
        # Dimensionality.
        self.assertTrue(env._task._dimensionality_enabled)
        self.assertEqual(env._task._num_random_state_observations, 10)
        # Safety.
        self.assertFalse(env._task._safety_enabled)
        # Multi-objective.
        self.assertFalse(env._task._multiobj_enabled)
      elif combined_challenge == 'medium':
        # Delay.
        self.assertTrue(env._task._delay_enabled)
        self.assertEqual(env._task._buffer_observations_len, 6+1)
        self.assertEqual(env._task._buffer_actions_len, 6+1)
        self.assertEqual(env._task._buffer_rewards_len, 20+1)
        # Noise.
        self.assertTrue(env._task._noise_guassian_enabled)
        self.assertEqual(env._task._noise_gaussian_observations, 0.3)
        self.assertEqual(env._task._noise_gaussian_actions, 0.3)
        self.assertTrue(env._task._noise_dropped_enabled)
        self.assertEqual(env._task._noise_dropped_obs_prob, 0.05)
        self.assertEqual(env._task._noise_dropped_obs_steps, 5)
        self.assertTrue(env._task._noise_stuck_enabled)
        self.assertEqual(env._task._noise_stuck_obs_prob, 0.05)
        self.assertEqual(env._task._noise_stuck_obs_steps, 5)
        self.assertTrue(env._task._noise_repetition_enabled)
        self.assertEqual(env._task._noise_repetition_actions_prob, 1.0)
        self.assertEqual(env._task._noise_repetition_actions_steps, 2)
        # Perturbation.
        self.assertTrue(env._task._perturb_enabled)
        self.assertEqual(env._task._perturb_period, 1)
        self.assertEqual(env._task._perturb_scheduler, 'uniform')
        if domain_name == 'cartpole':
          self.assertEqual(env._task._perturb_param, 'pole_length')
          self.assertEqual(env._task._perturb_min, 0.7)
          self.assertEqual(env._task._perturb_max, 1.7)
          self.assertEqual(env._task._perturb_std, 0.1)
        elif domain_name == 'quadruped':
          self.assertEqual(env._task._perturb_param, 'shin_length')
          self.assertEqual(env._task._perturb_min, 0.25)
          self.assertEqual(env._task._perturb_max, 0.8)
          self.assertEqual(env._task._perturb_std, 0.05)
        elif domain_name == 'walker':
          self.assertEqual(env._task._perturb_param, 'thigh_length')
          self.assertEqual(env._task._perturb_min, 0.225)
          self.assertEqual(env._task._perturb_max, 0.4)
          self.assertEqual(env._task._perturb_std, 0.015)
        elif domain_name == 'humanoid':
          self.assertEqual(env._task._perturb_param, 'contact_friction')
          self.assertEqual(env._task._perturb_min, 0.5)
          self.assertEqual(env._task._perturb_max, 0.9)
          self.assertEqual(env._task._perturb_std, 0.04)
        # Dimensionality.
        self.assertTrue(env._task._dimensionality_enabled)
        self.assertEqual(env._task._num_random_state_observations, 20)
        # Safety.
        self.assertFalse(env._task._safety_enabled)
        # Multi-objective.
        self.assertFalse(env._task._multiobj_enabled)
      elif combined_challenge == 'hard':
        # Delay.
        self.assertTrue(env._task._delay_enabled)
        self.assertEqual(env._task._buffer_observations_len, 9+1)
        self.assertEqual(env._task._buffer_actions_len, 9+1)
        self.assertEqual(env._task._buffer_rewards_len, 40+1)
        # Noise.
        self.assertTrue(env._task._noise_guassian_enabled)
        self.assertEqual(env._task._noise_gaussian_observations, 1.0)
        self.assertEqual(env._task._noise_gaussian_actions, 1.0)
        self.assertTrue(env._task._noise_dropped_enabled)
        self.assertEqual(env._task._noise_dropped_obs_prob, 0.1)
        self.assertEqual(env._task._noise_dropped_obs_steps, 10)
        self.assertTrue(env._task._noise_stuck_enabled)
        self.assertEqual(env._task._noise_stuck_obs_prob, 0.1)
        self.assertEqual(env._task._noise_stuck_obs_steps, 10)
        self.assertTrue(env._task._noise_repetition_enabled)
        self.assertEqual(env._task._noise_repetition_actions_prob, 1.0)
        self.assertEqual(env._task._noise_repetition_actions_steps, 3)
        # Perturbation.
        self.assertTrue(env._task._perturb_enabled)
        self.assertEqual(env._task._perturb_period, 1)
        self.assertEqual(env._task._perturb_scheduler, 'uniform')
        if domain_name == 'cartpole':
          self.assertEqual(env._task._perturb_param, 'pole_length')
          self.assertEqual(env._task._perturb_min, 0.5)
          self.assertEqual(env._task._perturb_max, 2.3)
          self.assertEqual(env._task._perturb_std, 0.15)
        elif domain_name == 'quadruped':
          self.assertEqual(env._task._perturb_param, 'shin_length')
          self.assertEqual(env._task._perturb_min, 0.25)
          self.assertEqual(env._task._perturb_max, 1.4)
          self.assertEqual(env._task._perturb_std, 0.1)
        elif domain_name == 'walker':
          self.assertEqual(env._task._perturb_param, 'thigh_length')
          self.assertEqual(env._task._perturb_min, 0.225)
          self.assertEqual(env._task._perturb_max, 0.55)
          self.assertEqual(env._task._perturb_std, 0.04)
        elif domain_name == 'humanoid':
          self.assertEqual(env._task._perturb_param, 'contact_friction')
          self.assertEqual(env._task._perturb_min, 0.4)
          self.assertEqual(env._task._perturb_max, 1.0)
          self.assertEqual(env._task._perturb_std, 0.06)
        # Dimensionality.
        self.assertTrue(env._task._dimensionality_enabled)
        self.assertEqual(env._task._num_random_state_observations, 50)
        # Safety.
        self.assertFalse(env._task._safety_enabled)
        # Multi-objective.
        self.assertFalse(env._task._multiobj_enabled)


if __name__ == '__main__':
  absltest.main()
