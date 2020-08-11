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

"""Runs a random policy on realworldrl."""

import os

from absl import app
from absl import flags
import numpy as np
import realworldrl_suite.environments as rwrl

flags.DEFINE_string('domain_name', 'cartpole', 'domain to solve')
flags.DEFINE_string('task_name', 'realworld_balance', 'task to solve')
flags.DEFINE_string('save_path', '/tmp/rwrl', 'where to save results')
flags.DEFINE_integer('total_episodes', 100, 'number of episodes')

FLAGS = flags.FLAGS


def random_policy(action_spec):

  def _act(timestep):
    del timestep
    return np.random.uniform(
        low=action_spec.minimum,
        high=action_spec.maximum,
        size=action_spec.shape)

  return _act


def run():
  """Runs a random agent on a given environment."""

  env = rwrl.load(
      domain_name=FLAGS.domain_name,
      task_name=FLAGS.task_name,
      safety_spec=dict(enable=True),
      delay_spec=dict(enable=True, actions=20),
      log_output=os.path.join(FLAGS.save_path, 'log.npz'),
      environment_kwargs=dict(
          log_safety_vars=True, log_every=20, flat_observation=True))

  policy = random_policy(action_spec=env.action_spec())

  rewards = []
  for _ in range(FLAGS.total_episodes):
    timestep = env.reset()
    total_reward = 0.
    while not timestep.last():
      action = policy(timestep)
      timestep = env.step(action)
      total_reward += timestep.reward
    rewards.append(total_reward)
  print('Random policy total reward per episode: {:.2f} +- {:.2f}'.format(
      np.mean(rewards), np.std(rewards)))


def main(argv):
  del argv  # Unused.
  run()


if __name__ == '__main__':
  app.run(main)
