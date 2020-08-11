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

"""Simple viewer for realworld environments."""
from absl import app
from absl import flags
from absl import logging

from dm_control import suite
from dm_control import viewer

import numpy as np
import realworldrl_suite.environments as rwrl

flags.DEFINE_enum('suite', 'rwrl', ['rwrl', 'dm_control'], 'Suite choice')
flags.DEFINE_string('domain_name', 'cartpole', 'domain name')
flags.DEFINE_string('task_name', 'realworld_balance', 'Task name')

FLAGS = flags.FLAGS


class RandomAgent(object):

  def __init__(self, action_spec):
    self.action_spec = action_spec

  def action(self, timestep):
    del timestep
    return np.random.uniform(
        self.action_spec.minimum,
        self.action_spec.maximum,
        size=self.action_spec.shape)


def main(_):
  if FLAGS.suite == 'dm_control':
    logging.info('Loading from dm_control...')
    env = suite.load(domain_name=FLAGS.domain_name, task_name=FLAGS.task_name)
  elif FLAGS.suite == 'rwrl':
    logging.info('Loading from rwrl...')
    env = rwrl.load(domain_name=FLAGS.domain_name, task_name=FLAGS.task_name)
  random_policy = RandomAgent(env.action_spec()).action
  viewer.launch(env, policy=random_policy)


if __name__ == '__main__':
  app.run(main)
