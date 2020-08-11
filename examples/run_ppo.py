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

"""Trains an OpenAI Baselines PPO agent on realworldrl.

Note that OpenAI Gym is not installed with realworldrl by default.
See also github.com/openai/baselines for more information.

This example also relies on dm2gym for its gym environment wrapper.
See github.com/zuoxingdong/dm2gym for more information.
"""

import os

from absl import app
from absl import flags
from baselines import bench
from baselines.common.vec_env import dummy_vec_env
from baselines.ppo2 import ppo2
import dm2gym.envs.dm_suite_env as dm2gym
import realworldrl_suite.environments as rwrl

flags.DEFINE_string('domain_name', 'cartpole', 'domain to solve')
flags.DEFINE_string('task_name', 'realworld_balance', 'task to solve')
flags.DEFINE_string('save_path', '/tmp/rwrl', 'where to save results')
flags.DEFINE_boolean('verbose', True, 'whether to log to std output')
flags.DEFINE_string('network', 'mlp', 'name of network architecture')
flags.DEFINE_float('agent_discount', .99, 'discounting on the agent side')
flags.DEFINE_integer('nsteps', 100, 'number of steps per ppo rollout')
flags.DEFINE_integer('total_timesteps', 1000000, 'total steps for experiment')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate for optimizer')

FLAGS = flags.FLAGS


class GymEnv(dm2gym.DMSuiteEnv):
  """Wrapper that convert a realworldrl environment to a gym environment."""

  def __init__(self, env):
    """Constructor. We reuse the facilities from dm2gym."""
    self.env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': round(1. / self.env.control_timestep())
    }
    self.observation_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.observation_spec())
    self.action_space = dm2gym.convert_dm_control_to_gym_space(
        self.env.action_spec())
    self.viewer = None


def run():
  """Runs a PPO agent on a given environment."""

  def _load_env():
    """Loads environment."""
    raw_env = rwrl.load(
        domain_name=FLAGS.domain_name,
        task_name=FLAGS.task_name,
        safety_spec=dict(enable=True),
        delay_spec=dict(enable=True, actions=20),
        log_output=os.path.join(FLAGS.save_path, 'log.npz'),
        environment_kwargs=dict(
            log_safety_vars=True, log_every=20, flat_observation=True))
    env = GymEnv(raw_env)
    env = bench.Monitor(env, FLAGS.save_path)
    return env

  env = dummy_vec_env.DummyVecEnv([_load_env])

  ppo2.learn(
      env=env,
      network=FLAGS.network,
      lr=FLAGS.learning_rate,
      total_timesteps=FLAGS.total_timesteps,  # make sure to run enough steps
      nsteps=FLAGS.nsteps,
      gamma=FLAGS.agent_discount,
  )


def main(argv):
  del argv  # Unused.
  run()


if __name__ == '__main__':
  app.run(main)
