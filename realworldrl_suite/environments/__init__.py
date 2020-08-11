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

"""Safety environments for real-world RL."""
from realworldrl_suite.environments import cartpole
from realworldrl_suite.environments import humanoid
from realworldrl_suite.environments import manipulator
from realworldrl_suite.environments import quadruped
from realworldrl_suite.environments import walker

# This is a tuple of all the domains and tasks present in the suite.  It is
# currently used mainly for unit test coverage but can be useful if one wants
# to sweep over all the tasks.
ALL_TASKS = (('cartpole_balance', 'cartpole', 'realworld_balance'),
             ('cartpole_swingup', 'cartpole', 'realworld_swingup'),
             ('humanoid_stand', 'humanoid', 'realworld_stand'),
             ('humanoid_walk', 'humanoid', 'realworld_walk'),
             ('manipulator_bring_ball', 'manipulator', 'realworld_bring_ball'),
             ('manipulator_bring_peg', 'manipulator', 'realworld_bring_peg'),
             ('manipulator_insert_ball', 'manipulator',
              'realworld_insert_ball'),
             ('manipulator_insert_peg', 'manipulator', 'realworld_insert_peg'),
             ('quadruped_walk', 'quadruped', 'realworld_walk'),
             ('quadruped_run', 'quadruped', 'realworld_run'),
             ('walker_stand', 'walker', 'realworld_stand'),
             ('walker_walk', 'walker', 'realworld_walk'))

DOMAINS = dict(
    cartpole=cartpole, humanoid=humanoid, manipulator=manipulator,
    quadruped=quadruped, walker=walker)


def load(domain_name, task_name, **kwargs):
  return DOMAINS[domain_name].load(task_name, **kwargs)
