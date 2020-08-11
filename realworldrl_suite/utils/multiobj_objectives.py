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

"""Some basic complementary rewards for the multi-objective task."""

import abc
import numpy as np


class Objective(abc.ABC):

  @abc.abstractmethod
  def get_objectives(self, task_obj):
    raise NotImplementedError

  @abc.abstractmethod
  def merge_reward(self, task_obj, reward, alpha):
    raise NotImplementedError


class SafetyObjective(Objective):
  """This class defines an extra objective related to safety."""

  def get_objectives(self, task_obj):
    """Returns the safety objective: sum of satisfied constraints."""
    if task_obj.safety_enabled:
      num_constraints = float(task_obj.constraints_obs.shape[0])
      num_satisfied = task_obj.constraints_obs.sum()
      s_reward = num_satisfied / num_constraints
      return np.array([s_reward])
    else:
      raise Exception('Safety not enabled.  Safety-based multi-objective reward'
                      ' requires safety spec to be enabled.')

  def merge_reward(self, task_obj, physics, base_reward, alpha):
    """Returns the sum of safety violations normalized to 1."""
    s_reward = self.get_objectives(task_obj)[0]
    return (1 - alpha) * base_reward + alpha * s_reward


OBJECTIVES = {'safety': SafetyObjective}
