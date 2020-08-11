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

"""Tests for loggers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
import numpy as np
import numpy.testing as npt
from realworldrl_suite.utils import loggers


class LoggersTest(absltest.TestCase):

  def test_write(self):
    temp_file = self.create_tempfile()
    plogger = loggers.PickleLogger(path=temp_file.full_path)
    write_meta = np.random.randn(10, 10)
    push_data = np.random.randn(10, 10)
    save_data = np.random.randn(10, 10)
    plogger.set_meta(write_meta)
    plogger.push(push_data)
    plogger.save(data=save_data)
    with open(plogger.logs_path, 'rb') as f:
      read_data = np.load(f, allow_pickle=True)
      npt.assert_array_equal(read_data['meta'], write_meta)
      npt.assert_array_equal(read_data['stack'][0], push_data)
      npt.assert_array_equal(read_data['data'], save_data)


if __name__ == '__main__':
  absltest.main()
