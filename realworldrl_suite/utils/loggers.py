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

"""Various logging classes to easily log to different backends."""
import copy
import io
import os
import time

from absl import logging
import numpy as np




class PickleLogger(object):
  """Saves data to a pickle file.

  This logger will save data as a list of stored elements, written to a python3
  pickle file.  This data can then get retrieved by a PickleReader class.
  """

  def __init__(self, path):
    """Generate a PickleLogger object.

    Args:
      path: string of path to write to
    """
    self._meta = []
    self._stack = []
    # To know if our current run is responsible for existing files.
    # Used to figure out if we can overwrite a previous log, or if we've
    # been checkpointed in between.
    ts = str(int(time.time()))
    split = os.path.split(path)
    self._pickle_path = os.path.join(split[0], '{}-{}'.format(ts, split[1]))

  def set_meta(self, meta):
    """Pickleable object of metadata about the task."""
    self._meta = copy.deepcopy(meta)

  def push(self, data):
    self._stack.append(copy.deepcopy(data))

  def save(self, data=None):
    """Save data to disk.

    Args:
      data: Additional data structure you want to save to disk, will use the
        'data' key for storage.
    """
    logs = self.logs
    if data is not None:
      logs['data'] = data
    with open(self._pickle_path, 'wb') as f:
      np.savez_compressed(f, **logs)
    logging.info('Saved stats to %s.', format(self._pickle_path))

  @property
  def logs(self):
    return dict(meta=self._meta, stack=self._stack)

  @property
  def logs_path(self):
    return self._pickle_path
