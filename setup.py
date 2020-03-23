# Copyright 2020 The Read World RL Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import sys
import unittest

from setuptools import find_packages
from setuptools import setup
from setuptools.command.test import test as TestCommandBase

REQUIRED_PACKAGES = [
    'six', 'absl-py', 'numpy', 'dm-env', 'dm-control', 'lxml', 'scipy',
    'matplotlib'
]


class StderrWrapper(io.IOBase):

  def write(self, *args, **kwargs):
    return sys.stderr.write(*args, **kwargs)

  def writeln(self, *args, **kwargs):
    if args or kwargs:
      sys.stderr.write(*args, **kwargs)
    sys.stderr.write('\n')


# Needed to ensure that flags are correctly parsed.
class Test(TestCommandBase):

  def run_tests(self):
    # Import absl inside run, where dependencies have been loaded already.
    from absl import app  # pylint: disable=g-import-not-at-top

    def main(_):
      test_loader = unittest.TestLoader()
      test_suite = test_loader.discover(
          'realworldrl_suite', pattern='*_test.py')
      stderr = StderrWrapper()
      result = unittest.TextTestResult(stderr, descriptions=True, verbosity=2)
      test_suite.run(result)
      result.printErrors()

      final_output = ('Tests run: {}.  '.format(result.testsRun) +
                      'Errors: {}  Failures: {}.'.format(
                          len(result.errors), len(result.failures)))

      header = '=' * len(final_output)
      stderr.writeln(header)
      stderr.writeln(final_output)
      stderr.writeln(header)

      if result.wasSuccessful():
        return 0
      else:
        return 1

    # Run inside absl.app.run to ensure flags parsing is done.
    return app.run(main)


def rwrl_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('realworldrl_suite', pattern='*_test.py')
  return test_suite


setup(
    name='realworldrl_suite',
    version='1.0',
    description='RL evaluation framework for the real world.',
    url='https://github.com/google-research/realworldrl_suite',
    author='Google',
    author_email='no-reply@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require={},
    platforms=['any'],
    license='Apache 2.0',
    cmdclass={
        'test': Test,
    },
)
