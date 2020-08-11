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

"""Helper lib to calculate RWRL evaluators from logged stats."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import pprint

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

FLAGS = flags.FLAGS
if 'stats_in' not in FLAGS:
  flags.DEFINE_string('stats_in', 'logs.npz', 'Filename to write out logs.')


_CONFIDENCE_LEVEL = .95
_ROLLING_WINDOW = 101
_COLORS = ['r', 'g', 'b', 'k', 'y']
_MARKERS = ['x', 'o', '*', '^', 'v']


class Evaluators(object):
  """Calculate standardized RWRL evaluators of a run."""

  def __init__(self, stats):
    """Generate object given stored statistics.

    Args:
      stats: list of standardized RWRL statistics.
    """
    self._meta = stats['meta'].item()
    self._stats = stats['data'].item()
    self._moving_average_cache = {}

  @property
  def task_name(self):
    return self._meta['task_name']

  def get_safety_evaluator(self):
    """Returns the RWRL safety function's evaluation.

    Function (3) in the RWRL paper, the per-constraint sum of violations.

    Returns:
      A dict of contraint_name -> # of violations in logs.
    """
    constraint_names = self._meta['safety_constraints']
    safety_stats = self._stats['safety_stats']
    violations = np.sum(safety_stats['total_violations'], axis=0)
    evaluator_results = collections.OrderedDict([
        (key, violations[idx]) for idx, key in enumerate(constraint_names)
    ])
    return evaluator_results

  def get_safety_plot(self, do_total=True, do_per_step=True):
    """Generates standardized plots describing safety constraint violations."""
    n_plots = int(do_total) + int(do_per_step)
    fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
    plot_idx = 0
    if do_total:
      self._get_safety_totals_plot(axes[plot_idx], self._stats['safety_stats'])
      plot_idx += 1
    if do_per_step:
      self._get_safety_per_step_plots(axes[plot_idx],
                                      self._stats['safety_stats'])
    return fig

  def _get_safety_totals_plot(self, ax, safety_stats):
    """Generates plots describing total # of violations / episode."""
    meta = self.meta
    violations_labels = meta['safety_constraints']
    total_violations = safety_stats['total_violations'].T

    for idx, violations in enumerate(total_violations):
      label = violations_labels[idx]
      ax.plot(np.arange(violations.shape[0]), violations, label=label)

    ax.set_title('# violations / episode')
    ax.legend()
    ax.set_ylabel('# violations')
    ax.set_xlabel('Episode')
    ax.plot()

  def _get_safety_per_step_plots(self, ax, safety_stats):
    """Generates plots describing mean # of violations / timestep."""
    meta = self.meta
    violations_labels = meta['safety_constraints']
    per_step_violations = safety_stats['per_step_violations']

    for idx, violations in enumerate(per_step_violations.T):
      label = violations_labels[idx]
      ax.plot(
          np.arange(violations.shape[0]), violations, label=label, alpha=0.75)

    ax.set_title('Mean violations / timestep')
    ax.legend(loc='upper right')
    ax.set_ylabel('Mean # violations')
    ax.set_xlabel('Timestep')
    ax.plot()

  def get_safety_vars_plot(self):
    """Get plots for statistics of safety-related variables."""
    if 'safety_vars_stats' not in self.stats:
      raise ValueError('No safety vars statistics present in this evaluator.')

    safety_vars = self.stats['safety_vars_stats'][0].keys()
    n_plots = len(safety_vars)
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 6 * n_plots))

    for idx, var in enumerate(safety_vars):
      series = collections.defaultdict(list)
      for ep in self.stats['safety_vars_stats']:
        for stat in ep[var]:
          series[stat].append(ep[var][stat])
      ax = axes[idx]
      for stat in ['min', 'max']:
        ax.plot(np.squeeze(np.array(series[stat])), label=stat)
      x = range(len(series['mean']))

      mean = np.squeeze(np.array(series['mean']))
      std_dev = np.squeeze(np.array(series['std_dev']))
      ax.plot(x, mean, label='Value')
      ax.fill_between(
          range(len(series['mean'])), mean - std_dev, mean + std_dev, alpha=0.3)
      ax.set_title('Stats for {}'.format(var))
      ax.legend()
      ax.spines['top'].set_visible(False)

      ax.xaxis.set_ticks_position('bottom')
      ax.set_xlabel('Episode #')
      ax.set_ylabel('Magnitude')
      ax.plot()
    return fig

  def _get_return_per_step_plot(self, ax, return_stats):
    """Plot per-episode return."""
    returns = return_stats['episode_totals']

    ax.plot(np.arange(returns.shape[0]), returns, label='Return')

    ax.set_title('Return / Episode')
    ax.legend()
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode')

  def get_return_plot(self):
    fig, ax = plt.subplots()
    self._get_return_per_step_plot(ax, self.stats['return_stats'])
    return fig

  def _moving_average(self, values, window=1, stride=1, p=None):
    """Computes moving averages and confidence intervals."""
    # Cache results for convenience.
    key = (id(values), window, stride, p)
    if key in self._moving_average_cache:
      return self._moving_average_cache[key]
    # Compute rolling windows efficiently.
    def _rolling_window(a, window):
      shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
      strides = a.strides + (a.strides[-1],)
      return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    x = np.mean(
        _rolling_window(np.arange(len(values)), window), axis=-1)[::stride]
    y = _rolling_window(values, window)
    y_mean = np.mean(y, axis=-1)[::stride]
    y_lower, y_upper = _errorbars(y, axis=-1, p=p)
    ret = (x, y_mean, (y_lower, y_upper))
    self._moving_average_cache[key] = ret
    return ret

  def get_convergence_episode(self):
    """Returns the first episode that reaches the final return."""
    values = self.stats['return_stats']['episode_totals']
    _, y, (y_lower, _) = self._moving_average(
        values, window=_ROLLING_WINDOW, p=_CONFIDENCE_LEVEL)
    # The convergence is established as the first time the average return
    # is above the lower bounds of the final return.
    first_episode = max(np.argmax(y >= y_lower[-1]), 1)
    return first_episode

  def get_final_return(self):
    """Returns the average final return."""
    values = self.stats['return_stats']['episode_totals']
    _, y, (_, _) = self._moving_average(values, window=_ROLLING_WINDOW,
                                        p=_CONFIDENCE_LEVEL)
    return y[-1]

  def get_absolute_regret(self):
    """Returns the total regret until convergence."""
    values = self.stats['return_stats']['episode_totals']
    first_episode = self.get_convergence_episode()
    final_return = self.get_final_return()
    regret = np.sum(final_return - values[:first_episode])
    return regret

  def get_normalized_regret(self):
    """Returns the normalized regret."""
    final_return = self.get_final_return()
    return self.get_absolute_regret() / final_return

  def get_normalized_instability(self):
    """Returns the ratio of episodes that dip below the final return."""
    values = self.stats['return_stats']['episode_totals']
    _, _, (y_lower, _) = self._moving_average(
        values, window=_ROLLING_WINDOW, p=_CONFIDENCE_LEVEL)
    first_episode = self.get_convergence_episode()
    if first_episode == len(values) - 1:
      return None
    episodes = np.arange(len(values))
    unstable_episodes = np.where(
        np.logical_and(values < y_lower[-1], episodes > first_episode))[0]
    return float(len(unstable_episodes)) / (len(values) - first_episode - 1)

  def get_convergence_plot(self):
    """Plots an illustration of the convergence analysis."""
    fig, ax = plt.subplots()
    first_episode = self.get_convergence_episode()

    values = self.stats['return_stats']['episode_totals']
    ax.plot(np.arange(len(values)), values, color='steelblue', lw=2, alpha=.9,
            label='Return')
    ax.axvline(first_episode, color='seagreen', lw=2, label='Converged')
    ax.set_xlim(left=0, right=first_episode * 2)

    ax.set_title('Normalized regret = {:.3f}'.format(
        self.get_normalized_regret()))
    ax.legend()
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode')
    return fig

  def get_stability_plot(self):
    """Plots an illustration of the algorithm stability."""
    fig, ax = plt.subplots()
    first_episode = self.get_convergence_episode()

    values = self.stats['return_stats']['episode_totals']
    _, _, (y_lower, _) = self._moving_average(
        values, window=_ROLLING_WINDOW, p=_CONFIDENCE_LEVEL)
    episodes = np.arange(len(values))
    unstable_episodes = np.where(
        np.logical_and(values < y_lower[-1], episodes > first_episode))[0]

    ax.plot(episodes, values, color='steelblue', lw=2, alpha=.9,
            label='Return')
    for i, episode in enumerate(unstable_episodes):
      ax.axvline(episode, color='salmon', lw=2,
                 label='Unstable' if i == 0 else None)
    ax.axvline(first_episode, color='seagreen', lw=2, label='Converged')

    ax.set_title('Normalized instability = {:.3f}%'.format(
        self.get_normalized_instability() * 100.))
    ax.legend()
    ax.set_ylabel('Return')
    ax.set_xlabel('Episode')
    return fig

  def get_multiobjective_plot(self):
    """Plots an illustration of the multi-objective analysis."""
    fig, ax = plt.subplots()

    values = self.stats['multiobj_stats']['episode_totals']
    for i in range(values.shape[1]):
      ax.plot(np.arange(len(values[:, i])), values[:, i],
              color=_COLORS[i % len(_COLORS)], lw=2, alpha=.9,
              label='Objective {}'.format(i))
    ax.legend()
    ax.set_ylabel('Objective value')
    ax.set_xlabel('Episode')
    return fig

  def get_standard_evaluators(self):
    """This method returns the standard RWRL evaluators.

    Returns:
      Dict of evaluators:
        Off-Line Performance: NotImplemented
        Efficiency: NotImplemented
        Safety: Per-constraint # of violations
        Robustness: NotImplemented
        Discernment: NotImplemented
    """
    evaluators = collections.OrderedDict(
        offline=None,
        efficiency=None,
        safety=self.get_safety_evaluator(),
        robustness=None,
        discernment=None)
    return evaluators

  @property
  def meta(self):
    return self._meta

  @property
  def stats(self):
    return self._stats


def _errorbars(values, axis=None, p=None):
  mean = np.mean(values, axis=axis)
  if p is None:
    std = np.std(values, axis=axis)
    return mean - std, mean + std
  return scipy.stats.t.interval(.95, values.shape[axis] - 1, loc=mean,
                                scale=scipy.stats.sem(values, axis=-1))


def _map(fn, values):
  return dict((k, fn(v)) for k, v in values.items())


def get_normalized_regret(evaluator_list):
  """Computes normalized regret for multiple seeds on multiple tasks."""
  values = collections.defaultdict(list)
  for e in evaluator_list:
    values[e.task_name].append(e.get_normalized_regret())
  return _map(np.mean, values), _map(np.std, values)


def get_regret_plot(evaluator_list):
  """Plots normalized regret for multiple seeds on multiple tasks."""
  means, stds = get_normalized_regret(evaluator_list)
  task_names = sorted(means.keys())
  heights = []
  errorbars = []
  for task_name in task_names:
    heights.append(means[task_name])
    errorbars.append(stds[task_name])
  x = np.arange(len(task_names))
  fig, ax = plt.subplots()
  ax.bar(x, heights, yerr=errorbars)
  ax.set_xticks(x)
  ax.set_xticklabels(task_names)
  ax.set_ylabel('Normalized regret')
  return fig


def get_return_plot(evaluator_list, stride=500):
  """Plots the return per episode for multiple seeds on multiple tasks."""
  values = collections.defaultdict(list)
  for e in evaluator_list:
    values[e.task_name].append(e.stats['return_stats']['episode_totals'])
  values = _map(np.vstack, values)
  means = _map(functools.partial(np.mean, axis=0), values)
  stds = _map(functools.partial(np.std, axis=0), values)

  fig, ax = plt.subplots()
  for i, task_name in enumerate(means):
    idx = i % len(_COLORS)
    x = np.arange(len(means[task_name]))
    ax.plot(x, means[task_name], lw=2, color=_COLORS[idx], alpha=.6, label=None)
    ax.plot(x[::stride], means[task_name][::stride], 'o', lw=2,
            marker=_MARKERS[idx], markersize=10, color=_COLORS[idx],
            label=task_name)
    ax.fill_between(x, means[task_name] - stds[task_name],
                    means[task_name] + stds[task_name], alpha=.4, lw=2,
                    color=_COLORS[idx])
  ax.legend()
  ax.set_ylabel('Return')
  ax.set_xlabel('Episode')
  return fig


def get_multiobjective_plot(evaluator_list, stride=500):
  """Plots the objectives per episode for multiple seeds on multiple tasks."""
  num_objectives = (
      evaluator_list[0].stats['multiobj_stats']['episode_totals'].shape[1])
  values = [collections.defaultdict(list) for _ in range(num_objectives)]
  for e in evaluator_list:
    for i in range(num_objectives):
      values[i][e.task_name].append(
          e.stats['multiobj_stats']['episode_totals'][:, i])
  means = [None] * num_objectives
  stds = [None] * num_objectives
  for i in range(num_objectives):
    values[i] = _map(np.vstack, values[i])
    means[i] = _map(functools.partial(np.mean, axis=0), values[i])
    stds[i] = _map(functools.partial(np.std, axis=0), values[i])

  fig, axes = plt.subplots(num_objectives, 1, figsize=(8, 6 * num_objectives))
  for objective_idx in range(num_objectives):
    ax = axes[objective_idx]
    for i, task_name in enumerate(means[objective_idx]):
      m = means[objective_idx][task_name]
      s = stds[objective_idx][task_name]
      idx = i % len(_COLORS)
      x = np.arange(len(m))
      ax.plot(x, m, lw=2, color=_COLORS[idx], alpha=.6, label=None)
      ax.plot(x[::stride], m[::stride], 'o', lw=2, marker=_MARKERS[idx],
              markersize=10, color=_COLORS[idx], label=task_name)
      ax.fill_between(x, m - s, m + s, alpha=.4, lw=2, color=_COLORS[idx])
    ax.legend()
    ax.set_ylabel('Objective {}'.format(objective_idx))
    ax.set_xlabel('Episode')
  return fig


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  with open(FLAGS.stats_in, 'rb') as f:
    stats = np.load(f)
    evals = Evaluators(stats)
    pp = pprint.PrettyPrinter(indent=2)
    pp.pprint(evals.get_standard_evaluators())


if __name__ == '__main__':
  flags.mark_flag_as_required('stats_in')

  app.run(main)
