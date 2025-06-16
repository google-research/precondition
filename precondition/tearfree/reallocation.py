# Copyright 2025 The precondition Authors.
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

"""Sketchy memory reallocation across layers based on checkpoint info."""

import concurrent
import copy
import os
from typing import Any, Optional

from absl import app
from absl import flags
from flax.training import checkpoints as flax_chpts
from jax import numpy as jnp


def load_checkpoints(file_dir: str) -> list[Any]:
  """Load checkpoints from the directory where checkpoints are saved."""
  files = []

  for f in os.listdir(file_dir):
    if f.startswith('ckpt_'):
      v = int(f[len('ckpt_'):])
      files.append((v, f))
    files.sort()
  return files


def create_state(file_dir: str, idx: list[int]):
  """Create states from selected checkpoints."""

  files = load_checkpoints(file_dir)

  def extract_state(args):
    _, prefix = args
    restored = flax_chpts.restore_checkpoint(
        file_dir, target=None, prefix=prefix
    )
    state = restored['optimizer_state']
    if 'base_state' in state:
      state = state['base_state']
    return state, prefix

  with concurrent.futures.ThreadPoolExecutor() as tpe:
    states, _ = zip(*tpe.map(extract_state, [files[i] for i in idx]))

  return tuple(states)


def layers_and_axes(sketches: dict[str, Any]):
  """List names for all of the layers."""

  def extract_paths(sketches, parent_key='', paths=None):
    if paths is None:
      paths = set()

    for key, value in sketches.items():
      new_key = parent_key + '/' + key if parent_key else key
      if isinstance(value, dict):
        extract_paths(value, new_key, paths)
      else:
        paths.add(parent_key)
    return paths

  all_layer_names = extract_paths(sketches)
  layer_names = {name for name in all_layer_names if name[-2] == '/'}
  axes_set = {name[-1] for name in all_layer_names if name[-2] == '/'}
  return layer_names, len(axes_set)


def create_groups(
    sketches: dict[str, Any], layer_names: set[str]
) -> dict[int, list[str]]:
  """Create groups for layers based on their dimensions."""

  group_dict = {}

  for name in layer_names:
    dirs = name.split('/')
    carry = sketches
    for d in dirs:
      carry = carry[d]
    if 'dim' in carry:
      key = carry['dim']
    else:
      key = carry['eigvecs'].shape[0]
    if key in group_dict:
      group_dict[key].append(name)
    else:
      group_dict[key] = [name]

  return group_dict


def score_fn(
    states, rule, layer_names: set[str], running_average=False
) -> dict[str, float]:
  """Calculate scores for each layer."""

  feasible_rules = [
      'ggt_intrinsic_rank',
      'ggt_trace',
      'tail_rho',
      'sketch_intrinsic_rank',
      'sketch_trace',
  ]

  if rule not in feasible_rules:
    raise NotImplementedError()

  if rule.startswith('ggt'):
    target = 'ema_ggt'
  elif rule.startswith('sketch'):
    target = 'eigvals'
  else:
    target = 'tail'
  ops_dict = {
      'ggt_intrinsic_rank': lambda x: jnp.trace(x) / jnp.linalg.norm(x, 2),
      'ggt_trace': jnp.trace,
      'tail_rho': lambda x: x,
      'sketch_intrinsic_rank': (
          lambda x: jnp.sum(x) / jnp.max(x) if jnp.sum(x) else 0
      ),
      'sketch_trace': jnp.sum,
  }
  if running_average:
    sketches = [
        st['inner_state']['0']['direction']['1']['sketches'] for st in states
    ]
  else:
    sketches = [states[-1]['inner_state']['0']['direction']['1']['sketches']]

  len_sketches = len(sketches)
  score_dict = {}
  for name in layer_names:
    dirs = name.split('/')
    current_target = copy.deepcopy(sketches)
    for i in range(len_sketches):
      ct = current_target[i]
      for d in dirs:
        ct = ct[d]
      current_target[i] = ct[target]
    score_dict[name] = jnp.mean(
        jnp.array([ops_dict[rule](ct) for ct in current_target])
    )
  return score_dict  # pytype: disable=bad-return-type  # jnp-type


def create_redist_dict(
    file_dir: str,
    idx: list[int],
    rule: str,
    running_average: bool,
    sketchy_rank: int,
    states: Optional[str] = None,
):
  """Create dictionary of reallocated memory to each layers."""
  if not states:
    states = create_state(file_dir, idx)
  sketches = states[-1]['inner_state']['0']['direction']['1']['sketches']
  layer_names, num_axes = layers_and_axes(sketches)
  group_dict = create_groups(sketches, layer_names)
  score_dict = score_fn(states, rule, layer_names, running_average)

  def create_redist():
    res = {}
    for p in list(score_dict):
      dirs = p.split('/')[:-2]
      cur = res
      for d in dirs[:-1]:
        cur = cur.setdefault(d, {})
      cur[dirs[-1]] = [0] * num_axes
    return res

  def alloc_fn(redist, group, realloc_dict):
    for key in group:
      dirs, axes_id = key.split('/')[:-2], int(key.split('/')[-1])
      carry = redist
      for d in dirs:
        carry = carry[d]
      carry[axes_id] = realloc_dict[key]
    return redist

  def rd(x):
    return int(x // 1) + 1

  def grp_info(dim):
    group = group_dict[dim]
    group_size = len(group)
    group_resource = group_size * sketchy_rank
    return group, group_size, group_resource

  def is_outlier(score, total_score, total_resource, dim):
    unit_rsc = total_resource / total_score if total_score else 0.0
    allocated_rsc = rd(score * unit_rsc) - 1
    return allocated_rsc > dim

  redist_dict = create_redist()

  for dim in group_dict:
    group, group_size, group_resource = grp_info(dim)
    assert group_resource >= group_size, (group_resource, group_size)
    group_resource -= group_size
    total_score = sum(score_dict[key] for key in group)
    sorted_scores = sorted(
        [(key, score_dict[key]) for key in group],
        key=lambda x: x[1],
        reverse=True,
    )
    realloc = {}
    for pair in sorted_scores:
      if is_outlier(pair[1], total_score, group_resource, dim - 1):
        realloc.update({pair[0]: dim})
        group_resource -= (dim - 1)
        total_score -= pair[1]
      else:
        unit_rsc = group_resource / total_score if total_score else 0.0
        realloc.update({pair[0]: rd(pair[1] * unit_rsc)})
        group_resource -= (rd(pair[1] * unit_rsc) - 1)
        total_score -= pair[1]

    for key in realloc:
      assert realloc[key] <= dim, (key, realloc[key], dim)

    allocated = sum(realloc.values())
    _, _, group_resource = grp_info(dim)
    assert allocated <= group_resource, (group_resource, allocated)

    if allocated < group_resource:
      extra = group_resource - allocated
      for (key, _) in sorted_scores:
        realloc[key] = min(realloc[key] + 1, dim)
        extra = extra - 1 if realloc[key] + 1 < dim else extra
        if extra <= 0:
          break

    redist_dict = alloc_fn(redist_dict, group, realloc)

  return redist_dict


_DIR = flags.DEFINE_string(
    'dir',
    '',
    'directory with checkpoints, must be set',
)

_IDX = flags.DEFINE_multi_integer(
    'idx',
    -1,
    'indices of checkpoints to anlayze, default last checkpoint'
)

_RULE = flags.DEFINE_string(
    'rule',
    'sketch_trace',
    'statistics to reallocate based on, default sketch trace',
)

_AVG = flags.DEFINE_bool(
    'avg',
    False,
    'whether to use running average of the statistics, default False',
)


_RANK = flags.DEFINE_integer(
    'rank',
    256,
    'rellocation base per-layer resource, default 256'
)


def _validate_flags():
  """Raise errors if flags are improperly set."""
  if not _DIR.value:
    raise ValueError('--dir must be set')
  return 0


def main(argv: ...):
  del argv
  _validate_flags()
  args = [_DIR.value, _IDX.value, _RULE.value, _AVG.value, _RANK.value]
  return create_redist_dict(*args)


if __name__ == '__main__':
  app.run(main)

