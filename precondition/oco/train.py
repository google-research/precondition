# Copyright 2023 The precondition Authors.
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

"""Utilities for running an OCO algorithm on a dataset."""

import functools
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
from precondition.oco import algorithms
from precondition.oco import datasets

LossAndGrad = Callable[
    [jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array]
]


@functools.partial(
    jax.jit,
    static_argnames=[
        'loss_and_grad',
        'update_fn',
        # 'algorithm',
        # 'sketch_size',
        'extra_loss',
    ],
)
def _compiled_run_dataset(
    x: jax.Array,
    y: jax.Array,
    state: algorithms.State,
    obs_ixs: jax.Array,
    # delta: algorithms.RuntimeScalar,
    # lr: algorithms.RuntimeScalar,
    loss_and_grad: LossAndGrad,
    update_fn: algorithms.UpdateFn,
    # algorithm: algorithms.Algorithm,
    # sketch_size: int,
    extra_loss: Optional[datasets.Loss],
) -> algorithms.State:
  """Run an OCO algorithm, saving history at obs_ixs."""

  # hparams = algorithms.HParams(
  #    delta,
  #    lr,
  #    sketch_size,
  #    algorithm,
  # )

  # assume obs_ixs starts at 0 and ends at nrows-1
  # assume state has various keys pre-initialized, see below.

  def process_row(idx, state):  # fori_loop index, so pylint: disable=unused-argument
    ix = state['n']
    r = x[ix]
    f, g = loss_and_grad(state['w'], r, y[ix])
    if extra_loss is not None:
      state['extra_loss'] += extra_loss(state['w'], r, y[ix])
    state = update_fn(state, f, g)
    state['loss'] += f
    state['n'] += 1

    return state

  chunks = jnp.diff(obs_ixs, prepend=0)

  def scan_reduce(state, chunksize):
    state = jax.lax.fori_loop(0, chunksize, process_row, state)
    return state, state

  _, history = jax.lax.scan(scan_reduce, state, chunks)

  return history


def run_dataset(
    dataset_name: str,
    num_obs: int,
    hparams: algorithms.HParams,
    extra_loss: Optional[datasets.Loss] = None,
    dataset_cache: str = '/tmp/cache',
) -> algorithms.State:
  """Run an OCO algorithm on a dataset, saving history at obs_ixs."""
  assert num_obs >= 2

  dataset = datasets.load_dataset(dataset_name, dataset_cache)
  init_fn, update_fn = algorithms.generate_init_update(dataset.w_shape, hparams)

  obs_ixs = np.round(
      np.linspace(0, dataset.x.shape[0], num=num_obs, endpoint=True)
  ).astype(int)

  initial_state = init_fn()
  loss_and_grad = jax.value_and_grad(dataset.loss)

  assert 'loss' not in initial_state, list(initial_state)
  assert 'w' in initial_state, list(initial_state)
  assert 'n' not in initial_state, list(initial_state)
  initial_state['loss'] = jnp.array(0.0, dtype=jnp.float64)
  initial_state['n'] = 0
  if extra_loss is not None:
    initial_state['extra_loss'] = jnp.array(0.0, dtype=jnp.float64)

  # Unpack the compilable and static HParams separately.
  return _compiled_run_dataset(
      dataset.x,
      dataset.y,
      initial_state,
      obs_ixs,
      # hparams.delta,
      # hparams.lr,
      loss_and_grad,
      update_fn,
      # hparams.algorithm,
      # hparams.sketch_size,
      extra_loss,
  )
