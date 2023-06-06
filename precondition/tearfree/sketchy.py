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

"""Sketchy low-rank second-order statistics preconditioning."""

import dataclasses
import functools
from typing import NamedTuple, Optional

import chex
import jax
from jax import numpy as jnp
import optax
from precondition.tearfree import praxis_shim


@dataclasses.dataclass
class Options:
  """Sketchy covariance approximation options.

  See https://arxiv.org/abs/2302.03764.

  Attributes:
    update_freq: Number of steps between covariance updates.
    second_moment_decay: Exponential moving average for second-order statistics
      tracking. If 1.0 then sums.
  """

  update_freq: int = 1
  second_moment_decay: float = 0.999


def apply(options: Options) -> praxis_shim.ShardedGradientTransformation:
  """Return gradient transform for (blocked) shampoo preconditioning."""

  _validate(options)

  return praxis_shim.ShardedGradientTransformation(
      functools.partial(_init, options),
      functools.partial(_update, options),
      functools.partial(_pspec, options),
  )


class _AxisState(NamedTuple):
  """Contains the covariance sketch state for each tensor dimension."""

  eigvecs: jax.Array
  eigvals: jax.Array
  inv_eigvals: jax.Array
  tail: jax.Array
  inv_tail: jax.Array


class _TensorState(NamedTuple):
  """Per-tensor state contains a list of axis states for each dimension."""

  axes: list[_AxisState]


class _SketchyState(NamedTuple):
  # A scalar int32 for step count.
  count: jax.Array
  # A tree of the same shape as the params of _TensorState leaves of f32.
  sketches: chex.ArrayTree


def _validate(options: Options) -> None:
  """Raise ValueError if options are invalid."""

  if options.update_freq <= 0:
    raise ValueError(
        "update_freq ({}) must be positive".format(options.update_freq)
    )

  if not (0 <= options.second_moment_decay <= 1):
    raise ValueError(
        f"second_moment_decay ({options.second_moment_decay}) "
        "should be in [0, 1]"
    )


def _init(options: Options, params: optax.Params) -> _SketchyState:
  """Inititialize stats to 0 and preconditioners to identity."""
  del options

  def _tensor_state(path: ..., param: jax.Array) -> _TensorState:
    if any(dim == 1 for dim in param.shape):
      raise ValueError(
          "param {} shape ({}) has unit dimensions".format(path, param.shape)
      )

    return _TensorState([])

  return _SketchyState(
      count=jnp.zeros([], jnp.int32),
      sketches=jax.tree_util.tree_map_with_path(_tensor_state, params),
  )


def _pspec(
    options: Options, params: praxis_shim.NestedHParams
) -> praxis_shim.NestedHParams:
  """Generate sharding specification for sketchy state."""
  del options

  count_pspec = praxis_shim.WeightHParams(
      shape=[],
      init=None,
      dtype=jnp.int32,
      collections=None,
      tensor_split_dims_mapping=[],
  )

  def _tensor_pspec(
      path: ...,
      param: praxis_shim.WeightHParams,
  ) -> praxis_shim.NestedHParams:
    del path, param
    return dict(axes=[])

  return dict(
      count=count_pspec,
      sketches=jax.tree_util.tree_map_with_path(
          _tensor_pspec, params, is_leaf=lambda x: hasattr(x, "shape")
      ),
  )


def _update(
    options: Options,
    updates: optax.Updates,
    state: _SketchyState,
    params: Optional[optax.Params] = None,
) -> tuple[optax.Updates, _SketchyState]:
  """Update internal shampoo stats and precondition gradients."""
  del params
  sketches = state.sketches
  is_tensor_state = lambda x: isinstance(x, _TensorState)

  updated_sketches = functools.partial(
      jax.tree_map,
      functools.partial(_update_sketches, options),
      updates,
      sketches,
      is_leaf=is_tensor_state,
  )
  should_update_stats = (state.count % options.update_freq) == 0
  state.sketches = jax.lax.cond(
      should_update_stats, updated_sketches, lambda: sketches
  )
  state.count += 1
  new_updates = jax.tree_map(
      _precondition, updates, state.sketches, is_leaf=is_tensor_state
  )
  return new_updates, state


def _update_sketches(
    options: Options,
    update: jax.Array,
    sketches: _TensorState,
) -> _TensorState:
  """Update sketched covariance statistics given a gradient."""

  del update, options

  def _update_axis(dim: int, axis_state: _AxisState) -> _AxisState:
    raise NotImplementedError

  new_axes = []
  for dim, axis_state in enumerate(sketches.axes):
    with jax.named_scope("UpdateSketchDim{}".format(dim)):
      new_axes.append(_update_axis(dim, axis_state))

  return _TensorState(new_axes)


def _precondition(update: jax.Array, sketches: _TensorState) -> jax.Array:
  """Precondition gradients."""

  def _precondition_axis(
      dim: int, axis_state: _AxisState, g: jax.Array
  ) -> jax.Array:
    raise NotImplementedError

  g = update

  for dim, axis_state in enumerate(sketches.axes):
    with jax.named_scope("SketchPreconditionDim{}".format(dim)):
      g = _precondition_axis(dim, axis_state, g)

  return g


def _ema_update(old: jax.Array, new: jax.Array, decay: float) -> jax.Array:
  if decay == 1.0:
    return old + new
  return old * decay + new * (1 - decay)
