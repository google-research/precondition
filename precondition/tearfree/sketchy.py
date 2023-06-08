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
    rank: The sketch size to use for FD sketch for each tensor's dimension.
    truncate_numerical_noise: Round all singular values less than `1e-6 *
      max_singular_value` down to 0.0 after `svd`.
  """

  update_freq: int = 1
  second_moment_decay: float = 0.999
  rank: int = 128
  truncate_numerical_noise: bool = True


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
  # These refer to the eigenvalues of the running *square root* of of the
  # covariance.
  eigvals: jax.Array
  # Analogously, but the -(1/(2*ndim)) root of the covariance, where ndim
  # is total tensor rank.
  inv_eigvals: jax.Array
  # The tail, however, tracks the cumulative escaped mass, which is the sum
  # of eigenvalues of the full gradient covariance which were subtracted out.
  tail: jax.Array
  # Analogously to inv_eigvals, the -(1/(2*ndim))-th root.
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

  if options.rank <= 0:
    raise ValueError(f"rank ({options.rank}) must be at least 1")


def _init(options: Options, params: optax.Params) -> _SketchyState:
  """Inititialize sketch."""
  def _tensor_state(path: ..., param: jax.Array) -> _TensorState:
    axes = []

    for d in param.shape:
      if d == 1:
        raise ValueError(
            "param {} shape ({}) has unit dimensions".format(path, param.shape)
        )

      k = min(d, options.rank)

      axes.append(
          _AxisState(
              eigvecs=jnp.zeros((d, k)),
              eigvals=jnp.zeros((k,)),
              inv_eigvals=jnp.zeros((k,)),
              tail=jnp.zeros(tuple()),
              inv_tail=jnp.zeros(tuple()),
          )
      )

    return _TensorState(axes)

  return _SketchyState(
      count=jnp.zeros([], jnp.int32),
      sketches=jax.tree_util.tree_map_with_path(_tensor_state, params),
  )


def _pspec(
    options: Options, params: praxis_shim.NestedHParams
) -> praxis_shim.NestedHParams:
  """Generate sharding specification for sketchy state."""

  count_pspec = praxis_shim.WeightHParams(
      shape=[],
      init=None,
      dtype=jnp.int32,
      collections=None,
      tensor_split_dims_mapping=[],
  )

  def _tensor_pspec(
      param: praxis_shim.WeightHParams,
  ) -> praxis_shim.NestedHParams:

    def _replicated(shape):
      return praxis_shim.WeightHParams(
          shape=list(shape),
          init=None,
          dtype=jnp.float32,
          collections=None,
          tensor_split_dims_mapping=[-1] * len(shape),
      )

    def _make_axis_state(d: int):
      k = min(d, options.rank)
      return dict(
          eigvecs=_replicated((d, k)),
          eigvals=_replicated((k,)),
          inv_eigvals=_replicated((k,)),
          tail=_replicated(tuple()),
          inv_tail=_replicated(tuple()),
      )

    return dict(axes=[_make_axis_state(d) for d in param.shape])

  return dict(
      count=count_pspec,
      sketches=jax.tree_util.tree_map(
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
  new_sketches = jax.lax.cond(
      should_update_stats, updated_sketches, lambda: sketches
  )
  new_updates = jax.tree_map(
      functools.partial(_precondition, options),
      updates,
      new_sketches,
      is_leaf=is_tensor_state,
  )
  return new_updates, _SketchyState(
      count=state.count + 1, sketches=new_sketches
  )


def _update_sketches(
    options: Options,
    update: jax.Array,
    sketches: _TensorState,
) -> _TensorState:
  """Update sketched covariance statistics given a gradient."""
  new_axes = []
  for dim, axis_state in enumerate(sketches.axes):
    with jax.named_scope("UpdateSketchDim{}".format(dim)):
      new_axes.append(_update_axis(options, dim, update, axis_state))
  return _TensorState(new_axes)


def _precondition(
    options: Options, update: jax.Array, sketches: _TensorState
) -> jax.Array:
  """Precondition gradients."""
  g = update
  original_shape = g.shape
  roll = tuple(range(1, g.ndim)) + (0,)
  for dim, axis_state in enumerate(sketches.axes):
    with jax.named_scope("SketchPreconditionDim{}".format(dim)):
      # Rotate g during this loop; keep the axis to precondition first.
      d = original_shape[dim]
      k = min(d, options.rank)
      assert g.shape[0] == d
      eigvecs = axis_state.eigvecs
      assert list(eigvecs.shape) == [d, k]
      lowrank_basis = jnp.tensordot(g, eigvecs, axes=[[0], [0]])
      lowrank_component = jnp.tensordot(
          lowrank_basis, eigvecs, axes=[[g.ndim - 1], [1]]
      )
      g = jnp.transpose(g, axes=roll)
      complement = g - lowrank_component
      scaled_basis = lowrank_basis * axis_state.inv_eigvals
      scaled_lowrank_component = jnp.tensordot(
          scaled_basis, eigvecs, axes=[[g.ndim - 1], [1]]
      )
      g = axis_state.inv_tail * complement + scaled_lowrank_component
  return g


def _update_axis(
    options: Options, dim: int, update: jax.Array, axis_state: _AxisState
) -> _AxisState:
  """Perform an FD update for statistics."""
  # _low_rank_root
  d = update.shape[dim]
  k = min(d, options.rank)

  sketch_dk = axis_state.eigvecs
  assert sketch_dk.shape == (d, k), (sketch_dk.shape, d, k, update.shape, dim)

  sketch_dk *= axis_state.eigvals[jnp.newaxis, :]
  all_but_dim = [i for i in range(update.ndim) if i != dim]
  g_dm = update.transpose([dim] + all_but_dim).reshape(d, -1)
  decay = jnp.sqrt(options.second_moment_decay)

  # This implementation uses only O(|gradient size|) memory because
  # full_matrices is False, but may be slow. Consider LOBPCG instead.
  updated = jnp.concatenate([sketch_dk * decay, g_dm], axis=1)
  u, s, vt = jnp.linalg.svd(updated, full_matrices=False)
  del vt
  assert u.shape[0] == d
  assert u.shape[1] >= k

  if options.truncate_numerical_noise:
    eps = 1e-6
    mask = s > eps * jnp.max(s)
    s *= mask

  cutoff = jnp.maximum(s[k], 0.0) if k < len(s) else 0.0
  top_eigs = s[:k]
  deflated = jnp.sqrt(jnp.maximum(0.0, top_eigs - cutoff)) * jnp.sqrt(
      top_eigs + cutoff
  )
  tail = axis_state.tail * decay + cutoff**2
  eigvecs = u[:, :k]
  mask = deflated > 0

  alpha = jnp.asarray(-1.0 / (2 * update.ndim), dtype=jnp.float32)
  eigvecs *= mask
  # Note that deflated already contains the square root.
  inv_eigvals = jnp.where(mask, (deflated**2 + tail) ** alpha, 0)
  eigvals = deflated * mask
  inv_tail = jnp.where(tail > 0, tail**alpha, 0.0)

  return _AxisState(eigvecs, eigvals, inv_eigvals, tail, inv_tail)
