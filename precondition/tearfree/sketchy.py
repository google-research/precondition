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
from typing import NamedTuple, Optional, Union

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
    epsilon: The diagonal positive perturbation added to preconditioner before
      inversion.
    rank: The sketch size to use for FD sketch for each tensor's dimension.
    relative_epsilon: Whether to scale epsilon by the top singular value of the
      covariance or not.
    second_moment_decay: Exponential moving average for second-order statistics
      tracking. If 1.0 then sums.
    update_freq: Number of steps between covariance updates.
    add_ggt: whether to store the exponentially moving GGT in the states. Set
    to TRUE to save the exponentially moving GGT.
  """

  epsilon: float = 1e-7
  rank: int = 128
  relative_epsilon: bool = True
  second_moment_decay: float = 0.999
  update_freq: int = 1
  add_ggt: bool = False


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
  # Add additional optional state to store ema GGT
  ema_ggt: Union[optax.MaskedNode, jax.Array]


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
    add_ggt = options.add_ggt
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
              ema_ggt=jnp.zeros((d, d)) if add_ggt else optax.MaskedNode(),
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
      add_ggt = options.add_ggt
      return dict(
          eigvecs=_replicated((d, k)),
          eigvals=_replicated((k,)),
          inv_eigvals=_replicated((k,)),
          tail=_replicated(tuple()),
          inv_tail=_replicated(tuple()),
          ema_ggt=_replicated((d, d)) if add_ggt else optax.MaskedNode(),
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
    options: Options, dim: int, update: jax.Array, axis_state: _AxisState,
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
  # This dimensionality reduction with QR is a mathematical no-op but required
  # to avoid b/286607548.
  updated = jnp.linalg.qr(updated.T, mode="r").T

  def _safe_svd(x):
    # Wrap with a nan check due to hangs per b/286654608.
    svd = lambda y: jnp.linalg.svd(y, full_matrices=False)[:2]

    def _all_nan(y):
      m = min(y.shape)
      u = jnp.full((d, m), jnp.nan, jnp.float32)
      s = jnp.full((m,), jnp.nan, jnp.float32)
      return u, s

    return jax.lax.cond(jnp.isfinite(x).all(), svd, _all_nan, x)

  u, s = _safe_svd(updated)
  assert u.shape[0] == d
  assert u.shape[1] >= k

  cutoff = jnp.maximum(s[k], 0.0) if k < len(s) else 0.0
  top_eigs = jnp.maximum(s[:k], 0.0)
  deflated = jnp.sqrt(jnp.maximum(0.0, top_eigs - cutoff)) * jnp.sqrt(
      top_eigs + cutoff
  )
  tail = axis_state.tail * decay + cutoff**2
  # Avoid numerical error from the sqrt computation and from subtracting
  # and re-adding cutoff^2 (mathematically, undeflated == deflated^2 + tail).
  undeflated = jnp.square(jnp.maximum(top_eigs, 0.0)) + axis_state.tail * decay
  eigvecs = u[:, :k]

  mask = deflated > 0
  # Would be nice to statically assert deflated == 0 implies undeflated == 0.

  alpha = jnp.asarray(-1.0 / (2 * update.ndim), dtype=jnp.float32)
  eigvecs *= mask
  if options.relative_epsilon and options.epsilon > 0:
    eps = jnp.max(undeflated) * options.epsilon
  else:
    eps = options.epsilon
  inv_eigvals = jnp.where(mask, (undeflated + eps) ** alpha, 0)
  eigvals = deflated * mask
  inv_tail = jnp.where(tail > 0, (tail + eps) ** alpha, 0.0)
  if options.add_ggt:
    ema_ggt = axis_state.ema_ggt * decay + g_dm.dot(g_dm.T) * (1 - decay)
  else:
    ema_ggt = axis_state.ema_ggt

  return _AxisState(eigvecs, eigvals, inv_eigvals, tail, inv_tail, ema_ggt)
