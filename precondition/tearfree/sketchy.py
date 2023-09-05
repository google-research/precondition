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
from typing import Any, NamedTuple, Optional, Union

from absl import logging
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
    memory_alloc: optional dictionary to indicate reallocation of memory used
    in sketching the covariance matrix.
    ekfac_svd: whether to use the ekfac_svd precondtioner instead of Sketchy,
    default setting to FALSE.
    linear_approx_tail: whether to use the approximately linear relationship
    between log(eigval) and log(eigval_rank) to calculate tail
  """

  epsilon: float = 1e-7
  rank: int = 128
  relative_epsilon: bool = True
  second_moment_decay: float = 0.999
  update_freq: int = 1
  add_ggt: bool = False
  memory_alloc: Optional[dict[str, Any]] = None
  ekfac_svd: bool = False
  linear_approx_tail: bool = False


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
  # Save svd result to perform ekfac preconditioning
  svd_result_u: Union[optax.MaskedNode, jax.Array]
  # Save svd result to perform ekfac preconditioning, this corresponeds to the
  # inv_eigvals using Sketchy
  svd_result_s: Union[optax.MaskedNode, jax.Array]
  # Analogous to inv_tail in the Sketchy case, but up to time t-1
  inv_prev_tail: Union[optax.MaskedNode, jax.Array]


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


def _path_to_key(path: ...) -> str:
  concat_path = ""
  for dickey in path:
    if hasattr(dickey, "key"):
      concat_path = concat_path + "/" + dickey.key
    elif hasattr(dickey, "idx"):
      concat_path = concat_path + "/" + str(dickey.idx)
    else:
      raise ValueError("no key or idx found")
  return concat_path


def _locate_path(
    path: ..., dictionary: dict[str, Any]
) -> Union[dict[str, Any], list[int]]:

  """Locate a path in a dictionary."""

  carry = dictionary
  for p in path:
    if not hasattr(p, "key") and not hasattr(p, "idx"):
      raise ValueError("no key or idx found")
    carry = (
        carry[p.key] if hasattr(p, "key") else carry[p.idx]
    )
  assert isinstance(carry, list), type(carry)
  return carry


def _init(options: Options, params: optax.Params) -> _SketchyState:
  """Inititialize sketch."""

  def _tensor_state(path: ..., param: jax.Array) -> _TensorState:
    axes = []
    axes_idx = 0
    add_ggt = options.add_ggt
    memory_alloc = options.memory_alloc
    ekfac = options.ekfac_svd
    total_dim_prod = jnp.prod(jnp.array(param.shape))
    for d in param.shape:
      if d == 1:
        raise ValueError(
            "param {} shape ({}) has unit dimensions".format(path, param.shape)
        )
      if memory_alloc:
        k = min(d, _locate_path(path, memory_alloc)[axes_idx])
        logging.info("custom rank: %d rank allocated to %s",
                     k, _path_to_key(path))
      else:
        logging.warning(
            "memory_alloc not found for path %s, using global rank %d",
            _path_to_key(path),
            options.rank,
        )
        k = min(d, options.rank)
      others_dim_prod = int(total_dim_prod / d) if d else 0
      m = min(d, k + others_dim_prod)
      axes_idx += 1

      axes.append(
          _AxisState(
              eigvecs=jnp.zeros((d, k)),
              eigvals=jnp.zeros((k,)),
              inv_eigvals=jnp.zeros((k,)),
              tail=jnp.zeros(tuple()),
              inv_tail=jnp.zeros(tuple()),
              ema_ggt=jnp.zeros((d, d)) if add_ggt else optax.MaskedNode(),
              svd_result_u=jnp.zeros((d, m)) if ekfac else optax.MaskedNode(),
              svd_result_s=jnp.zeros((m,)) if ekfac else optax.MaskedNode(),
              inv_prev_tail=jnp.zeros(tuple()) if ekfac else optax.MaskedNode(),
          )
      )
    return _TensorState(axes)
  return _SketchyState(
      count=jnp.zeros([], jnp.int32),
      sketches=jax.tree_util.tree_map_with_path(_tensor_state, params)
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
      path: ...,
      param: praxis_shim.WeightHParams,
  ) -> praxis_shim.NestedHParams:

    total_dim_prod = jnp.prod(jnp.array(param.shape))

    def _replicated(shape):
      return praxis_shim.WeightHParams(
          shape=list(shape),
          init=None,
          dtype=jnp.float32,
          collections=None,
          tensor_split_dims_mapping=[-1] * len(shape),
      )

    def _make_axis_state(d: int, axes_idx: int):
      memory_alloc = options.memory_alloc
      ekfac = options.ekfac_svd
      if memory_alloc:
        k = min(d, _locate_path(path, memory_alloc)[axes_idx])
      else:
        k = min(d, options.rank)
      add_ggt = options.add_ggt
      others_dim_prod = int(total_dim_prod / d) if d else 0
      m = min(d, k + others_dim_prod)
      axes_idx += 1
      return dict(
          eigvecs=_replicated((d, k)),
          eigvals=_replicated((k,)),
          inv_eigvals=_replicated((k,)),
          tail=_replicated(tuple()),
          inv_tail=_replicated(tuple()),
          ema_ggt=_replicated((d, d)) if add_ggt else optax.MaskedNode(),
          svd_result_u=_replicated((d, m)) if ekfac else optax.MaskedNode(),
          svd_result_s=_replicated((m,)) if ekfac else optax.MaskedNode(),
          inv_prev_tail=_replicated(tuple()) if ekfac else optax.MaskedNode(),
      )

    return dict(
        axes=[
            _make_axis_state(d, axes_idx)
            for d, axes_idx in zip(param.shape, range(len(param.shape)))
        ]
    )
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

  should_update_stats = (state.count % options.update_freq) == 0
  updated_sketches = functools.partial(
      jax.tree_util.tree_map_with_path,
      functools.partial(_update_sketches, options),
      updates,
      sketches,
      is_leaf=is_tensor_state,
  )

  if not options.ekfac_svd:
    new_sketches = jax.lax.cond(
        should_update_stats, updated_sketches, lambda: sketches
    )

  else:
    # when using ekfac_svd, need to call updated_sketches() every iteration
    # since the preconditioner needs to be updated every iteration
    # even when the sketches are updated every update_freq iterations
    updated_preconditioner_only = functools.partial(
        jax.tree_util.tree_map_with_path,
        lambda p, u, s: _update_sketches(options, p, u, s, False),
        updates,
        sketches,
        is_leaf=is_tensor_state,
    )
    new_sketches = jax.lax.cond(
        should_update_stats, updated_sketches, updated_preconditioner_only
    )

  new_updates = jax.tree_util.tree_map_with_path(
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
    path: ...,
    update: jax.Array,
    sketches: _TensorState,
    update_sketches: bool = True,
) -> _TensorState:
  """Update sketched covariance statistics given a gradient."""
  new_axes = []
  for dim, axis_state in enumerate(sketches.axes):
    with jax.named_scope("UpdateSketchDim{}".format(dim)):
      new_axes.append(
          _update_axis(options, dim, path, update, axis_state, update_sketches)
      )
  return _TensorState(new_axes)


def _precondition(
    options: Options,
    path: ...,
    update: jax.Array,
    sketches: _TensorState,
) -> jax.Array:
  """Precondition gradients."""
  g = update
  original_shape = g.shape
  roll = tuple(range(1, g.ndim)) + (0,)
  memory_alloc = options.memory_alloc
  ekfac = options.ekfac_svd
  for dim, axis_state in enumerate(sketches.axes):
    with jax.named_scope("SketchPreconditionDim{}".format(dim)):
      # Rotate g during this loop; keep the axis to precondition first.
      d = original_shape[dim]
      assert g.shape[0] == d
      if memory_alloc:
        k = min(d, _locate_path(path, memory_alloc)[dim])
      else:
        k = min(d, options.rank)
      assert list(axis_state.eigvecs.shape) == [d, k]
      eigvecs = axis_state.eigvecs if not ekfac else axis_state.svd_result_u
      lowrank_basis = jnp.tensordot(g, eigvecs, axes=[[0], [0]])
      lowrank_component = jnp.tensordot(
          lowrank_basis, eigvecs, axes=[[g.ndim - 1], [1]]
      )
      g = jnp.transpose(g, axes=roll)
      complement = g - lowrank_component
      inv_eigvals = (
          axis_state.inv_eigvals if not ekfac else axis_state.svd_result_s
      )
      scaled_basis = lowrank_basis * inv_eigvals
      scaled_lowrank_component = jnp.tensordot(
          scaled_basis, eigvecs, axes=[[g.ndim - 1], [1]]
      )
      g = scaled_lowrank_component
      inv_tail = axis_state.inv_tail if not ekfac else axis_state.inv_prev_tail
      g += inv_tail * complement
  return g


# pylint: disable = g-long-lambda
def _update_axis(
    options: Options, dim: int, path: ..., update: jax.Array,
    axis_state: _AxisState, update_sketches: bool = True,
) -> _AxisState:
  """Perform an FD update for statistics."""
  # _low_rank_root
  d = update.shape[dim]
  memory_alloc = options.memory_alloc
  if memory_alloc:
    k = min(d, _locate_path(path, memory_alloc)[dim])
  else:
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
  if options.linear_approx_tail and d > k:
    num_points = (k + 1) // 2
    assert num_points > 0
    ranks = jnp.arange(1, num_points + 1)
    vals = axis_state.eigvals[:num_points]
    assert ranks.shape == vals.shape
    sample_cov = jnp.cov(ranks, vals)
    s_x, s_xy = sample_cov[0, 0], sample_cov[0, 1]
    slope = jax.lax.cond(s_x > 0, lambda: s_xy / (s_x ** 2), lambda: 0.0)
    intercept = jnp.mean(vals) - slope * jnp.mean(ranks)
    log_ranks = jnp.log(jnp.arange(k + 1, d + 1))
    fitted_vals = slope * log_ranks + intercept
    tail = jnp.exp(jax.scipy.special.logsumexp(fitted_vals * 2)) / (d - k)
    undeflated = jnp.square(jnp.maximum(top_eigs, 0.0))
  else:
    tail = axis_state.tail * decay + cutoff**2
    # Avoid numerical error from the sqrt computation and from subtracting
    # and re-adding cutoff^2 (mathematically, undeflated == deflated^2 + tail).
    undeflated = (
        jnp.square(jnp.maximum(top_eigs, 0.0)) + axis_state.tail * decay
    )
  eigvecs = u[:, :k]

  mask = deflated > 0

  alpha = jnp.asarray(-1.0 / (2 * update.ndim), dtype=jnp.float32)
  eigvecs *= mask
  if options.relative_epsilon and options.epsilon > 0:
    eps = jnp.max(undeflated) * options.epsilon
  else:
    eps = options.epsilon
  inv_eigvals = jnp.where(mask, (undeflated + eps) ** alpha, 0.0)
  eigvals = deflated * mask
  inv_tail = jnp.where(tail > 0, (tail + eps) ** alpha, 0.0)

  if options.add_ggt:
    ema_ggt = axis_state.ema_ggt * decay + g_dm.dot(g_dm.T) * (1 - decay)
  else:
    ema_ggt = axis_state.ema_ggt

  if options.ekfac_svd:
    assert u.shape[1] <= d
    prev_tail = axis_state.tail
    undeflated_ekfac = jnp.square(jnp.maximum(s, 0.0)) + prev_tail * decay
    svd_result_u = u
    svd_result_s = jnp.where(
        undeflated_ekfac > 0, (undeflated_ekfac + eps) ** alpha, 0.0
    )
    inv_prev_tail = axis_state.inv_tail
  else:
    svd_result_u = axis_state.svd_result_u
    svd_result_s = axis_state.svd_result_s
    inv_prev_tail = axis_state.inv_prev_tail

  res = _AxisState(
      eigvecs,
      eigvals,
      inv_eigvals,
      tail,
      inv_tail,
      ema_ggt,
      svd_result_u,
      svd_result_s,
      inv_prev_tail,
  )

  return jax.lax.cond(
      update_sketches,
      lambda: res,
      lambda: res._replace(
          eigvecs=axis_state.eigvecs,
          eigvals=axis_state.eigvals,
          inv_eigvals=axis_state.inv_eigvals,
          tail=axis_state.tail,
          inv_tail=axis_state.inv_tail,
      )
  )
