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

"""Dynamic Sketchy: Sketchy w/ online rank reallocation grouped by dimension."""
import dataclasses
import functools
from typing import Any, DefaultDict, NamedTuple, Optional

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
    delta: BCO exploration parameter.
    eta: BCO step size.
    err_tol: BCO rounding error tolerance.
    seed: random seed used in BCO exploration.
  """

  epsilon: float = 1e-7
  rank: int = 128
  relative_epsilon: bool = True
  second_moment_decay: float = 0.999
  update_freq: int = 1
  delta: float = 1e-5
  eta: float = 1e-5
  err_tol: float = 1e-6
  seed: int = 0


def apply(options: Options) -> praxis_shim.ShardedGradientTransformation:
  """Return gradient transform for (blocked) shampoo preconditioning."""

  _validate(options)

  return praxis_shim.ShardedGradientTransformation(
      functools.partial(_init, options),
      functools.partial(_update, options),
      functools.partial(_pspec, options),
  )


class _GroupState(NamedTuple):
  """Contains the covariance sketch state for each dimension group."""

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
  # Allocation vector, projected, float32
  memory_x: jax.Array
  # Allocation vector, actually played, rounded, int32
  memory_pi: jax.Array


class _GroupMaps(NamedTuple):
  """Group maps for each dimension group.

  Online rank reallocation is performed within each group.

  Attributes:
    layer2dimidx: A dictionary that maps each (path, axis) pair to dimidx,
    where dimidx is a jnp.int32, representing the position of the axis in the
    dimension group.
    dim2layers: A dictionary that maps each dimension to a list of (path, axis)
    pairs with that dimension.
  """

  layer2dimidx: dict[tuple[Any, int], int]
  dim2layers: dict[int, list[tuple[Any, int]]]


class _GroupedUpdates(NamedTuple):
  """Group updates by dimension."""

  # dimension of the group
  dim: int
  # updates of all axes in the group
  updates: list[jax.Array]
  # number of axes of the tensor each axis belongs to in the group
  ndims: list[int]


class _DynamicSketchyState(NamedTuple):
  # A scalar int32 for step count.
  count: jax.Array
  # A dictionary that stores group states for each group.
  group_states: dict[int, _GroupState]


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


def _create_groups(params: optax.Params) -> _GroupMaps:
  """Generate group maps."""

  layer2dimidx = DefaultDict(list)
  dim2layers = DefaultDict(list)

  def _create_group_maps(
      path: ...,
      param: jax.Array,
  ):
    for axis, dim in enumerate(param.shape):
      if dim == 1:
        raise ValueError(
            "param {} shape ({}) has unit dimensions".format(path, param.shape)
        )
      cur_count = len(dim2layers.get(dim, []))
      layer2dimidx[(path, axis)] = cur_count
      dim2layers[dim].append((path, axis))
    return None

  _ = jax.tree_util.tree_map_with_path(_create_group_maps, params)
  group_maps = _GroupMaps(layer2dimidx, dim2layers)
  return group_maps


def _group_updates(
    updates: optax.Updates,
    layer2dimidx: dict[tuple[Any, int], int],
) -> dict[int, _GroupedUpdates]:
  """Group updates by dim, ordered by count."""

  grouped = DefaultDict(dict)

  def _group_update(
      path: ...,
      update: jax.Array,
      grouped: dict[int, dict[int, tuple[jax.Array, int]]]
  ):
    for axis, d in enumerate(update.shape):
      idx = layer2dimidx[(path, axis)]
      all_but_axis = [i for i in range(update.ndim) if i != axis]
      g_dm = update.transpose([axis] + all_but_axis).reshape(d, -1)
      grouped[d][idx] = (g_dm, update.ndim)
    return grouped

  jax.tree_util.tree_map_with_path(
      lambda path, update: _group_update(path, update, grouped), updates
  )

  res = {}
  for d, updates_dict in grouped.items():
    sorted_keys = sorted(updates_dict.keys())
    updates = [updates_dict[key][0] for key in sorted_keys]
    ndims = [updates_dict[key][1] for key in sorted_keys]
    res[d] = _GroupedUpdates(d, updates, ndims)

  return res


def _init(options: Options, params: optax.Params) -> _DynamicSketchyState:
  """Inititialize sketch."""

  group_maps = _create_groups(params)
  dim2layers = group_maps.dim2layers

  def _create_group_states(
      dim2layers: dict[int, list[tuple[Any, int]]],
  ) -> dict[int, _GroupState]:
    """Create state for each group."""

    res = {}
    for d in dim2layers:
      k = min(d, options.rank)
      n = len(dim2layers[d])
      cols = k * n
      res[d] = _GroupState(
          eigvecs=jnp.zeros((d, cols)),
          eigvals=jnp.zeros((cols,)),
          inv_eigvals=jnp.zeros((cols,)),
          tail=jnp.zeros((n,)),
          inv_tail=jnp.zeros((n,)),
          memory_x=jnp.ones((n,))*k,
          memory_pi=jnp.ones((n,), dtype=jnp.int32)*k,
      )
    return res

  group_states = _create_group_states(dim2layers)

  return _DynamicSketchyState(
      count=jnp.zeros([], jnp.int32),
      group_states=group_states,
  )


def _pspec(
    options: Options, params: praxis_shim.NestedHParams
) -> praxis_shim.NestedHParams:
  """Generate sharding specification for sketchy state."""

  def _replicated(shape, d_type=jnp.float32):
    return praxis_shim.WeightHParams(
        shape=list(shape),
        init=None,
        dtype=d_type,
        collections=None,
        tensor_split_dims_mapping=[-1] * len(shape),
    )

  count_pspec = _replicated((), jnp.int32)

  group_maps = _create_groups(params)
  dim2layers = group_maps.dim2layers

  def _group_state_pspec() -> praxis_shim.NestedHParams:

    def _make_group_state(d: int):
      k = min(d, options.rank)
      n = len(dim2layers[d])
      cols = k * n
      return dict(
          eigvecs=_replicated((d, cols)),
          eigvals=_replicated((cols,)),
          inv_eigvals=_replicated((cols,)),
          tail=_replicated((n,)),
          inv_tail=_replicated((n,)),
          memory_x=_replicated((n,)),
          memory_pi=_replicated((n,), jnp.int32),
      )

    return dict(axes=[_make_group_state(d) for d in dim2layers])

  return dict(
      count=count_pspec,
      group_state=_group_state_pspec(),
  )


def _update(
    options: Options,
    updates: optax.Updates,
    state: _DynamicSketchyState,
    params: Optional[optax.Params] = None,
) -> tuple[optax.Updates, _DynamicSketchyState]:
  """Update internal shampoo stats and precondition gradients."""

  del params
  group_maps = _create_groups(updates)
  grouped_updates = _group_updates(updates, group_maps.layer2dimidx)

  should_update_stats = (state.count % options.update_freq) == 0

  key = jax.random.fold_in(jax.random.PRNGKey(options.seed), state.count)

  def updated_group_states(_):
    return jax.tree_util.tree_map(
        lambda u: _update_group_state(
            options=options, key=key, grouped_updates=u, state=state,
            group_maps=group_maps,
        ),
        grouped_updates,
        is_leaf=lambda x: isinstance(x, _GroupedUpdates),
    )

  new_group_states = jax.lax.cond(
      should_update_stats,
      updated_group_states,
      lambda _: state.group_states,
      None,
  )

  new_updates = jax.tree_util.tree_map_with_path(
      functools.partial(
          _precondition,
          group_states=new_group_states,
          layer2dimidx=group_maps.layer2dimidx,
      ),
      updates,
  )

  return new_updates, _DynamicSketchyState(
      count=state.count + 1,
      group_states=new_group_states,
  )


def _precondition(
    path: ...,
    update: jax.Array,
    group_states: dict[int, _GroupState],
    layer2dimidx: dict[tuple[Any, int], list[int]],
) -> jax.Array:
  """Precondition gradients."""

  g = update
  original_shape = g.shape
  roll = tuple(range(1, g.ndim)) + (0,)

  for axis in range(g.ndim):
    with jax.named_scope("SketchPreconditionDim{}".format(axis)):
      # Rotate g during this loop; keep the axis to precondition first.
      d = original_shape[axis]
      assert g.shape[0] == d
      idx = layer2dimidx[(path, axis)]
      budget = group_states[d].memory_pi[idx]
      start_id = jnp.sum(group_states[d].memory_pi[:idx])
      group_eigvecs = group_states[d].eigvecs
      group_eigvecs = jnp.hstack((group_eigvecs, jnp.zeros((d, d-1))))
      eigvecs = jax.lax.dynamic_slice_in_dim(
          group_eigvecs, start_id, d, axis=1
      )
      mask = jnp.broadcast_to(jnp.arange(d) < budget, eigvecs.shape)
      eigvecs = eigvecs * mask
      lowrank_basis = jnp.tensordot(g, eigvecs, axes=[[0], [0]])
      lowrank_component = jnp.tensordot(
          lowrank_basis, eigvecs, axes=[[g.ndim - 1], [1]]
      )
      g = jnp.transpose(g, axes=roll)
      complement = g - lowrank_component
      group_inv_eigvals = group_states[d].inv_eigvals
      group_inv_eigvals = jnp.hstack((group_inv_eigvals, jnp.zeros((d-1,))))
      inv_eigvals = jax.lax.dynamic_slice_in_dim(
          group_inv_eigvals, start_id, d
      )
      inv_eigvals *= jnp.arange(d) < budget
      scaled_basis = lowrank_basis * inv_eigvals
      scaled_lowrank_component = jnp.tensordot(
          scaled_basis, eigvecs, axes=[[g.ndim - 1], [1]]
      )
      g = scaled_lowrank_component
      inv_tail = group_states[d].inv_tail[idx]
      g += inv_tail * complement
  return g


def _update_group_state(
    options: Options,
    key: jax.Array,
    grouped_updates: _GroupedUpdates,
    state: _DynamicSketchyState,
    group_maps: _GroupMaps,
) -> _GroupState:
  """Perform FD statistics update for a dimension group."""

  d = grouped_updates.dim
  prev_group_state = state.group_states[d]
  prev_group_eigvecs = prev_group_state.eigvecs
  prev_group_eigvecs = jnp.hstack((prev_group_eigvecs, jnp.zeros((d, d-1))))
  prev_group_eigvals = prev_group_state.eigvals
  prev_group_eigvals = jnp.hstack((prev_group_eigvals, jnp.zeros((d-1,))))
  prev_group_tail = prev_group_state.tail
  prev_memory_x = prev_group_state.memory_x
  prev_memory_pi = prev_group_state.memory_pi

  k = min(d, options.rank)
  layers = group_maps.dim2layers[d]
  n = len(layers)

  # BCO step to update memory allocation vector
  pi, new_x = _one_step_fkm(
      options, key, prev_memory_x, prev_group_tail, (d, k, n)
  )

  group_state = _GroupState(
      eigvecs=jnp.zeros((d, k*n)),
      eigvals=jnp.zeros((k*n,)),
      inv_eigvals=jnp.zeros((k*n,)),
      tail=jnp.zeros((n,)),
      inv_tail=jnp.zeros((n,)),
      memory_x=new_x,
      memory_pi=pi,
  )

  # python loop since g_dms have different shapes (m) in the same group
  for idx in range(n):
    mask = (jnp.arange(n) < idx)
    start_id = jnp.sum(prev_memory_pi * mask)
    prev_budget = prev_memory_pi[idx]
    new_start = jnp.sum(pi * mask)
    new_budget = pi[idx]

    eigvecs = jax.lax.dynamic_slice_in_dim(
        prev_group_eigvecs, start_id, d, axis=1
    )
    eigvals = jax.lax.dynamic_slice_in_dim(
        prev_group_eigvals, start_id, d
    )
    eigvals *= jnp.arange(d) < prev_budget
    sketch = eigvecs * eigvals[jnp.newaxis, :]
    g_dm, ndim = grouped_updates.updates[idx], grouped_updates.ndims[idx]
    decay = jnp.sqrt(options.second_moment_decay)

    updated = jnp.concatenate([sketch * decay, g_dm], axis=1)
    updated = jnp.linalg.qr(updated.T, mode="r").T

    def _safe_svd(x):
      svd = lambda y: jnp.linalg.svd(y, full_matrices=False)[:2]

      def _all_nan(y):
        m = min(y.shape)
        u = jnp.full((d, m), jnp.nan, jnp.float32)
        s = jnp.full((m,), jnp.nan, jnp.float32)
        return u, s

      return jax.lax.cond(jnp.isfinite(x).all(), svd, _all_nan, x)

    u, s = _safe_svd(updated)
    assert u.shape[0] == d

    s_full = jnp.zeros((d+1,))
    s_full = jax.lax.dynamic_update_slice(s_full, s, (0,))
    cutoff = jnp.maximum(s_full[new_budget], 0.0)

    top_eigs = jnp.maximum(s*(jnp.arange(d) < new_budget), 0.0)
    deflated = jnp.sqrt(jnp.maximum(0.0, top_eigs - cutoff)) * jnp.sqrt(
        top_eigs + cutoff
    )
    tail = prev_group_tail[idx] * decay + cutoff**2

    undeflated = (
        jnp.square(jnp.maximum(top_eigs, 0.0)) + prev_group_tail[idx] * decay
    )
    mask = jnp.broadcast_to(jnp.arange(u.shape[1]) < new_budget, u.shape)
    eigvecs = u * mask
    mask = deflated > 0
    eigvecs *= mask
    eigvals = deflated * mask

    alpha = jnp.asarray(-1.0 / (2 * ndim), dtype=jnp.float32)
    if options.relative_epsilon and options.epsilon > 0:
      eps = jnp.max(undeflated) * options.epsilon
    else:
      eps = options.epsilon

    inv_eigvals = jnp.where(mask, (undeflated + eps) ** alpha, 0)
    inv_tail = jnp.where(tail > 0, (tail + eps) ** alpha, 0.0)

    group_state = _update_group_state_from_updates(
        group_state,
        (idx, new_start),
        (
            eigvecs,
            eigvals,
            inv_eigvals,
            jnp.array([tail]),
            jnp.array([inv_tail]),
            new_x,
            pi,
        ),
        (d, k, n),
    )

  return group_state


def _update_group_state_from_updates(
    group_state: _GroupState,
    ids: tuple[int, jax.Array],
    new_stats: tuple[
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
    ],
    sizes: tuple[int, int, int],
) -> _GroupState:
  """Update group state from updates of an axis."""

  eigvecs, eigvals, inv_eigvals, tail, inv_tail, x, pi = new_stats
  d, k, n = sizes
  idx, new_start = ids

  new_eigvecs = jax.lax.dynamic_update_slice_in_dim(
      jnp.hstack((group_state.eigvecs, jnp.zeros((d, d - 1)))),
      eigvecs,
      new_start,
      axis=1,
  )
  new_eigvecs = jax.lax.dynamic_slice_in_dim(new_eigvecs, 0, n*k, axis=1)
  new_eigvals = jax.lax.dynamic_update_slice(
      jnp.hstack((group_state.eigvals, jnp.zeros(d - 1))),
      eigvals,
      (new_start,)
  )
  new_eigvals = jax.lax.dynamic_slice_in_dim(new_eigvals, 0, n*k, axis=0)
  new_inv_eigvals = jax.lax.dynamic_update_slice(
      jnp.hstack((group_state.inv_eigvals, jnp.zeros(d - 1))),
      inv_eigvals,
      (new_start,)
  )
  new_inv_eigvals = jax.lax.dynamic_slice_in_dim(
      new_inv_eigvals, 0, n*k, axis=0
  )
  new_tail = jax.lax.dynamic_update_slice(
      group_state.tail, tail, jnp.array([idx])
  )
  new_inv_tail = jax.lax.dynamic_update_slice(
      group_state.inv_tail, inv_tail, jnp.array([idx])
  )

  group_state = _GroupState(
      eigvecs=new_eigvecs,
      eigvals=new_eigvals,
      inv_eigvals=new_inv_eigvals,
      tail=new_tail,
      inv_tail=new_inv_tail,
      memory_x=x,
      memory_pi=pi,
  )

  return group_state


def _one_step_fkm(
    options: Options,
    key: jax.Array,
    prev_x: jax.Array,
    tail: jax.Array,
    sizes: tuple[int, int, int],
) -> tuple[jax.Array, jax.Array]:
  """Perform one-step FKM update given previous memory_x and tail."""

  d, k, n = sizes
  key1, key2 = jax.random.split(key)
  tail_loss = jnp.sum(tail)
  delta, eta = options.delta, options.eta
  noise = jax.random.normal(key1, (n,))
  noise /= jnp.linalg.norm(noise)
  pi = _random_round(key2, prev_x + delta * noise)
  grad_est = n * tail_loss * noise / delta
  one_step = prev_x - eta * grad_est
  pi = jnp.minimum(pi, d)
  pi = jnp.maximum(pi, 1)
  return pi, _proj(one_step, k * n, delta)


def _random_round(key: jax.Array, v: jax.Array) -> jax.Array:
  """Round (unbiased) a vector to integer value."""

  delta = jax.random.uniform(key, shape=v.shape, minval=-0.5, maxval=0.5)
  return round(v + delta)


def _proj(v: jax.Array, constraint: int, delta: jnp.float32) -> jax.Array:
  """Projection on the feasible set: the ell_1 inequality constrained set."""

  def _proj_simplex(v, r):
    v_sorted = jnp.sort(v)[::-1]
    css = jnp.cumsum(v_sorted)
    rho = jnp.sum(
        jnp.where(v_sorted > (css - r) / jnp.arange(1, len(v) + 1), 1, 0)
    )
    theta = (css[rho - 1] - r) / rho
    proj = jnp.maximum(v - theta, 0)
    return proj

  r = (1 - delta) * constraint

  cond = (jnp.all(v >= 0)) & (jnp.sum(v) <= r)

  return jax.lax.cond(
      cond,
      lambda _: v,
      lambda _: _proj_simplex(v, r),
      None,
  )
