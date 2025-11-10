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

"""Shampoo second-order statistics preconditioning."""

import dataclasses
import functools
import math
import string
from typing import Iterator, NamedTuple, Optional, Sequence

import chex
import jax
from jax import numpy as jnp
import optax
from precondition.tearfree import praxis_shim


@dataclasses.dataclass
class Options:
  """Shampoo covariance approximation options.

  See https://arxiv.org/abs/2002.09018.

  Attributes:
    block_size: Determines the block size for Shampoo's block-diagonal gradient
      covariance approximation.
    update_preconditioners_freq: Number of steps between preconditioner updates.
    update_statistics_freq: Number of steps between statistics updates.
    second_moment_decay: Decay rate for second moment exponential moving
      average. If 1.0 then sums.
  """

  block_size: int = 1024
  update_preconditioners_freq: int = 1
  update_statistics_freq: int = 1
  second_moment_decay: float = 0.999
  # TODO(vladf):
  # lb, rb, b sharding: maybe later?
  # spmd_mesh_axis_names: Sequence[str] = () maybe later?


def apply(options: Options) -> praxis_shim.ShardedGradientTransformation:
  """Return gradient transform for (blocked) shampoo preconditioning."""

  _validate(options)

  # raise if no unit dims (must be scalar)

  # not intentional constants from distributed shampoo
  # exponent_override = 0
  # matrix_epsilon = 0 and eigh = True

  return praxis_shim.ShardedGradientTransformation(
      functools.partial(_init, options),
      functools.partial(_update, options),
      functools.partial(_pspec, options),
  )


class _AxesBlocks(NamedTuple):
  """Represents statistics or preconditioner matrices for a single tensor.

  Maintains the second-order statistics for gradients, to be used in second
  order optimization.

  There are two key matrices that are maintained per axis, when not using any
  blocking approximations.

  For each axis `i` with length `d_i`, we track the `(d_i, d_i)` covariances
  and their inverse p-th roots, where `p` is twice the rank of the gradients
  whose covariances we're tracking, excluding unit dimensions.

  A covariance for a higher-order tensor's i-th axis is recovered from the outer
  product of contracting all but the axis i. E.g., for an order-3 tensor G, the
  covariance for its 0th dimension is `C_{ij} = sum_{k,l} G_{ikl}G_{jkl}`.

  Since these matrices can be quite large for tensors with large dimensions, we
  introduce an approximation which stores just block diagonal components of
  these matrices. This corresponds to the full covariance statistics of the
  disjoint partitions of these tensors, with large dimensions blocked by a
  provided block size.

  Below, we refer to all large dimensions as those at least equal to the block
  size.

  When storing block diagonal matrices, we store them as `N` blocks of size `B`.

  Attributes:
    stats: A list of length equal to the tensor's ndim with blocks of matrices
      of shape [N, B, B] where B is an axis-specific block size, but at most
      block_size.
    roots: Same shape as stats, inverse p-th roots for statistics, where p is
      the twice the rank of the original parameter to optimize.
  """

  stats: list[jax.Array]
  roots: list[jax.Array]


class _ShampooState(NamedTuple):
  # A scalar int32 for step count.
  count: jax.Array
  # A tree of the same shape as the params of _AxesBlocks leaves of f32.
  blocks: chex.ArrayTree


@dataclasses.dataclass(frozen=True)
class _BlocksMetadata:
  """Metadata for _AxesBlocks to track indexing information.

  Attributes:
    block_sizes: Per-dimension statistics & preconditioner block sizes, length
      is equal to rank of tensor we have metadata for.
    num_blocks: `N`, the total number of blocks.
    debug_name: A way to refer to this parameter in debug messages.
    large_block_size: The block size originally specified in the shampoo
      options, this is the minimum size for a dimension to be considered large.
    param_shape: The shape of the original parameter we're blocking.
    large_axes: Axes with dimension at least `large_block_size` in the original
      tensor.
    blocks_per_large_axis: Number of blocks in the corresponding axis for each
      axis in large_axes.
    blocks_axis: The axis of the the `N` in the blocked tensor (see _blockify).
  """

  block_sizes: list[int]
  num_blocks: int
  debug_name: str
  large_block_size: int
  param_shape: list[int]
  large_axes: list[int]
  blocks_per_large_axis: list[int]
  blocks_axis: int


def _blocks_metadata(
    options: Options, param_shape: Sequence[int], debug: str
) -> _BlocksMetadata:
  """Generate the blocks metadata for a parameter."""
  dims = [min(dim, options.block_size) for dim in param_shape]
  large_axes = [i for i, d in enumerate(param_shape) if d >= options.block_size]
  blocks_per_large_axis = [
      param_shape[i] // options.block_size for i in large_axes
  ]
  num_blocks = math.prod(blocks_per_large_axis + [1])

  return _BlocksMetadata(
      block_sizes=dims,
      num_blocks=num_blocks,
      debug_name=debug,
      large_block_size=options.block_size,
      large_axes=large_axes,
      param_shape=list(param_shape),
      blocks_per_large_axis=blocks_per_large_axis,
      blocks_axis=min(large_axes, default=0),
  )


def _validate(options: Options) -> None:
  """Raise ValueError if options are invalid."""
  if options.block_size <= 1:
    raise ValueError(f"block_size ({options.block_size}) must be >1")

  if options.update_preconditioners_freq <= 0:
    raise ValueError(
        "update_preconditioners_freq ({}) must be positive".format(
            options.update_preconditioners_freq
        )
    )

  if options.update_statistics_freq <= 0:
    raise ValueError(
        "update_statistics_freq ({}) must be positive".format(
            options.update_statistics_freq
        )
    )

  if not (0 <= options.second_moment_decay <= 1):
    raise ValueError(
        f"second_moment_decay ({options.second_moment_decay}) "
        "should be in [0, 1]"
    )


def _init(options: Options, params: optax.Params) -> _ShampooState:
  """Inititialize stats to 0 and preconditioners to identity."""

  def make_blocks(path: ..., param: jax.Array) -> _AxesBlocks:
    if any(dim == 1 for dim in param.shape):
      raise ValueError(
          "param {} shape ({}) has unit dimensions".format(path, param.shape)
      )

    if sum(dim >= options.block_size for dim in param.shape) > 2:
      raise ValueError(
          "param {} shape ({}) has >2 large dims for block size {}".format(
              path, param.shape, options.block_size
          )
      )

    if any(
        dim % options.block_size != 0
        for dim in param.shape
        if dim >= options.block_size
    ):
      raise ValueError(
          "param {} shape ({}) has large dims indivisible by block size {}"
          .format(path, param.shape, options.block_size)
      )

    meta = _blocks_metadata(options, param.shape, str(path))
    n = meta.num_blocks
    dims = meta.block_sizes
    stats = [jnp.zeros((n, d, d)) for d in dims]
    precond = [jnp.eye(d) * jnp.ones((n, 1, 1)) for d in dims]
    return _AxesBlocks(stats, precond)

  return _ShampooState(
      count=jnp.zeros([], jnp.int32),
      blocks=jax.tree_util.tree_map_with_path(make_blocks, params),
  )


def _pspec(
    options: Options, params: praxis_shim.NestedHParams
) -> praxis_shim.NestedHParams:
  """Generate sharding specification for shampoo state."""
  count_pspec = praxis_shim.WeightHParams(
      shape=[],
      init=None,
      dtype=jnp.int32,
      collections=None,
      tensor_split_dims_mapping=[],
  )

  def make_blocks_pspec(
      path: ...,
      param: praxis_shim.WeightHParams,
  ) -> praxis_shim.NestedHParams:
    meta = _blocks_metadata(options, param.shape, str(path))
    num_blocks = meta.num_blocks
    dims = meta.block_sizes
    replicated = functools.partial(
        praxis_shim.WeightHParams,
        init=None,
        dtype=jnp.float32,
        collections=None,
        tensor_split_dims_mapping=[-1, -1, -1],
    )
    stats = [replicated((num_blocks, d, d)) for d in dims]  # pytype: disable=wrong-arg-types
    precond = stats
    return dict(stats=stats, roots=precond)

  return dict(
      count=count_pspec,
      blocks=jax.tree_util.tree_map_with_path(
          make_blocks_pspec, params, is_leaf=lambda x: hasattr(x, "shape")
      ),
  )


def _update(
    options: Options,
    updates: optax.Updates,
    state: _ShampooState,
    params: Optional[optax.Params] = None,
) -> tuple[optax.Updates, _ShampooState]:
  """Update internal shampoo stats and precondition gradients."""
  del params
  meta = jax.tree_util.tree_map_with_path(
      lambda path, x: _blocks_metadata(options, x.shape, str(path)), updates
  )
  blocks = state.blocks
  blockified_updates = jax.tree.map(_blockify, updates, meta)
  is_block = lambda x: isinstance(x, _AxesBlocks)

  stats_updated_blocks = functools.partial(
      jax.tree.map,
      functools.partial(_update_block_stats, options.second_moment_decay),
      blockified_updates,
      blocks,
      meta,
      is_leaf=is_block,
  )
  should_update_stats = (state.count % options.update_statistics_freq) == 0
  blocks = jax.lax.cond(
      should_update_stats, stats_updated_blocks, lambda: blocks
  )

  precond_updated_blocks = functools.partial(
      jax.tree.map,
      _update_block_precond,
      blocks,
      meta,
      is_leaf=is_block,
  )
  should_update_precond = (
      state.count % options.update_preconditioners_freq
  ) == 0
  blocks = jax.lax.cond(
      should_update_precond, precond_updated_blocks, lambda: blocks
  )
  new_state = _ShampooState(count=state.count + 1, blocks=blocks)
  new_updates = jax.tree.map(
      _precondition_blocks, blockified_updates, blocks, meta, is_leaf=is_block
  )
  new_updates = jax.tree.map(_deblockify, new_updates, meta)

  return new_updates, new_state


def _blockify(x: jax.Array, meta: _BlocksMetadata) -> jax.Array:
  """Reshape the update such that it is blocked along large dimensions.

  Inserts the `N` dimension dimension right on the first axis in
  `meta.large_axes`, which is the `meta.blocks_axis` in the returned tensor.

  Shifts all original axes in `x` that are on or after `meta.blocks_axis`
  (including what was originally the first axis in `meta.large_axes`) forward
  by one. All large axes will now be of length equal to the largest block size.

  In the case that there's no blocking, we put a dummy blocks axis in axis 0
  with dimension 1, so the handling of the original axes is the same as the
  blocked cases.

  For example:
    - Suppose block size is 5 and x is shaped [3, 20, 25, 4]. Then
      x has two large axes (1 and 2). The resulting blocked value will be
      shaped [3, (4*5), 5, 5, 4], with the large axes being converted
      to block size and a new 20-dimensional axis with the product of the
      4 blocks for the original axis 1 and 5 blocks for the original axis 2.
      All other axes are kept the same. Note meta.blocks_axis precedes the
      large axes' new locations, so it's set to 1.
    - Suppose block size is 5 and x is shaped [5, 2]. The result will be
      [1, 5, 2], with the first dimension corresponding to the single block at
      axis 0.
    - Suppose block size is 5 and x is [3, 4]. There are no large axes, but to
      get rid of edge cases we still add a meta.blocks_axis at axis 0 with a
      single block [1, 3, 4].
    - Suppose block size is 5 and x is [15, 2, 10]. We'll return
      [(3*2), 5, 2, 5], following the same rules as before. Note the large
      axes stay in place.

  Args:
    x: Input to block.
    meta: Metadata about the input.

  Returns:
    A blocked version of the input and the dimension with the number of
    blocks.
  """
  assert list(x.shape) == meta.param_shape, (x.shape, meta.param_shape)

  if not meta.large_axes:
    # Just create a unit N/blocks axis.
    return jnp.expand_dims(x, meta.blocks_axis)

  if len(meta.large_axes) == 1:
    # Block the only large axis.
    before, after = _split_exclusively(x.shape, meta.large_axes)
    new_shape = before + [meta.num_blocks, meta.large_block_size] + after
    return x.reshape(new_shape)

  assert len(meta.large_axes) == 2, meta.large_axes

  # Extract the blocks from both large axes.
  l_blocks, r_blocks = meta.blocks_per_large_axis
  before, middle, after = _split_exclusively(x.shape, meta.large_axes)
  stitch = lambda l, r: before + l + middle + r + after
  split_blocked_shape = stitch(
      [l_blocks, meta.large_block_size], [r_blocks, meta.large_block_size]
  )
  split_blocked_x = x.reshape(split_blocked_shape)

  # Move over the blocks from the right axis next to the left one.
  perm = list(range(len(split_blocked_shape)))
  l_blocks_ix = len(before)
  r_blocks_ix = len(before) + 2 + len(middle)
  perm.pop(r_blocks_ix)
  perm.insert(l_blocks_ix + 1, r_blocks_ix)
  adjacent_blocked_x = jnp.transpose(split_blocked_x, perm)

  # Transpose the previous sharding too.
  new_shape = stitch(
      [meta.num_blocks, meta.large_block_size], [meta.large_block_size]
  )
  reshaped = adjacent_blocked_x.reshape(new_shape)
  assert l_blocks_ix == meta.blocks_axis

  return reshaped


def _deblockify(blocked_x: jax.Array, meta: _BlocksMetadata) -> jax.Array:
  """Invert _blockify()."""
  if not meta.large_axes:
    return jnp.squeeze(blocked_x, meta.blocks_axis)

  if len(meta.large_axes) == 1:
    return blocked_x.reshape(meta.param_shape)

  assert len(meta.large_axes) == 2

  # Re-split the blocks axis.
  assert blocked_x.shape[meta.blocks_axis] == meta.num_blocks
  before, after = _split_exclusively(blocked_x.shape, [meta.blocks_axis])
  split_blocks_shape = before + meta.blocks_per_large_axis + after
  split_blocked_x = blocked_x.reshape(split_blocks_shape)

  # Move the right large axis blocks back in front of their axis.
  perm = list(range(len(split_blocked_x.shape)))
  # In blocked_x:
  # [..., blocks axis, left block, ..., right block, ...]
  #       ^blocks_axis                  ^large_axes[1] + 1
  # In split_blocked_x:
  # [..., left blocks, right blocks, left block, ..., right block, ...]
  #       ^blocks_axis                                ^large_axes[1] + 2
  r_blocks_ix = meta.blocks_axis + 1
  r_blocks_val = perm.pop(r_blocks_ix)
  # After pop:
  # [..., left blocks, left block, ..., right block, ...]
  #       ^blocks_axis                  ^large_axes[1] + 1
  r_blocked_axis_ix = meta.large_axes[1] + 1
  perm.insert(r_blocked_axis_ix, r_blocks_val)
  split_blocked_x = jnp.transpose(split_blocked_x, perm)

  reshaped = jnp.reshape(split_blocked_x, meta.param_shape)
  return reshaped


def _update_block_stats(
    second_moment_decay: float,
    update: jax.Array,
    block: _AxesBlocks,
    meta: _BlocksMetadata,
) -> _AxesBlocks:
  """Update covariance statistics given a blocked gradient."""

  new_stats = []
  with jax.named_scope("ShampooStats"):
    for axis, cov in enumerate(block.stats):
      all_axes = list(range(len(meta.param_shape)))
      all_axes.remove(axis)

      dot_all = functools.partial(jnp.tensordot, axes=(all_axes, all_axes))
      batched_tensordot = jax.vmap(
          dot_all, in_axes=meta.blocks_axis, out_axes=0
      )
      new_cov = batched_tensordot(update, update)
      new_stats.append(_ema_update(cov, new_cov, second_moment_decay))

  return _AxesBlocks(stats=new_stats, roots=block.roots)


def _pth_inv_root(p: int, cov: jax.Array) -> jax.Array:
  """Calculate a batch of p-th inverse roots."""
  eps = 1e-6
  w, v = jnp.linalg.eigh(cov)
  mask = w <= eps * jnp.max(w)
  half = jnp.where(mask, 1.0, w) ** (-0.5 / p)
  half = jnp.where(mask, 0.0, half)
  half_v = jnp.expand_dims(half, -2) * v
  return jnp.einsum("bik,bjk->bij", half_v, half_v)


def _update_block_precond(
    block: _AxesBlocks, meta: _BlocksMetadata
) -> _AxesBlocks:
  """Update preconditioners."""
  p = len(meta.param_shape) * 2

  with jax.named_scope("PthInvRoot"):
    new_roots = list(map(functools.partial(_pth_inv_root, p), block.stats))

  return _AxesBlocks(roots=new_roots, stats=block.stats)


def _precondition_blocks(
    update: jax.Array, blocks: _AxesBlocks, meta: _BlocksMetadata
) -> jax.Array:
  """Precondition blocked gradients."""
  it = _einsum_letters(meta)
  blocks_axis_letter = next(it)

  # Contract along the innermost axis of each preconditioner,
  # making the other equal-length axis the output.
  contraction_letters = [next(it) for _ in meta.param_shape]
  output_letters = [next(it) for _ in meta.param_shape]
  preconditioners = blocks.roots
  preconditioner_inputs = [
      blocks_axis_letter + o + c
      for c, o in zip(contraction_letters, output_letters)
  ]

  blocked_input = contraction_letters[:]
  blocked_input.insert(meta.blocks_axis, blocks_axis_letter)
  blocked_input = "".join(blocked_input)
  blocked_output = output_letters[:]
  blocked_output.insert(meta.blocks_axis, blocks_axis_letter)
  blocked_output = "".join(blocked_output)

  # Build up the einsum equation and invoke it.
  inputs = ",".join([blocked_input] + preconditioner_inputs)
  formula = inputs + "->" + blocked_output
  with jax.named_scope("PreconditionShampoo"):
    print(formula, update.shape, [x.shape for x in preconditioners])
    return jnp.einsum(formula, update, *preconditioners)


def _split_exclusively(
    ls: Sequence[int], splits: Sequence[int]
) -> list[list[int]]:
  """Returns possibly-empty segments between sorted split points in ls."""
  assert all(
      l < r for l, r in zip(splits, splits[1:])
  ), f"splits {splits} must be distinct ascending"
  assert all(
      0 <= i < len(ls) for i in splits
  ), f"splits {splits} must index into list {ls} of length {len(ls)}"
  splits = [-1] + list(splits) + [len(ls)]
  return [list(ls[l + 1 : r]) for l, r in zip(splits, splits[1:])]


def _einsum_letters(meta: _BlocksMetadata) -> Iterator[str]:
  for c in string.ascii_letters:
    yield c

  raise ValueError(
      f"shape {meta.param_shape} too high-dimensional for {meta.debug_name}"
  )


def _ema_update(old: jax.Array, new: jax.Array, decay: float) -> jax.Array:
  if decay == 1.0:
    return old + new
  return old * decay + new * (1 - decay)
