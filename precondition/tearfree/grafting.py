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

"""Grafting norm adjustment (https://openreview.net/forum?id=FpKgG31Z_i9)."""

import copy
import dataclasses
import enum
import functools
from typing import Any, NamedTuple

import chex
from flax import struct
import jax
from jax import numpy as jnp
import optax
from precondition.tearfree import praxis_shim


@enum.unique
class GraftingType(enum.Enum):
  """Different grafting types."""

  NONE = 'none'
  SGD = 'sgd'
  RMSPROP = 'rmsprop'
  ADAFACTOR = 'adafactor'


@dataclasses.dataclass
class Options:
  """Grafting configuration to change norms for updates.

  A grafting update is computed as if it was running alongside the
  tearfree optimizer. Its norm is used for updates. During the initial
  few steps before preconditioning is applied, the grafting update
  is used entirely.

  Note that the grafting optimizer is ignorant of both weight decay,
  learning rate, and the momentum.

  Attributes:
    grafting_type: Which optimizer to use for grafting updates.
    second_moment_decay: Second moment accumulator decay. For ADA-Factor, this
    value must be bounded between (0, 1). For RMSProp, the second moment
    accumulator becomes sum if set to 1 (i.e., Adagrad), should be in (0, 1].
    Must be 0 if unused (e.g., for SGD/NONE).
    start_preconditioning_step: When to start applying preconditioning.
    epsilon: Avoids divide by zero in RMSProp and ADA-Factor by adding this term
      to the expression `(epsilon + acc)^(-1/2)` when taking the inverse square
      root of the accumulator; should be non-negative.
    skip_preconditioning_any_dim_gt: Skip second-order preconditioning if any
      dimension of a tensor is greater than this value (only apply grafting
      update). Argument ignored if NONE grafting.
    skip_preconditioning_rank1: Skip preconditioning the tensor if the rank is 1
      or less. Argument ignored if NONE grafting.
    min_dim_size_to_factor: (Applies to ADA-Factor Only.) Only factor the
      statistics if two array dimensions have at least this size.
    multiply_by_parameter_scale: (Applies to ADA-Factor Only.) If True, then
      scale learning_rate by parameter norm. If False, provided learning_rate is
      absolute step size.
    clipping_threshold: (Applies to ADA-Factor Only.) Clipping
      threshold. Must be >= 1.
  """

  grafting_type: GraftingType = GraftingType.RMSPROP
  second_moment_decay: float = 0.999
  start_preconditioning_step: int = 0
  epsilon: float = 1e-23
  skip_preconditioning_any_dim_gt: int = 4096
  skip_preconditioning_rank1: bool = True
  min_dim_size_to_factor: int = 128
  multiply_by_parameter_scale: float = True
  clipping_threshold: float = 1.0


def graft(
    options: Options,
    direction: praxis_shim.ShardedGradientTransformation,
) -> praxis_shim.ShardedGradientTransformation:
  """Generate the grafting update from options and direction update.

  Args:
    options: The grafting options.
    direction: A sharded gradient transformation which determines the direction
      of the update (grafting, if applied, changes the norm of this update).

  Returns:
    The wrapped transformation which applies either the grafting update
    directly or the `direction` update with grafting norm, depending on the
    current step or whether to apply grafting/preconditioning at all.
  """
  _validate(options)

  if options.grafting_type == GraftingType.NONE:
    return direction

  if options.grafting_type == GraftingType.SGD:
    return _graft_with(direction, _sgd(), options)

  if options.grafting_type == GraftingType.RMSPROP:
    return _graft_with(
        direction,
        _rmsprop(options),
        options,
    )

  if options.grafting_type == GraftingType.ADAFACTOR:
    return _graft_with(
        direction,
        _adafactor(options),
        options,
    )
  # check options for validity (SGD/none and no 2nd moment, appropriate range)
  # test to check sharded gradient transform is otherwise identical to
  #   praxis'
  raise NotImplementedError


def _validate(options: Options):
  """Raise ValueError if the options have an invalid specification."""
  if options.grafting_type in [GraftingType.RMSPROP, GraftingType.ADAFACTOR]:
    if options.epsilon < 0:
      raise ValueError(
          'epsilon ({}) should be non-negative'.format(options.epsilon)
      )
  if options.grafting_type == GraftingType.RMSPROP:
    if not (0 < options.second_moment_decay <= 1.0):
      raise ValueError(
          'second_moment_decay ({}) not in (0, 1] for graft ({})'.format(
              options.second_moment_decay, options.grafting_type
          )
      )
  if options.grafting_type == GraftingType.ADAFACTOR:
    if not (0 < options.second_moment_decay < 1.0):
      raise ValueError(
          'second_moment_decay ({}) not in (0, 1) for graft ({})'.format(
              options.second_moment_decay, options.grafting_type
          )
      )
    if not (0 < options.min_dim_size_to_factor):
      raise ValueError(
          'min_dim_size_to_factor ({}) should be positive for graft ({})'
          .format(options.min_dim_size_to_factor, options.grafting_type)
      )
    if (options.clipping_threshold < 1):
      raise ValueError(
          'clipping_threshold ({}) should be >= 1 for graft ({})'
          .format(options.clipping_threshold, options.grafting_type)
      )


def _sgd() -> praxis_shim.ShardedGradientTransformation:
  """Create SGD sharded gradient transform."""
  grad_transform = optax.identity()
  return praxis_shim.ShardedGradientTransformation(
      grad_transform.init,
      grad_transform.update,
      optax.EmptyState,
  )


def _adafactor(options: Options) -> praxis_shim.ShardedGradientTransformation:
  """Create AdaFactor sharded gradient transform."""
  grad_transform = optax.adafactor(
      min_dim_size_to_factor=options.min_dim_size_to_factor,
      decay_rate=options.second_moment_decay,
      multiply_by_parameter_scale=options.multiply_by_parameter_scale,
      eps=options.epsilon, clipping_threshold=options.clipping_threshold)

  def _adafactor_pspec_fn(params_unused):
    del params_unused
    raise NotImplementedError

  return praxis_shim.ShardedGradientTransformation(
      grad_transform.init,
      grad_transform.update,
      _adafactor_pspec_fn,
  )


# Dummy wrapper for better state pretty printing, to identify what parameters
# are for.
class RMSPropAccumulator(NamedTuple):
  """State holding the sum/ema of gradient squares so far."""

  acc: optax.Updates


def _rmsprop(options: Options) -> praxis_shim.ShardedGradientTransformation:
  """Create RMSProp sharded gradient transform."""

  def init_fn(params):
    acc = jax.tree_map(jnp.zeros_like, params)
    return RMSPropAccumulator(acc=acc)

  def update_fn(updates, state, params=None):
    del params

    def ema(prev, new):
      second_moment_decay = options.second_moment_decay
      snew = jnp.square(new)
      if second_moment_decay == 1.0:
        return snew + prev
      else:
        return snew * (1 - second_moment_decay) + second_moment_decay * prev

    new_state = RMSPropAccumulator(jax.tree_map(ema, state.acc, updates))
    epsilon = options.epsilon
    new_updates = jax.tree_map(
        lambda g, acc: g * jax.lax.rsqrt(acc + epsilon), updates, new_state.acc
    )
    return new_updates, new_state

  def init_partition_spec_fn(mdl_params):
    def _opt_state_sharding_spec(var_hparams):
      s_var_hparams = copy.deepcopy(var_hparams)
      s_var_hparams.init = None
      return s_var_hparams

    mdl_sharding = jax.tree_map(_opt_state_sharding_spec, mdl_params)
    return RMSPropAccumulator(acc=mdl_sharding)

  return praxis_shim.ShardedGradientTransformation(
      init=init_fn, update=update_fn, init_partition_spec=init_partition_spec_fn
  )


class GraftingState(NamedTuple):
  """State holding the count for grafting."""

  count: jax.Array
  direction: optax.OptState
  norm: optax.OptState


def _graft_with(
    direction: praxis_shim.ShardedGradientTransformation,
    norm: praxis_shim.ShardedGradientTransformation,
    options: Options,
) -> praxis_shim.ShardedGradientTransformation:
  """Created a maybe-grafted update from a base update and a graft one."""

  start_preconditioning_step = options.start_preconditioning_step
  mask = functools.partial(_mask_skipped, options)

  def init_fn(params):
    return GraftingState(
        count=jnp.zeros([], jnp.int32),
        direction=direction.init(mask(params)),
        norm=norm.init(params),
    )

  def update_fn(updates, state, params=None):
    base_updates, base_state = direction.update(
        mask(updates), state.direction, mask(params)
    )
    graft_updates, graft_state = norm.update(updates, state.norm, params)
    new_state = GraftingState(
        count=state.count + 1,
        direction=base_state,
        norm=graft_state,
    )

    def maybe_graft(graft_upd, base):
      if _masked(base):
        return graft_upd
      assert graft_upd.shape == base.shape

      base_norm = jnp.linalg.norm(base)
      multiplier = jnp.where(
          base_norm > 0.0, jnp.linalg.norm(graft_upd) / base_norm, 0.0
      )
      return jnp.where(
          state.count >= start_preconditioning_step,
          base * multiplier,
          graft_upd,
      )

    new_updates = jax.tree_map(
        maybe_graft, graft_updates, base_updates, is_leaf=_masked
    )
    return new_updates, new_state

  def init_partition_spec_fn(mdl_params):
    count_pspec = praxis_shim.WeightHParams(
        shape=[],
        init=None,
        dtype=jnp.int32,
        collections=None,
        tensor_split_dims_mapping=[],
    )
    return dict(
        count=count_pspec,
        direction=direction.init_partition_spec(mdl_params),
        norm=norm.init_partition_spec(mdl_params),
    )

  return praxis_shim.ShardedGradientTransformation(
      init_fn,
      update_fn,
      init_partition_spec_fn,
  )


@struct.dataclass
class _GraftMask:
  """Helper tuple which masks out params before preconditioning."""

  pass


def _mask_skipped(options: Options, tree: chex.ArrayTree) -> chex.ArrayTree:
  """Masks out arrays to which preconditioning should not be applied."""

  def _maybe_mask(x: jax.Array):
    if options.skip_preconditioning_rank1 and x.ndim <= 1:
      return _GraftMask()
    if any(s > options.skip_preconditioning_any_dim_gt for s in x.shape):
      return _GraftMask()
    return x

  return jax.tree_map(_maybe_mask, tree)


def _masked(tree_node: Any) -> bool:
  """Returns whether a tree node has been masked out."""
  return isinstance(tree_node, _GraftMask)
