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
from typing import NamedTuple

import jax
from jax import numpy as jnp
import optax
from praxis import optimizers


@enum.unique
class GraftingType(enum.Enum):
  """Different grafting types."""

  NONE = 'none'
  SGD = 'sgd'
  RMSPROP = 'rmsprop'


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
    second_moment_decay: Second moment accumulator decay, becomes sum if set to
      1, should be in [0, 1]. Must be 0 if unused (e.g., for SGD/NONE).
    start_preconditioning_step: when to start applying preconditioning
  """

  grafting_type: GraftingType = GraftingType.RMSPROP
  second_moment_decay: float = 0.999
  start_preconditioning_step: int = 0
  # TODO(vladf): skip_any_dim_gt
  # TODO(vladf): skip_rank_1


def graft(
    options: Options,
    direction: optimizers.ShardedGradientTransformation,
) -> optimizers.ShardedGradientTransformation:
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
    return _graft_with(direction, _sgd(), options.start_preconditioning_step)

  if options.grafting_type == GraftingType.RMSPROP:
    return _graft_with(
        direction,
        _rmsprop(options.second_moment_decay),
        options.start_preconditioning_step,
    )
  # check options for validity (SGD/none and no 2nd moment, appropriate range)
  # test to check sharded gradient transform is otherwise identical to
  #   praxis'
  raise NotImplementedError


def _validate(options: Options):
  """Raise ValueError if the options have an invalid specification."""
  no_second_order = options.grafting_type in [
      GraftingType.NONE,
      GraftingType.SGD,
  ]

  if no_second_order and options.second_moment_decay != 0.0:
    raise ValueError(
        'second_moment_decay ({}) not 0 for graft ({}) that does not use it'
        .format(options.second_moment_decay, options.grafting_type)
    )

  if not (no_second_order or 0 < options.second_moment_decay <= 1.0):
    raise ValueError(
        'second_moment_decay ({}) not in (0, 1] for graft ({})'.format(
            options.second_moment_decay, options.grafting_type
        )
    )

  if (
      options.grafting_type == GraftingType.NONE
      and options.start_preconditioning_step > 0
  ):
    raise ValueError(
        'start_preconditioning_step ({}) is >0 but grafting_type is NONE'
        .format(options.start_preconditioning_step)
    )


def _sgd() -> optimizers.ShardedGradientTransformation:
  grad_transform = optax.identity()
  return optimizers.ShardedGradientTransformation(
      grad_transform.init,
      grad_transform.update,
      optax.EmptyState,
  )


# Dummy wrapper for better state pretty printing, to identify what parameters
# are for.
class RMSPropAccumulator(NamedTuple):
  """State holding the sum/ema of gradient squares so far."""

  acc: optax.Updates


def _rmsprop(
    second_moment_decay: float,
) -> optimizers.ShardedGradientTransformation:
  """Create RMSProp sharded gradient transform."""
  initial_accumulator_value = 1.0
  epsilon = 1e-7

  def init_fn(params):
    acc = jax.tree_map(
        lambda x: jnp.full_like(x, initial_accumulator_value), params
    )
    return RMSPropAccumulator(acc=acc)

  def update_fn(updates, state, params=None):
    del params

    def ema(prev, new):
      snew = jnp.square(new)
      if second_moment_decay == 1.0:
        return snew + prev
      else:
        return snew * (1 - second_moment_decay) + second_moment_decay * prev

    new_state = RMSPropAccumulator(jax.tree_map(ema, state.acc, updates))
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

  return optimizers.ShardedGradientTransformation(
      init=init_fn, update=update_fn, init_partition_spec=init_partition_spec_fn
  )


class GraftingState(NamedTuple):
  """State holding the count for grafting."""

  count: jax.Array
  direction: optax.OptState
  norm: optax.OptState


def _graft_with(
    direction: optimizers.ShardedGradientTransformation,
    norm: optimizers.ShardedGradientTransformation,
    start_preconditioning_step: int,
) -> optimizers.ShardedGradientTransformation:
  """Created a maybe-grafted update from a base update and a graft one."""

  def init_fn(params):
    return GraftingState(
        count=jnp.zeros([], jnp.int32),
        direction=direction.init(params),
        norm=norm.init(params),
    )

  def update_fn(updates, state, params=None):
    base_updates, base_state = direction.update(
        updates, state.direction, params
    )
    graft_updates, graft_state = norm.update(updates, state.norm, params)
    new_state = GraftingState(
        count=state.count + 1,
        direction=base_state,
        norm=graft_state,
    )

    def maybe_graft(graft_upd, base):
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

    new_updates = jax.tree_map(maybe_graft, graft_updates, base_updates)
    return new_updates, new_state

  def init_partition_spec_fn(mdl_params):
    return GraftingState(
        count=optimizers.count_init_partition_spec_fn(mdl_params)['count'],
        direction=direction.init_partition_spec(mdl_params),
        norm=norm.init_partition_spec(mdl_params),
    )

  return optimizers.ShardedGradientTransformation(
      init_fn,
      update_fn,
      init_partition_spec_fn,
  )
