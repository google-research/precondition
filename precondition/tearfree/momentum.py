# Copyright 2024 The precondition Authors.
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

"""Momentum configuration and transform."""

import copy
import dataclasses
from typing import Union

import jax
import optax
from precondition.tearfree import praxis_shim


@dataclasses.dataclass
class Options:
  """Configuration dataclass for momentum.

  Notably, this class contains weight decay parameters. Why?

  In classical convex literature, Nesterov acceleration applied to gradient
  descent can be viewed as "revising" the last iterate's momentum based on
  the gradient we observe immediately after taking a momentum "gamble"
  (see viz, https://stats.stackexchange.com/a/191727).

  To maintain this interpretation exactly, we would need to go against
  the grain on how weight decay is implemented. Momentum must be the last*
  gradient transformation applied to the iterate, which would require the
  weight decay to be applied to the update before it's used to change
  the velocity (momentum's state, the first moment).

  In particular, AdamW and Adafactor suggest direct weight downscaling,
  excluding weight decay from the velocity accumulation.

  As a result, the true meaning of Nesterov acceleration here is better
  understood literally, described in its parameter doc.

  *Technically, some optimizers include the learning rate in the update used to
  update the velocity (e.g., Adafactor), but others apply the learning rate
  scaling last, after momentum (e.g., Adam). We can recover the former from the
  latter by dividing the decay by the root of the learning rate, so this
  particular "gradient transformation" shouldn't be viewed as affecting
  the Nesterov interpretation, up to tuning constants.

  Attributs:
    ema: If true, momentum is computed as an exponential moving
      average: `velocity(t+1) = decay * velocity(t) + (1 - decay) * update(t)`
      If false, then uses "trace" accumulation for momentum:
      `velocity(t+1) = decay * velocity(t) + update(t)`. Note that if the
      updates were the same (they aren't) then these would be the same up to a
      factor of `(1 - decay)`. This corresponds to distributed_shampoo argument
      `moving_average_for_momentum`.
    nesterov: Toggle for Nesterov acceleration. If false, then the new
      update `update'(t+1)` simply equals `velocity(t+1)`. If true, then
      `update'(t+1) = maybe_decay * update(t) + decay * velocity(t+1)`, where
      `maybe_decay` is `(1 - decay)` if `ema` and 1 otherwise.
    momentum_decay: The decay referred to in `ema` and `nesterov` formulas.
    weight_decay: Add `weight_decay * x(t)` to the `update(t)` value, where
      `x(t)` is the value of the current parameters.
    weight_decay_after_momentum: Whether weight decay addition is performed
      after the momentum transformation.
  """

  ema: bool = False
  nesterov: bool = True
  momentum_decay: float = 0.9
  weight_decay: float = 0.0
  weight_decay_after_momentum: bool = True


State = Union[optax.MaskedNode, optax.TraceState]


def apply(options: Options) -> praxis_shim.ShardedGradientTransformation:
  """Generate the momentum update from options."""
  _validate(options)

  momentum_transforms = []
  if options.momentum_decay:
    if options.ema:
      momentum_transforms.append(optax.scale(1 - options.momentum_decay))
    momentum_transforms.append(
        _sharded_trace(options.momentum_decay, options.nesterov)
    )

  wd_transforms = [optax.add_decayed_weights(options.weight_decay)] * (
      options.weight_decay > 0.0
  )

  if options.weight_decay_after_momentum:
    transforms = momentum_transforms + wd_transforms
  else:
    transforms = wd_transforms + momentum_transforms

  return praxis_shim.sharded_chain(*transforms)


def _validate(options: Options):
  """Raise ValueError if options are invalid."""
  if not (0 <= options.momentum_decay <= 1):
    raise ValueError(
        'momentum_decay ({}) must be in [0, 1]'.format(options.momentum_decay)
    )

  if not (options.weight_decay >= 0):
    raise ValueError(
        'weight_decay ({}) must be >= 0'.format(options.weight_decay)
    )


def _sharded_trace(
    momentum: float, nesterov: bool
) -> praxis_shim.ShardedGradientTransformation:
  """Extend optax's trace to allow sharding."""
  trace = optax.trace(momentum, nesterov)

  def init_pspec_fn(mdl_params):
    def _opt_state_sharding_spec(var_hparams):
      s_var_hparams = copy.deepcopy(var_hparams)
      s_var_hparams.init = None
      return s_var_hparams

    mdl_sharding = jax.tree_map(_opt_state_sharding_spec, mdl_params)
    return optax.TraceState(trace=mdl_sharding)

  return praxis_shim.ShardedGradientTransformation(
      trace.init, trace.update, init_pspec_fn
  )
