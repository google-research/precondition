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

"""Tearfree optimizer implementation.

OOM making your eyes water? Try the Tearfree Shampoo optimizer.

This module handles logic for

1. Statistics/preconditioner update frequency
2. Applying momentum
3. Combining grafting and preconditioning updates, applying grafting
4. Typical update procedures, like learning rate, momentum, etc.
"""

import dataclasses
from typing import Union

import chex
import optax
from precondition.tearfree import grafting
from precondition.tearfree import momentum
from precondition.tearfree import praxis_shim
from precondition.tearfree import second_order


@dataclasses.dataclass
class TearfreeOptions:
  """Configuration dataclass for tearfree optimizer.

  Attributes:
    grafting_options: Grafting options to modify update norm (see
      `grafting.Options`).
    second_order_options: Second-order statistics tracking options (see
      `second_order.Options`).
    momentum_options: Momentum options (see `momentum.Options`).
  """

  grafting_options: grafting.Options = grafting.Options()
  second_order_options: second_order.Options = second_order.Options()
  momentum_options: momentum.Options = momentum.Options()


def tearfree(
    learning_rate: Union[chex.Numeric, optax.Schedule],
    options: TearfreeOptions,
) -> praxis_shim.ShardedGradientTransformation:
  """Tearfree optimizer, supports pjit and jit.

  Preconditioned, grafted updates with momentum.

  One key difference in the logic is to only use a single momentum between
  the graft and preconditioned update. `distributed_shampoo` keeps a separate
  `diagonal_momentum` buffer, but never uses it after preconditioning is
  active (it is not used to adjust the grafting norm). This implies (1)
  we save memory (only one momentum buffer), (2) we are identical to
  `distributed_shampoo` if there is no warmup or no preconditioning
  (`options.start_preconditioning_step` is inf or 0).

  Args:
    learning_rate: The learning rate value or schedule. Learning rate is
      "decoupled", i.e., we always apply it last to the update (after weight
      decay, after momentum, etc.).
    options: Tearfree optimizer options.

  Returns:
    The sharded gradient transformation corresponding to an updated,
      preconditioned gradient, times the negative learning rate.
  """

  second_order_tx = second_order.apply(options.second_order_options)
  graft_tx = grafting.graft(options.grafting_options, second_order_tx)
  momentum_tx = momentum.apply(options.momentum_options)
  if callable(learning_rate):
    lr_tx = optax.scale_by_schedule(lambda x: -1.0 * learning_rate(x))
  else:
    lr_tx = optax.scale(-1.0 * learning_rate)
  return praxis_shim.sharded_chain(
      graft_tx,
      momentum_tx,
      lr_tx,
  )
