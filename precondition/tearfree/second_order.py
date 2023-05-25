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

"""Various strategies for tracking second order statistics."""

import dataclasses
import enum
from typing import Optional

import optax
from praxis import optimizers
from precondition.tearfree import reshaper
from precondition.tearfree import shampoo


@enum.unique
class SecondOrderType(enum.Enum):
  """Different second order covariance tracking methods."""

  SHAMPOO = 'shampoo'
  # TODO(vladf): Add Sketchy, SketchyOne, perhaps even AdaFactor (for debug?)


@dataclasses.dataclass
class Options:
  """Toggle which second order statistics to track.

  Attributes:
    merge_dims: Merges small dimensions, see `reshaper.Options.merge_dims`.
    second_order_type: Which optimizer to use for grafting updates.
    shampoo_options: Options for blocked shampoo.
  """

  second_order_type: SecondOrderType = SecondOrderType.SHAMPOO
  shampoo_options: Optional[shampoo.Options] = shampoo.Options()
  merge_dims: int = 1024
  # As further SecondOrderTypes are added, add other optimizers.


def apply(options: Options) -> optimizers.ShardedGradientTransformation:
  """Generate the second order update from options."""
  reshaper_options = _reshaper_options(options)
  merge_tx = reshaper.merge(reshaper_options)
  precond_tx = _update_stats_and_precondition(options)

  def wrap_init(params: optax.Params):
    reshaped_params, _ = merge_tx.update(params, merge_tx.init(params), params)
    return precond_tx.init(reshaped_params)

  # TODO(vladf): later, we'll need to wrap pspec as well.
  wrapped_precond_tx = optimizers.ShardedGradientTransformation(
      wrap_init, precond_tx.update, precond_tx.init_partition_spec
  )

  return optimizers.sharded_chain(
      merge_tx,
      wrapped_precond_tx,
      reshaper.unmerge(reshaper_options),
  )


def _reshaper_options(options: Options) -> reshaper.Options:
  if options.second_order_type == SecondOrderType.SHAMPOO:
    assert options.shampoo_options
    block_size = options.shampoo_options.block_size
    return reshaper.Options(options.merge_dims, block_size)
  else:
    raise ValueError(
        'unknown second order type {}'.format(options.second_order_type)
    )


def _update_stats_and_precondition(
    options: Options,
) -> optimizers.ShardedGradientTransformation:
  if options.second_order_type == SecondOrderType.SHAMPOO:
    assert options.shampoo_options
    return shampoo.apply(options.shampoo_options)
  else:
    raise ValueError(
        'unknown second order type {}'.format(options.second_order_type)
    )
