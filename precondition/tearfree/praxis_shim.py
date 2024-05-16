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

"""Shim interfaces for praxis, to avoid circular dependencies."""

import dataclasses
from typing import Any, NamedTuple, Union

import jax
from jax import numpy as jnp
import optax


@dataclasses.dataclass(frozen=True)
class ShardedGradientTransformation:
  """GradientTransformation that supports spmd."""

  init: optax.TransformInitFn
  update: optax.TransformUpdateFn
  init_partition_spec: Any


NestedHParams = Any


class WeightHParams(NamedTuple):
  shape: list[int]
  init: Any
  dtype: jnp.dtype
  collections: Any
  tensor_split_dims_mapping: list[int]


def sharded_chain(
    *args: Union[optax.GradientTransformation, ShardedGradientTransformation],
) -> ShardedGradientTransformation:
  """Chain as in praxis.optimizers.sharded_chain."""

  def init_fn(params):
    return tuple(fn.init(params) for fn in args)

  def update_fn(updates, state, params=None):
    if len(args) != len(state):
      raise ValueError(
          'The number of updates and states has to be the same in '
          f'sharded chain. got {len(args)=}, {len(state)=}'
      )

    new_state = []
    for s, fn in zip(state, args):
      updates, new_s = fn.update(updates, s, params)
      # Some of the new states may have None instead of optax.MaskedNode.
      new_s = jax.tree.map(
          lambda x: optax.MaskedNode() if x is None else x,
          new_s,
          is_leaf=lambda x: x is None,
      )
      new_state.append(new_s)
    return updates, tuple(new_state)

  def init_partition_spec_fn(mdl_vars):
    partition_specs = []
    for fn in args:
      init_partition_spec = getattr(fn, 'init_partition_spec', None)
      if callable(init_partition_spec):
        nmap = init_partition_spec(mdl_vars)
        partition_specs.append(nmap)
      else:
        # Raise ValueError as we are attempting to sharded_chain an optimizer
        # that does not have an `init_partition_spec` method defined.
        raise ValueError(
            'Attempting to use an optimizer in sharded_chain that '
            'does not have an init_partition_spec.'
        )
    return optax.MaskedState(inner_state=tuple(partition_specs))

  return ShardedGradientTransformation(
      init=init_fn, update=update_fn, init_partition_spec=init_partition_spec_fn
  )
