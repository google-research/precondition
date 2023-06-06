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

"""Smoke tests for tearfree.

The smoke test uses CPU-based sharding to verify that, under a variety of
settings, (1) the optimizer results in finite, not-nan gradients and (2)
distributed computation options don't change the math.
"""

import copy
from typing import Sequence, Union

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
from jax import numpy as jnp
import numpy as np
import optax
from precondition.tearfree import grafting
from precondition.tearfree import momentum
from precondition.tearfree import optimizer
from precondition.tearfree import second_order
from precondition.tearfree import shampoo


def _make_distributed_equality_cases() -> list[dict[str, ...]]:
  """Make test cases of options for optimizer checks."""
  cases = []

  # Basic options exercise all of shampoo, grafting after the first step.
  basic_options = optimizer.TearfreeOptions(
      grafting_options=grafting.Options(
          grafting_type=grafting.GraftingType.RMSPROP,
          second_moment_decay=0.9,
          epsilon=1e-5,
          start_preconditioning_step=1,
          skip_preconditioning_any_dim_gt=4096,
          skip_preconditioning_rank1=False,
      ),
      second_order_options=second_order.Options(
          second_order_type=second_order.SecondOrderType.SHAMPOO,
          shampoo_options=shampoo.Options(
              block_size=1024,
              update_preconditioners_freq=1,
              update_statistics_freq=1,
              second_moment_decay=0.9,
          ),
          merge_dims=4096,
      ),
      momentum_options=momentum.Options(
          ema=True,
          nesterov=True,
          momentum_decay=0.5,
          weight_decay=0.0,
          weight_decay_after_momentum=True,
      ),
  )
  basic_case = {
      'testcase_name': 'basic',
      'nsteps': 3,
      'options': basic_options,
      'lr': 0.1,
      'shape': (4,),
  }
  cases.append(basic_case)

  case = copy.deepcopy(basic_case)
  case['lr'] = lambda x: 0.1 / (x + 1)
  case['testcase_name'] = 'schedule'
  cases.append(case)

  # Need to test we at least parallelize the identical-to-tensor shapes
  # without any blocks.
  # Additional variants:
  # wd
  # wd with decay before momentum
  # grid of nesterov/ema
  # exercise merge dims 2d doing a merge
  # exercise merge dims 3d with only one thing merged
  # skip preconditioning any dim gt activating
  # skip preconditioning any dim gt rank1 activating
  # update stats/precond every 2 (6 steps)
  # update stats/precond every 2/4 (6 steps)

  # Test block-wise parallelism for Shampoo

  return cases


class OptimizerSmokeTest(parameterized.TestCase):
  """Basic test for optimizer configurations."""

  def _unroll(self, options, shape, transform=None, lr=0.1, n=4):
    """Generate states and grad updates n times."""
    rng = jax.random.PRNGKey(0)
    params = jnp.zeros(shape)
    grads = jax.random.normal(rng, (n, *shape))

    if transform is not None:
      params = transform(params)
      grads = jnp.stack([transform(g) for g in grads])

    tx = optimizer.tearfree(lr, options)

    init = tx.init(params)

    def reduce(state, grad):
      new_grad, new_state = tx.update(grad, state, params)
      return new_state, new_grad

    _, out_grads = jax.lax.scan(reduce, init, grads)
    return out_grads

  @parameterized.named_parameters(_make_distributed_equality_cases())
  def test_distributed_equality(
      self,
      options: optimizer.TearfreeOptions,
      shape: Sequence[int],
      lr: Union[float, optax.Schedule],
      nsteps: int,
  ) -> None:
    single_core = self._unroll(options, shape, lr=lr, n=nsteps)
    multi_core = self._unroll(options, shape, lr=lr, n=nsteps)

    chex.assert_tree_all_finite(single_core)
    np.testing.assert_allclose(single_core, multi_core)


if __name__ == '__main__':
  absltest.main()
