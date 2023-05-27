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

"""Tests for grafting implementations."""

import itertools
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import optax
from precondition.tearfree import grafting
from precondition.tearfree import praxis_shim


def _minustwo() -> praxis_shim.ShardedGradientTransformation:
  """Generate a direction-reversing gradient transformation."""
  return praxis_shim.ShardedGradientTransformation(
      lambda _: optax.EmptyState,
      lambda u, s, _: (-2 * u, s),
      optax.EmptyState,
  )


def _make_invalid_cases() -> Sequence[dict[str, ...]]:
  """Generate invalid cases which should throw."""
  return [
      {
          'testcase_name': 'rmsprop_0',
          'invalid_options': grafting.Options(
              grafting.GraftingType.RMSPROP,
              second_moment_decay=0.0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'rmsprop_neg',
          'invalid_options': grafting.Options(
              grafting.GraftingType.RMSPROP,
              second_moment_decay=-1.0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'none_nnz',
          'invalid_options': grafting.Options(
              grafting.GraftingType.NONE,
              second_moment_decay=0.99,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'sgd_nnz',
          'invalid_options': grafting.Options(
              grafting.GraftingType.SGD,
              second_moment_decay=0.99,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'nonegraft_start',
          'invalid_options': grafting.Options(
              grafting.GraftingType.NONE,
              second_moment_decay=0.0,
              start_preconditioning_step=1,
          ),
      },
  ]


class GraftingTest(parameterized.TestCase):
  """Basic test for grafting praxis_shim implementations."""

  def _check_equal(self, expected_tx, actual_tx, nsteps):
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    params = jax.random.normal(key, (3,))
    expected_state = expected_tx.init(params)
    actual_state = actual_tx.init(params)

    for i in range(nsteps):
      rng, key = jax.random.split(rng)
      grad = jax.random.normal(key, (3,))
      expected_grad, expected_state = expected_tx.update(
          grad, expected_state, params
      )
      actual_grad, actual_state = actual_tx.update(grad, actual_state, params)
      self.assertSequenceAlmostEqual(expected_grad, actual_grad, msg=i)

  def test_no_graft(self):
    """Check that no graft behaves exactly as the base transform."""
    options = grafting.Options(
        grafting.GraftingType.NONE, 0.0, start_preconditioning_step=0
    )
    grafted = grafting.graft(options, _minustwo())
    nsteps = 4
    self._check_equal(_minustwo(), grafted, nsteps)

  def _check_norm_direction(
      self, norm_tx, direction_tx, actual_tx, nsteps, start_precond_step
  ):
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    params = jax.random.normal(key, (3,))
    state = actual_tx.init(params)
    norm_state = norm_tx.init(params)
    direction_state = norm_tx.init(params)

    for i in range(nsteps):
      rng, key = jax.random.split(rng)
      grad = jax.random.normal(key, (3,))
      actual_grad, state = actual_tx.update(grad, state, params)

      norm_grad, norm_state = norm_tx.update(grad, norm_state, params)
      direction_grad, direction_state = direction_tx.update(
          grad, direction_state, params
      )

      if i >= start_precond_step:
        direction_norm = jnp.linalg.norm(direction_grad)
        actual_norm = jnp.linalg.norm(actual_grad)
        norm_norm = jnp.linalg.norm(norm_grad)
        direction_grad_unit = direction_grad / direction_norm
        actual_grad_unit = actual_grad / actual_norm
        self.assertSequenceAlmostEqual(
            direction_grad_unit, actual_grad_unit, delta=1e-6
        )
        self.assertAlmostEqual(actual_norm, norm_norm, delta=1e-6)
      else:
        self.assertSequenceAlmostEqual(norm_grad, actual_grad)

  def _norm_tx(self, options):
    if options.grafting_type == grafting.GraftingType.SGD:
      return grafting._sgd()
    if options.grafting_type == grafting.GraftingType.RMSPROP:
      return grafting._rmsprop(options.second_moment_decay)
    raise ValueError('unsupported grafting type ' + str(options.grafting_type))

  @parameterized.parameters(itertools.product([0, 1, 2], ['sgd', 'rmsprop']))
  def test_norm_direction(self, step, graft):
    """Validate initial graft update, then switch to its norm."""
    options = grafting.Options(
        grafting.GraftingType(graft),
        0.9 if graft == 'rmsprop' else 0.0,
        start_preconditioning_step=step,
    )
    grafted = grafting.graft(options, _minustwo())
    nsteps = 4
    norm_tx = self._norm_tx(options)
    self._check_norm_direction(norm_tx, _minustwo(), grafted, nsteps, step)

  @parameterized.named_parameters(_make_invalid_cases())
  def test_invalid(self, invalid_options):
    with self.assertRaises(ValueError):
      grafting.graft(invalid_options, _minustwo())


if __name__ == '__main__':
  absltest.main()
