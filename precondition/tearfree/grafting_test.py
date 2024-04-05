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

"""Tests for grafting implementations."""

import functools
import itertools
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import optax
from precondition.tearfree import grafting
from precondition.tearfree import praxis_shim


def _minustwo() -> praxis_shim.ShardedGradientTransformation:
  """Generate a direction-reversing gradient transformation."""
  update = functools.partial(jax.tree_map, lambda x: -2 * x)
  return praxis_shim.ShardedGradientTransformation(
      lambda _: optax.EmptyState,
      lambda u, s, _: (update(u), s),
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
          'testcase_name': 'rmsprop_eps_neg',
          'invalid_options': grafting.Options(
              grafting.GraftingType.RMSPROP,
              epsilon=-1.0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'adafactor_0',
          'invalid_options': grafting.Options(
              grafting.GraftingType.ADAFACTOR,
              second_moment_decay=-1.0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'adafactor_neg',
          'invalid_options': grafting.Options(
              grafting.GraftingType.ADAFACTOR,
              second_moment_decay=-1.0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'adafactor_not_less_than_1',
          'invalid_options': grafting.Options(
              grafting.GraftingType.ADAFACTOR,
              second_moment_decay=1.0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'adafactor_eps_neg',
          'invalid_options': grafting.Options(
              grafting.GraftingType.ADAFACTOR,
              epsilon=-1.0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'adafactor_min_size_0',
          'invalid_options': grafting.Options(
              grafting.GraftingType.ADAFACTOR,
              min_dim_size_to_factor=0,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'adafactor_min_size_neg',
          'invalid_options': grafting.Options(
              grafting.GraftingType.ADAFACTOR,
              min_dim_size_to_factor=-1,
              start_preconditioning_step=0,
          ),
      },
      {
          'testcase_name': 'adafactor_clip_less_than_1',
          'invalid_options': grafting.Options(
              grafting.GraftingType.ADAFACTOR,
              clipping_threshold=0.5,
              start_preconditioning_step=0,
          ),
      },
  ]


class GraftingTest(parameterized.TestCase):
  """Basic test for grafting praxis_shim implementations."""

  def _check_equal(self, expected_tx, actual_tx, nsteps, shape=(3,)):
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    params = jax.random.normal(key, shape)
    expected_state = expected_tx.init(params)
    actual_state = actual_tx.init(params)

    for i in range(nsteps):
      rng, key = jax.random.split(rng)
      grad = jax.random.normal(key, shape)
      expected_grad, expected_state = expected_tx.update(
          grad, expected_state, params
      )
      actual_grad, actual_state = actual_tx.update(grad, actual_state, params)
      np.testing.assert_allclose(expected_grad, actual_grad, err_msg=i)

  def test_no_graft(self):
    """Check that no graft behaves exactly as the base transform."""
    options = grafting.Options(
        grafting.GraftingType.NONE,
        0.0,
        start_preconditioning_step=0,
        skip_preconditioning_rank1=False,
    )
    grafted = grafting.graft(options, _minustwo())
    nsteps = 4
    self._check_equal(_minustwo(), grafted, nsteps)

  def _check_norm_direction(
      self,
      norm_tx,
      direction_tx,
      actual_tx,
      nsteps,
      start_precond_step,
      shape=(3,),
  ):
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    params = jax.random.normal(key, shape)
    state = actual_tx.init(params)
    norm_state = norm_tx.init(params)
    direction_state = norm_tx.init(params)

    for i in range(nsteps):
      rng, key = jax.random.split(rng)
      grad = jax.random.normal(key, shape)
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
        np.testing.assert_allclose(
            direction_grad_unit, actual_grad_unit, rtol=1e-6
        )
        np.testing.assert_allclose(actual_norm, norm_norm, rtol=1e-6)
      else:
        np.testing.assert_allclose(norm_grad, actual_grad)

  def _norm_tx(self, options):
    if options.grafting_type == grafting.GraftingType.SGD:
      return grafting._sgd()
    if options.grafting_type == grafting.GraftingType.RMSPROP:
      return grafting._rmsprop(options)
    if options.grafting_type == grafting.GraftingType.ADAFACTOR:
      return grafting._adafactor(options)
    raise ValueError('unsupported grafting type ' + str(options.grafting_type))

  @parameterized.parameters(
      itertools.product(
          [0, 1, 2], ['sgd', 'rmsprop', 'adafactor'], [(3,), (3, 2)]
      )
  )
  def test_norm_direction(self, step, graft, shape):
    """Validate initial graft update, then switch to its norm."""
    options = grafting.Options(
        grafting.GraftingType(graft),
        0.9 if (graft == 'rmsprop' or graft == 'adafactor') else 0.0,
        start_preconditioning_step=step,
        skip_preconditioning_rank1=len(shape) > 1,
        min_dim_size_to_factor=1
    )
    grafted = grafting.graft(options, _minustwo())
    nsteps = 4
    norm_tx = self._norm_tx(options)
    self._check_norm_direction(
        norm_tx, _minustwo(), grafted, nsteps, step, shape
    )

  @parameterized.parameters({'shape': s} for s in [tuple(), (3,), (5,), (5, 2)])
  def test_skip(self, shape):
    """Make sure we skip preconditioning if out-of-bounds."""
    options = grafting.Options(
        start_preconditioning_step=2,
        skip_preconditioning_any_dim_gt=4,
        skip_preconditioning_rank1=True,
    )
    grafted = grafting.graft(options, _minustwo())
    nsteps = 4
    norm_tx = self._norm_tx(options)
    self._check_equal(norm_tx, grafted, nsteps, shape)

  @parameterized.named_parameters(_make_invalid_cases())
  def test_invalid(self, invalid_options):
    with self.assertRaises(ValueError):
      grafting.graft(invalid_options, _minustwo())


if __name__ == '__main__':
  absltest.main()
