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

"""Tests for momentum implementation."""

from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from precondition.tearfree import sketchy


def _make_invalid_cases() -> Sequence[dict[str, ...]]:
  """Generate invalid cases which should throw."""
  return [
      {
          'testcase_name': 'freq0',
          'invalid_options': sketchy.Options(
              update_freq=0,
          ),
      },
      {
          'testcase_name': 'decay_neg',
          'invalid_options': sketchy.Options(
              second_moment_decay=-0.1,
          ),
      },
      {
          'testcase_name': 'decay_large',
          'invalid_options': sketchy.Options(
              second_moment_decay=1.1,
          ),
      },
  ]


class SketchyTest(parameterized.TestCase):
  """Basic test for shampoo implementation."""

  def setUp(self):
    super().setUp()
    jax.config.update('jax_debug_nans', True)

  def _unroll(self, options, n, shape):
    """Generate states and grad updates n times."""
    rng = jax.random.PRNGKey(0)
    params = jnp.zeros(shape)
    grads = jax.random.normal(rng, (n, *shape))
    return self._unroll_concrete(options, params, grads)

  def _unroll_concrete(self, options, params, grads):
    """Unrolls with provided params and grads."""
    tx = sketchy.apply(options)
    init = tx.init(params)

    def reduce(state, grad):
      new_grad, new_state = tx.update(grad, state, params)
      return new_state, new_grad

    _, out_grads = jax.lax.scan(reduce, init, grads)
    return grads, out_grads

  @parameterized.parameters(
      {'shape': (1, 2, 1)},
      {'shape': (1, 1, 3, 1, 2, 1)},
      {'shape': (2, 1, 3, 2)},
      {'shape': (1, 1)},
      {'shape': (1,)},
  )
  def test_unit_dims_raise(self, shape):
    """Assert raises if unit dimensions are present."""
    with self.assertRaises(ValueError):
      self._unroll(sketchy.Options(), 1, shape)

  @parameterized.named_parameters(_make_invalid_cases())
  def test_invalid(self, invalid_options):
    with self.assertRaises(ValueError):
      sketchy.apply(invalid_options)

  # TODO(vladf): test reduction to shampoo in full rank


if __name__ == '__main__':
  absltest.main()
