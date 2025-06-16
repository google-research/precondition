# Copyright 2025 The precondition Authors.
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

import itertools
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import optax
from precondition.tearfree import momentum

jax.config.update('jax_threefry_partitionable', False)


def _make_no_state_cases() -> Sequence[dict[str, ...]]:
  bools = [False, True]
  cases = []
  for ema, nesterov, wd, wd_after in itertools.product(
      bools, bools, [0.0, 0.9], bools
  ):
    momentum_decay = 0.0
    options = momentum.Options(
        ema,
        nesterov,
        momentum_decay,
        wd,
        wd_after,
    )
    cases.append({'options': options})
  return cases


def _make_invalid_cases() -> Sequence[dict[str, ...]]:
  """Generate invalid cases which should throw."""
  return [
      {
          'testcase_name': 'momentum_neg',
          'invalid_options': momentum.Options(
              momentum_decay=-1.0,
          ),
      },
      {
          'testcase_name': 'wd_neg',
          'invalid_options': momentum.Options(
              weight_decay=-0.1,
          ),
      },
      {
          'testcase_name': 'momentum_large',
          'invalid_options': momentum.Options(
              momentum_decay=1.1,
          ),
      },
  ]


class MomentumTest(parameterized.TestCase):
  """Basic test for momentum implementation."""

  def _unroll(self, tx, n, extract=False, wd=0):
    """Generate states and grad updates n times."""
    rng = jax.random.PRNGKey(0)
    params = jnp.ones((3,))
    grads = jax.random.normal(rng, (n, 3)) + wd * params
    init = tx.init(params)

    def scan(state, grad):
      new_grad, new_state = tx.update(grad, state, params)
      return new_state, (new_state, new_grad)

    _, (states, out_grad) = jax.lax.scan(scan, init, grads)
    if not extract:
      return out_grad
    return self._extract_velocity(states), out_grad, grads

  def _check_equal(self, expected_tx, actual_tx, nsteps):
    expected_grads = self._unroll(expected_tx, nsteps)
    actual_grads = self._unroll(actual_tx, nsteps)
    np.testing.assert_allclose(expected_grads, actual_grads)

  @parameterized.parameters(0.1, 0.9, 0.99)
  def test_ema(self, decay):
    """Check that we simulate ema decay."""
    options = momentum.Options(ema=True, nesterov=False, momentum_decay=decay)
    nsteps = 4
    actual = momentum.apply(options)
    expected = optax.ema(decay, debias=False)
    self._check_equal(expected, actual, nsteps)

  def _extract_velocity(self, state):
    """Asserts only velocity state exists, extracts it."""
    flat = jax.tree_util.tree_flatten(state)[0]
    self.assertLen(flat, 1)
    return flat[0]

  @parameterized.parameters(itertools.product([False, True], repeat=2))
  def test_wd_before_momentum(self, ema, nesterov):
    options = momentum.Options(
        ema=ema,
        nesterov=nesterov,
        momentum_decay=0.9,
        weight_decay=0.0,
    )
    nsteps = 4
    tx = momentum.apply(options)
    expected_grads = self._unroll(tx, nsteps, wd=0.1)
    options = momentum.Options(
        ema=ema,
        nesterov=nesterov,
        momentum_decay=0.9,
        weight_decay=0.1,
        weight_decay_after_momentum=False,
    )
    tx = momentum.apply(options)
    actual_grads = self._unroll(tx, nsteps)
    np.testing.assert_allclose(expected_grads, actual_grads)

  @parameterized.parameters(itertools.product([False, True], repeat=2))
  def test_basic(self, ema, decay_after):
    wd = 0.1 if decay_after else 0.0
    if decay_after:
      return
    decay = 0.9
    options = momentum.Options(
        ema=ema,
        nesterov=True,
        momentum_decay=decay,
        weight_decay=wd,
        weight_decay_after_momentum=True,
    )
    tx = momentum.apply(options)
    v, g, ig = self._unroll(tx, 2, extract=True)

    ev = jnp.zeros((3,))
    factor = (1 - decay) if ema else 1.0
    ev += factor * ig[0]
    self.assertSequenceAlmostEqual(v[0], ev, msg=v)
    expected_grad = decay * ev + factor * ig[0]
    expected_grad += jnp.ones((3,)) * wd
    self.assertSequenceAlmostEqual(g[0], expected_grad)

    ev = ev * decay + factor * ig[1]
    self.assertSequenceAlmostEqual(v[1], ev, delta=1e-6)
    expected_grad = decay * ev + factor * ig[1]
    expected_grad += jnp.ones((3,)) * wd
    self.assertSequenceAlmostEqual(g[1], expected_grad, delta=1e-6)

  @parameterized.parameters(_make_no_state_cases())
  def test_no_state(self, options):
    """Ensure no state is created when decay is 0.0."""
    assert options.momentum_decay == 0.0
    tx = momentum.apply(options)
    state = tx.init(jnp.zeros((3,)))
    flat = jax.tree_util.tree_flatten(state)[0]
    self.assertEmpty(flat)

  @parameterized.named_parameters(_make_invalid_cases())
  def test_invalid(self, invalid_options):
    with self.assertRaises(ValueError):
      momentum.apply(invalid_options)


if __name__ == '__main__':
  absltest.main()
