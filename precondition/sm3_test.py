# Copyright 2026 The precondition Authors.
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

"""Tests for distributed_shampoo."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp

from precondition import sm3


class SM3Test(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (
        jnp.array([[0.5, 0.5], [0.5, 0.5]]))
    self.per_step_updates = (jnp.array([[0.1, -0.1], [0.01, 0.01]]))

  @chex.all_variants(with_pmap=False)
  def test_sm3_basic(self):
    params = self.init_params

    optim = sm3.sm3(0.1, 0.9, 0.999)
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    def _update(unused_batch):
      return transform_fn(self.per_step_updates, state, params)
    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    pmap_fn = jax.pmap(_update, axis_name='batch')

    updates, state = pmap_fn(jnp.array([1.0]))
    chex.assert_tree_all_finite((params, updates, state))


class SM3ScalarTest(parameterized.TestCase):

  @parameterized.product(
      compiled=[False, True],
      beta1=[0.0, 0.9],
      beta2=[0.999, 1.0],
      normalize_grads=[False, True],
  )
  def test_scalar_matches_length_one_parameter(
      self, compiled, beta1, beta2, normalize_grads
  ):
    params = {
        'scalar': jnp.array(2.0),
        'vector': jnp.array([1.0, -1.0]),
        'matrix': jnp.ones((2, 3)),
    }
    reference_params = dict(params, scalar=params['scalar'][None])
    optim = sm3.sm3(
        learning_rate=lambda step: 0.1 / (step + 1),
        beta1=beta1,
        beta2=beta2,
        weight_decay=0.05,
        normalize_grads=normalize_grads,
    )
    initialize = jax.jit(optim.init) if compiled else optim.init
    update = jax.jit(optim.update) if compiled else optim.update
    state = initialize(params)
    reference_state = initialize(reference_params)
    for step, value in enumerate([2.0, 0.0, -1.0, 0.5]):
      grads = {
          'scalar': jnp.array(value),
          'vector': jnp.array([value, -value]),
          'matrix': jnp.full((2, 3), value),
      }
      reference_grads = dict(grads, scalar=grads['scalar'][None])
      updates, state = update(grads, state, params)
      expected, reference_state = update(
          reference_grads, reference_state, reference_params
      )
      chex.assert_tree_all_finite((updates, state))
      chex.assert_trees_all_equal_shapes(updates, params)
      chex.assert_trees_all_close(
          jax.tree.map(jnp.atleast_1d, updates), expected
      )
      self.assertEqual(int(state.count), step + 1)
      params = jax.tree.map(lambda p, g: p + g, params, updates)
      reference_params = jax.tree.map(
          lambda p, g: p + g, reference_params, expected
      )

  def test_scalar_adagrad_recurrence(self):
    optim = sm3.sm3(0.1, beta1=0.0, beta2=1.0, diagonal_epsilon=0.25)
    param = jnp.array(1.0)
    state = optim.init(param)
    accumulator = 0.0
    for value in [2.0, 0.0, -1.0, 0.5]:
      grad = jnp.array(value)
      update, state = optim.update(grad, state, param)
      accumulator += value**2
      expected = -0.1 * value / (accumulator + 0.25) ** 0.5
      self.assertEqual(update.shape, ())
      self.assertAlmostEqual(float(update), expected, places=6)
      param += update


if __name__ == '__main__':
  absltest.main()
