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

"""Shampoo preconditioners must not depend on unrelated block scales."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from precondition.tearfree import shampoo


class BlockLocalCutoffTest(parameterized.TestCase):

  @parameterized.product(p=(2, 4, 6), rotated=(False, True))
  def test_inverse_roots_match_independent_matrices(self, p, rotated):
    rotation = np.array([[0.6, -0.8], [0.8, 0.6]]) if rotated else np.eye(2)
    eigenvalues = np.array([1.0, 4.0])
    scales = (1e-8, 1.0, 1e8)
    covariances = jnp.asarray(
        [(rotation * (scale * eigenvalues)) @ rotation.T for scale in scales]
    )
    expected = np.array(
        [
            (rotation * (scale * eigenvalues) ** (-1.0 / p)) @ rotation.T
            for scale in scales
        ]
    )
    compute = lambda cov: shampoo._pth_inv_root(p, cov)
    for fn in (compute, jax.jit(compute)):
      result = fn(covariances)
      np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-6)
      independent = jax.vmap(lambda cov: compute(cov[None])[0])(covariances)
      np.testing.assert_allclose(result, independent, rtol=2e-5, atol=1e-6)

  @parameterized.parameters(2, 4)
  def test_singular_and_zero_blocks_preserve_cutoff(self, p):
    eigenvalues = np.array([[1.0, 1e-8, 0.0], [1e8, 1.0, 0.0], [0.0, 0.0, 0.0]])
    covariances = jnp.asarray([np.diag(row) for row in eigenvalues])
    expected = np.zeros_like(eigenvalues)
    keep = eigenvalues > 1e-6 * eigenvalues.max(axis=-1, keepdims=True)
    expected[keep] = eigenvalues[keep] ** (-1.0 / p)
    result = jax.jit(lambda cov: shampoo._pth_inv_root(p, cov))(covariances)
    np.testing.assert_allclose(
        result, [np.diag(row) for row in expected], rtol=2e-5, atol=1e-7
    )
    self.assertTrue(np.isfinite(result).all())

  @parameterized.product(decay=(0.0, 0.9, 1.0), axis=(0, 1))
  def test_blocked_updates_match_separate_parameters(self, decay, axis):
    tx = shampoo.apply(
        shampoo.Options(
            block_size=2,
            second_moment_decay=decay,
            update_preconditioners_freq=2,
        )
    )
    small = jnp.diag(jnp.array([1.0, 2.0]))
    large = small * 1e4
    combined = jnp.concatenate([small, large], axis=axis)
    combined_state = tx.init(combined)
    separate = {'small': small, 'large': large}
    separate_state = tx.init(separate)
    for step in range(4):
      factor = 1.0 + 0.25 * step
      result, combined_state = jax.jit(tx.update)(
          factor * combined, combined_state
      )
      wanted, separate_state = jax.jit(tx.update)(
          jax.tree.map(lambda x: factor * x, separate), separate_state
      )
      expected = jnp.concatenate([wanted['small'], wanted['large']], axis=axis)
      np.testing.assert_allclose(result, expected, rtol=2e-5, atol=1e-6)


if __name__ == '__main__':
  absltest.main()
