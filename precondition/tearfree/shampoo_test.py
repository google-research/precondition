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

"""Tests for momentum implementation."""

import itertools
from typing import Sequence

from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from precondition.tearfree import shampoo


def _make_invalid_cases() -> Sequence[dict[str, ...]]:
  """Generate invalid cases which should throw."""
  return [
      {
          'testcase_name': 'block_size0',
          'invalid_options': shampoo.Options(
              block_size=0,
          ),
      },
      {
          'testcase_name': 'precond0',
          'invalid_options': shampoo.Options(
              update_preconditioners_freq=0,
          ),
      },
      {
          'testcase_name': 'stats0',
          'invalid_options': shampoo.Options(
              update_statistics_freq=0,
          ),
      },
      {
          'testcase_name': 'decay_neg',
          'invalid_options': shampoo.Options(
              second_moment_decay=-0.1,
          ),
      },
      {
          'testcase_name': 'decay_large',
          'invalid_options': shampoo.Options(
              second_moment_decay=1.1,
          ),
      },
      {
          'testcase_name': 'block_size1',
          'invalid_options': shampoo.Options(
              block_size=1,
          ),
      },
  ]


def _make_blockify_deblockify_cases() -> Sequence[dict[str, ...]]:
  shapes_blocks = [
      (tuple(), 2, 'scalar'),
      ((5,), 6, '1d_0large'),
      ((5,), 5, '1d_1large'),
      ((4,), 2, '1d_1large_moreblocks'),
      ((2, 3), 6, '2d_0large'),
      ((2, 3), 3, '2d_1large'),
      ((2, 2), 2, '2d_2large'),
      ((4, 4), 2, '2d_2large_moreblocks'),
      ((2, 3, 3, 2), 4, 'highdim_0large'),
      ((2, 3, 2, 2), 3, 'highdim_1large'),
      ((2, 2 * 3, 2, 2), 3, 'highdim_1large_moreblocks'),
      ((2, 3, 3, 2), 3, 'highdim_2large_together'),
      ((2, 3, 2, 3), 3, 'highdim_2large_separate'),
  ]

  cases = []
  for shape, block_size, name in shapes_blocks:
    cases.append(
        dict(
            shape=shape,
            block_size=block_size,
            testcase_name=name,
        )
    )
  return cases


class ShampooTest(parameterized.TestCase):
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
    tx = shampoo.apply(options)
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
      self._unroll(shampoo.Options(), 1, shape)

  def test_scalars(self):
    """Validate scalar parameters aren't preconditioned."""
    grads, out_grads = self._unroll(shampoo.Options(), 2, tuple())
    np.testing.assert_allclose(grads, out_grads)

  def _root(self, x, p):
    """Computes the matrix root x**(-1/(2*p))."""
    return shampoo._pth_inv_root(p * 2, x[np.newaxis, ...])[0]

  @parameterized.parameters(1, 2)
  def test_basic(self, ndim):
    """Check basic numerical example without blocking or decay."""
    options = shampoo.Options(second_moment_decay=1.0)
    shape = (2,) * ndim
    nsteps = 2
    grads, out_grads = self._unroll(options, 2, shape)

    l, r = 0, 0
    for i in range(nsteps):
      if len(shape) == 1:
        l += np.multiply.outer(grads[i], grads[i])
      elif len(shape) == 2:
        l += grads[i].dot(grads[i].T)
        r += grads[i].T.dot(grads[i])

      pl, pr = self._root(l, len(shape)), r
      if len(shape) == 2:
        pr = self._root(r, len(shape))

      pg = pl.dot(grads[i])
      if len(shape) == 2:
        pg = pg.dot(pr)

      np.testing.assert_allclose(pg, out_grads[i], rtol=1e-3)

  def test_basic_block(self):
    """Check basic numerical example with blocking."""
    options = shampoo.Options(second_moment_decay=1.0, block_size=2)
    shape = (4,)
    nsteps = 2

    # Don't use unroll here to allow state-printing.
    rng = jax.random.PRNGKey(0)
    params = jnp.zeros(shape)
    grads = jax.random.normal(rng, (nsteps, *shape))

    tx = shampoo.apply(options)
    state = tx.init(params)
    logging.info('init state: %s', state)

    b0, b1 = 0, 0
    for i in range(nsteps):
      out_grad, state = tx.update(grads[i], state, params)
      logging.info('state @ %s: %s', i, state)
      g0, g1 = grads[i][:2], grads[i][2:]
      b0 += np.multiply.outer(g0, g0)
      b1 += np.multiply.outer(g1, g1)
      p0, p1 = self._root(b0, len(shape)), self._root(b1, len(shape))
      logging.info('g0 %s g1 %s', g0, g1)
      logging.info('b0 %s b1 %s', b0, b1)
      logging.info('p0 %s p1 %s', p0, p1)
      pg = np.concatenate([p0.dot(g0), p1.dot(g1)], axis=0)
      np.testing.assert_allclose(pg, out_grad, rtol=1e-3)

  @parameterized.named_parameters(_make_invalid_cases())
  def test_invalid(self, invalid_options):
    with self.assertRaises(ValueError):
      shampoo.apply(invalid_options)

  @parameterized.named_parameters(_make_blockify_deblockify_cases())
  def test_blockify_deblockify(self, shape, block_size):
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, shape)
    options = shampoo.Options(block_size=block_size)
    meta = shampoo._blocks_metadata(options, x.shape, debug='')
    bx = shampoo._blockify(x, meta)
    dx = shampoo._deblockify(bx, meta)
    self.assertSequenceEqual(dx.shape, x.shape)
    np.testing.assert_array_equal(x, dx)

  @parameterized.parameters(
      [
          {'decay': d, 'last': b}
          for d, b in itertools.product([0, 0.8], [False, True])
      ]
  )
  def test_basic_ema(self, decay, last):
    """Tests EMA accumulation in stats."""
    z = jnp.zeros((2,))
    g = jnp.array([0.5, -0.5])

    if last:
      seq = jnp.stack([z, z, g])
      one = jnp.stack([g])
      expected_decay = 1 - decay
    else:
      seq = jnp.stack([g, z, z, g])
      one = jnp.stack([g])
      expected_decay = (1 - decay) * (decay**3 + 1)

    decayed = shampoo.Options(second_moment_decay=decay)
    no_decay = shampoo.Options(second_moment_decay=1.0)

    last = self._unroll_concrete(decayed, z, seq)[1][-1]
    last_no_decay = self._unroll_concrete(no_decay, z, one)[1][-1]
    last_no_decay /= np.sqrt(expected_decay)
    np.testing.assert_allclose(last, last_no_decay, rtol=1e-3)

  @parameterized.named_parameters(_make_blockify_deblockify_cases())
  def test_blocks_equality(self, shape, block_size):
    rng = jax.random.PRNGKey(0)
    nsteps = 3
    grads = jax.random.normal(rng, (nsteps, *shape))
    options = shampoo.Options(block_size=block_size)

    meta = shampoo._blocks_metadata(options, shape, debug='')
    grads_for_each_block = [[] for _ in range(meta.num_blocks)]
    for grad in grads:
      bgrad = shampoo._blockify(grad, meta)
      for i in range(meta.num_blocks):
        grads_for_each_block[i].append(jnp.take(bgrad, i, meta.blocks_axis))
    last_grad = []
    unblocked_options = shampoo.Options(block_size=1 + block_size)
    for block in grads_for_each_block:
      block = jnp.stack(block)
      block_grads, block_out_grads = self._unroll_concrete(
          unblocked_options, block[0], block
      )
      del block_grads
      last_grad.append(block_out_grads[-1])

    expected = jnp.stack(last_grad, axis=meta.blocks_axis)
    expected = shampoo._deblockify(expected, meta)
    actual = self._unroll_concrete(options, grads[0], grads)[1]
    np.testing.assert_allclose(expected, actual[-1])

  def test_stats_freq(self):
    rng = jax.random.PRNGKey(0)
    grads = jax.random.normal(rng, (9, 3))
    options = shampoo.Options(update_statistics_freq=3)
    _, out_grads = self._unroll_concrete(options, grads[0], grads)
    options = shampoo.Options(update_statistics_freq=1)
    _, out_grads_skip = self._unroll_concrete(options, grads[0], grads[::3])
    np.testing.assert_allclose(out_grads[::3], out_grads_skip)

  def test_precond_freq(self):
    rng = jax.random.PRNGKey(0)
    rng, key = jax.random.split(rng)
    freq = 5
    grads = jax.random.normal(rng, (freq * 2, 3))

    rng1, rng2 = jax.random.split(key, 2)
    seq1 = jnp.arange(freq, dtype=int)
    seq2 = jnp.copy(seq1)
    # Shuffle within groups of <freq>
    shuffled = jnp.take(grads, jnp.concatenate([seq1, seq2 + freq]), axis=0)

    grads = jnp.concatenate([jnp.zeros((1, 3)), grads])
    shuffled = jnp.concatenate([jnp.zeros((1, 3)), shuffled])

    options = shampoo.Options(
        update_preconditioners_freq=freq, second_moment_decay=1
    )
    _, out_grads = self._unroll_concrete(options, grads[0], grads)
    _, out_grads_shuf = self._unroll_concrete(options, grads[0], shuffled)
    np.testing.assert_allclose(out_grads, out_grads_shuf)

  def test_tree(self):
    shape = (3, 2)
    n = 4
    options = shampoo.Options()
    rng = jax.random.PRNGKey(0)
    params = jnp.zeros(shape)
    grads = jax.random.normal(rng, (n, *shape))
    _, out_grads = self._unroll_concrete(options, params, grads)

    params = {'w': [{'b': params}]}
    grads = {'w': [{'b': grads}]}
    _, actual_out_grads = self._unroll_concrete(options, params, grads)

    np.testing.assert_allclose(out_grads, actual_out_grads['w'][0]['b'])


if __name__ == '__main__':
  absltest.main()
