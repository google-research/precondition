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

import itertools
from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from precondition.tearfree import shampoo
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
      self._unroll(sketchy.apply(sketchy.Options()), 1, shape)

  @parameterized.named_parameters(_make_invalid_cases())
  def test_invalid(self, invalid_options):
    with self.assertRaises(ValueError):
      sketchy.apply(invalid_options)

  def _make_null_state(self, d, k) -> sketchy._AxisState:
    return sketchy._init(
        sketchy.Options(rank=k), jnp.zeros((d,))
    ).sketches.axes[0]

  def _make_eye_state(self, d, eigs, tail, ndim) -> sketchy._AxisState:
    k = len(eigs)
    state = self._make_null_state(d, k)
    state = state._replace(eigvecs=jnp.eye(d, k))
    state = state._replace(eigvals=state.eigvals + jnp.asarray(eigs))
    state = state._replace(tail=tail)
    if tail > 0:
      state._replace(inv_tail=tail ** (-1 / (2 * ndim)))
    mask = state.eigvals > 0
    ie = jnp.where(mask, (state.tail + state.eigvals**2), 1.0) ** (
        -1 / (2 * ndim)
    )
    ie *= mask
    state._replace(inv_eigvals=ie)
    return state

  def _no_decay_options(self, sketch_size, epsilon=0.0):
    return sketchy.Options(
        rank=sketch_size, second_moment_decay=1, epsilon=epsilon
    )

  @parameterized.parameters(range(1, 5))
  def test_dynamic_exponent(self, ndim):
    """Test that exponent for various gradient ndim is correct."""
    size = 4
    prev = self._make_eye_state(size, [0], 0.0, ndim)
    grad = np.zeros([size] * ndim, np.float32)
    grad[(0,) * ndim] = 2**ndim
    ret = sketchy._update_axis(self._no_decay_options(1), 0, '', grad, prev)
    self.assertAlmostEqual(ret.inv_eigvals, 1 / 2, delta=1e-6)

    prev = self._make_eye_state(size, [2**ndim], 0.0, ndim)
    grad = np.zeros([size] * ndim, np.float32)
    ret = sketchy._update_axis(self._no_decay_options(1), 0, '', grad, prev)
    self.assertAlmostEqual(ret.inv_eigvals, 1 / 2, delta=1e-6)

  def test_epsilon(self):
    """Test that epsilon is properly calculated."""
    size = 4
    ndim = 2
    prev = self._make_eye_state(size, [0], 4, ndim)
    grad = np.zeros([size] * ndim, np.float32)
    grad[(0,) * ndim] = 2
    options = self._no_decay_options(1, epsilon=1e-3)
    ret = sketchy._update_axis(options, 0, '', grad, prev)
    self.assertAlmostEqual(
        ret.inv_eigvals[0], ((4 + 4) * 1.001) ** (-1 / 4), delta=1e-3, msg=ret
    )
    self.assertAlmostEqual(ret.inv_tail, (4 * 1.001) ** (-1 / 4), delta=1e-3)
    options.relative_epsilon = False
    ret = sketchy._update_axis(options, 0, '', grad, prev)
    self.assertAlmostEqual(
        ret.inv_eigvals[0], (4 + 4 + 0.001) ** (-1 / 4), delta=1e-6
    )
    self.assertAlmostEqual(ret.inv_tail, (4 + 0.001) ** (-1 / 4), delta=1e-3)

  def _make_rand_state(self, size, eigs, tail, ndim):
    rng = np.random.default_rng(1234)
    b = rng.standard_normal(size=[size, size])
    b = b.dot(b.T)
    _, v = np.linalg.eigh(b)
    state = self._make_eye_state(size, eigs, tail, ndim)
    state = state._replace(eigvecs=v[:, : len(eigs)])
    return state

  # pylint: disable=g-long-lambda
  def test_realloc(self):
    """Test the memory reallocation functions properly."""
    dim, nsteps = 8, 3
    memory_dict = {
        'a': [2],
        'b': [[6], [8]],
        'c': {'d': [4], 'e': [8]},
    }
    tx = sketchy.apply(sketchy.Options(memory_alloc=memory_dict))
    shape = jax.tree_map(
        lambda x: (dim,),
        memory_dict,
        is_leaf=lambda x: isinstance(x, list)
        and all(not isinstance(y, list) for y in x),
    )
    grads_tree, updates = self._unroll(tx, nsteps, shape, None, True)
    emw_run = jax.tree_map(
        lambda k, sp, grad: self._unroll(
            tx=sketchy.apply(sketchy.Options(rank=k[0])),
            n=nsteps,
            shape=sp,
            grads=grad,
        ),
        memory_dict,
        shape,
        grads_tree,
        is_leaf=lambda x: isinstance(x, list)
        and all(not isinstance(y, list) for y in x),
    )
    jax.tree_map(np.testing.assert_allclose, updates, emw_run)

  # test covariance-adding equality from FD
  # with rand initial state, and with zero
  #
  # Do it under ndim 1 2 or 3 (choose random axis for higher dims)

  @parameterized.parameters(
      itertools.product(
          [1, 2, 3],
          [0.1, 0.9, 1.0],
          ['zero', 'id', 'rand'],
          [0, 1],
          [False, True],
      )
  )
  def test_basic(self, ndim, decay, init, tail, last_axis):
    """Validate low rank returned matrix."""
    d = 3
    k = 2
    rng = np.random.default_rng(1234)

    # Make other dims slightly larger
    shape = [d + i for i in range(ndim)]
    if last_axis:
      shape = shape[::-1]
    grad = rng.standard_normal(size=shape)

    if last_axis:
      grad_2d = grad.reshape(-1, d)
      added_cov = grad_2d.T.dot(grad_2d)
    else:
      grad_2d = grad.reshape(d, -1)
      added_cov = grad_2d.dot(grad_2d.T)
    top_added_eig = np.linalg.eigvalsh(added_cov).max()
    # Test out one eig above, one below.
    eigs = np.array([top_added_eig * 4, top_added_eig / 4])

    if init == 'zero':
      prev = self._make_null_state(d, k)
    elif init == 'id':
      prev = self._make_eye_state(d, eigs, tail, ndim)
    else:
      assert init == 'rand', init
      prev = self._make_rand_state(d, eigs, tail, ndim)

    options = sketchy.Options(
        second_moment_decay=decay,
        rank=k,
        epsilon=0.0,
    )
    dim = ndim - 1 if last_axis else 0
    updated = sketchy._update_axis(options, dim, '', grad, prev)

    if updated.tail > 0:
      self.assertAlmostEqual(
          updated.tail ** (-1 / (2 * ndim)), updated.inv_tail
      )
    else:
      self.assertAlmostEqual(updated.inv_tail, 0)
      self.assertAlmostEqual(updated.tail, 0)

    ie = updated.inv_eigvals
    e = updated.eigvals**2 + updated.tail
    mask = updated.eigvals > 0
    expected_ie = mask * np.where(mask, e, 1.0) ** (-1 / (2 * ndim))
    delta = 1e-5 * min(expected_ie.max(), ie.max())
    self.assertSequenceAlmostEqual(expected_ie, ie, delta=delta)

    def _make_cov(sketch: sketchy._AxisState, add_tail=True):
      # Note eigvals refer to the *root* singular values, so squaring as
      # we do below recovers covariance.
      eigvals = np.sqrt(add_tail * sketch.tail + np.square(sketch.eigvals))
      half = sketch.eigvecs * eigvals
      complement = np.eye(d) - sketch.eigvecs.dot(sketch.eigvecs.T)
      tail = complement * sketch.tail if add_tail else 0.0
      return half.dot(half.T) + tail

    self.assertGreaterEqual(updated.tail, prev.tail * decay)

    prev_cov = _make_cov(prev)
    new_cov = _make_cov(updated)
    pd_eigs = np.linalg.eigvalsh(new_cov - decay * prev_cov)
    # Validate positive definiteness up to numerical error.
    self.assertGreaterEqual(pd_eigs.min(), -pd_eigs.max() * 1e-4)

    prev_no_tail = _make_cov(prev, add_tail=False)
    w2, v2 = np.linalg.eigh(decay * prev_no_tail + added_cov)
    w2 = np.maximum(0, w2 - w2[d - k - 1])
    half = v2 * jnp.sqrt(w2)
    expected_cov = half.dot(half.T)
    actual_cov = _make_cov(updated, add_tail=False)
    np.testing.assert_allclose(expected_cov, actual_cov, rtol=1e-3)

  def _unroll(self, tx, n, shape, grads=None, return_grads=False):
    """Generate states and grad updates n times."""
    rng = jax.random.PRNGKey(0)
    params = jax.tree_map(
        jnp.zeros,
        shape,
        is_leaf=lambda x: isinstance(x, tuple)
        and all(isinstance(y, int) for y in x),
    )
    if grads is None:
      grads = jax.tree_map(
          lambda sp: jax.random.normal(rng, (n, *sp)),
          shape,
          is_leaf=lambda x: isinstance(x, tuple)
          and all(isinstance(y, int) for y in x),
      )

    init = tx.init(params)

    def reduce(state, grad):
      new_grad, new_state = tx.update(grad, state, params)
      return new_state, new_grad

    _, out_grads = jax.lax.scan(reduce, init, grads)
    if return_grads:
      return grads, out_grads
    return out_grads

  def test_reduction_to_shampoo(self):
    tx = sketchy.apply(sketchy.Options(second_moment_decay=0.99, epsilon=0.0))
    shampoo_tx = shampoo.apply(shampoo.Options(second_moment_decay=0.99))
    # Choose a shape well below sketchy rank & shampoo block size.
    shape = (4, 5)
    nsteps = 3
    sketchy_run = self._unroll(tx, nsteps, shape)
    # Shampoo 2nd moment is computed as (1 - decay) * update + decay * update
    # so we must adjust the preconditioned grad by a factor sqrt(1/(1-decay)).
    shampoo_run = self._unroll(shampoo_tx, nsteps, shape) / 10
    np.testing.assert_allclose(shampoo_run, sketchy_run, rtol=3e-3, atol=2e-4)


if __name__ == '__main__':
  absltest.main()
