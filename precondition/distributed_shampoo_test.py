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

"""Tests for distributed_shampoo."""

import functools
import itertools

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from precondition import distributed_shampoo
import scipy


class PaddingTest(parameterized.TestCase):

  def assertAllClose(self, x, y, atol=1e-5, rtol=1e-5):
    np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)

  @parameterized.named_parameters(
      {
          'testcase_name': 'NoPadding',
          'max_size': 3,
          'result': [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
      },
      {
          'testcase_name':
              'Padding',
          'max_size':
              5,
          'result': [[1., 1., 1., 0., 0.], [1., 1., 1., 0., 0.],
                     [1., 1., 1., 0., 0.], [0., 0., 0., 1., 0.],
                     [0., 0., 0., 0., 1.]],
      },
  )
  def test_pad_square_matrix(self, max_size, result):
    self.assertAllClose(
        distributed_shampoo.pad_square_matrix(
            mat=jnp.ones(shape=(3, 3), dtype=jnp.float32), max_size=max_size),
        jnp.asarray(result, dtype=jnp.float32))

  @parameterized.named_parameters(
      {
          'testcase_name': 'TooLarge',
          'shape': (3, 3),
          'max_size': 2
      },
      {
          'testcase_name': 'NotSquare',
          'shape': (3, 4),
          'max_size': 5
      },
  )
  def test_pad_square_matrix_error(self, shape, max_size):
    with self.assertRaises(ValueError):
      distributed_shampoo.pad_square_matrix(
          mat=jnp.ones(shape=shape), max_size=max_size)


def _pth_root_difference_cases():
  """Returns cases for _pth_root_difference() test."""
  cases = []
  # The test checks accuracy of
  # (w + a)^(-1/p) - (w + b)^(-1/p)
  # so generate corresponding parameters.
  p_vals = [2, 4, 6, 8]
  a_vals = b_vals = [1e-6, 1e-5, 0.0, 1.0]
  w_vals = [1e-6, 1e-5, 1.0, 1e3]
  for p, a, b, w in itertools.product(p_vals, a_vals, b_vals, w_vals):
    cases.append({'p': p, 'a': a, 'b': b, 'w': w})
  return cases


class DistributedShampooTest(chex.TestCase, parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.init_params = (jnp.array([[1., 3.],
                                   [2., 4.]]), jnp.array([[3., 4.], [3., 4.]]))
    self.per_step_updates = (jnp.array([[500., 5.], [500., 5.]]),
                             jnp.array([[300., 3.], [300., 3.]]))
    self.per_step_updates_custom_preconditioner = (self.per_step_updates,
                                                   (jnp.array([[200., 4.],
                                                               [200., 4.]]),
                                                    jnp.array([[600., 2.],
                                                               [600., 2.]])))
    self.rng = np.random.default_rng(1234)
    shape = ([2, 5], [6, 3])
    dt = self.init_params[0].dtype

    def make_shape(bigger_first_entry):
      x = tuple(self.rng.standard_normal(size=s) for s in shape)
      if bigger_first_entry:
        for xx in x:
          xx[..., 0] *= 100
      return tuple(jnp.array(xx).astype(dt) for xx in x)

    self.init_params_larger = make_shape(False)
    self.per_step_updates_larger = make_shape(True)

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters(
      {
          'testcase_name': 'default',
          'best_effort_memory_usage_reduction': True,
          'expected_value': -0.57,
      },
      {
          'testcase_name': 'default_nomerge',
          'best_effort_memory_usage_reduction': True,
          'merge_small_dims_block_size': 1,
          'expected_value': -0.57,
      },
      {
          'testcase_name': 'default_larger',
          'best_effort_memory_usage_reduction': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'default_larger_nomerge',
          'best_effort_memory_usage_reduction': True,
          'slightly_larger': True,
          'merge_small_dims_block_size': 1,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'materialize_statistics',
          'best_effort_memory_usage_reduction': True,
      },
      {
          'testcase_name': 'blocked_statistics',
          'best_effort_memory_usage_reduction': True,
      },
      {
          'testcase_name': 'default_quantized',
      },
      {
          'testcase_name': 'materialize_statistics_quantized',
      },
      {
          'testcase_name': 'blocked_statistics_quantized',
      },
      {
          'testcase_name': 'pos_compression_rank',
          'compression_rank': 1,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'pos_compression_rank_nomerge',
          'compression_rank': 1,
          'slightly_larger': True,
          'merge_small_dims_block_size': 1,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'neg_compression_rank',
          'compression_rank': -1,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'neg_compression_rank_nomerge',
          'compression_rank': -1,
          'slightly_larger': True,
          'merge_small_dims_block_size': 1,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'no_training_metrics',
          'generate_training_metrics': False,
      },
      {
          'testcase_name': 'larger_reuse',
          'best_effort_memory_usage_reduction': True,
          'reuse_preconditioner': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'larger_reuse_highmem',
          'best_effort_memory_usage_reduction': False,
          'reuse_preconditioner': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
      {
          'testcase_name': 'larger_reuse_highmem_nomerge',
          'best_effort_memory_usage_reduction': False,
          'merge_small_dims_block_size': 1,
          'reuse_preconditioner': True,
          'slightly_larger': True,
          'expected_value': -0.17019942,
      },
  )
  def test_distributed_shampoo(
      self,
      best_effort_memory_usage_reduction=False,
      compression_rank=0,
      merge_small_dims_block_size=4096,
      generate_training_metrics=True,
      slightly_larger=False,
      expected_value=None,
      reuse_preconditioner=False,
  ):
    params = self.init_params_larger if slightly_larger else self.init_params

    optim = distributed_shampoo.distributed_shampoo(
        0.1,
        32,
        batch_axis_name='batch',
        preconditioning_compute_steps=2,
        best_effort_memory_usage_reduction=best_effort_memory_usage_reduction,
        relative_matrix_epsilon=True,
        compression_rank=compression_rank,
        merge_small_dims_block_size=merge_small_dims_block_size,
        generate_training_metrics=generate_training_metrics,
        reuse_preconditioner=reuse_preconditioner,
    )
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    if slightly_larger:
      updates = self.per_step_updates_larger
    else:
      updates = self.per_step_updates

    def _update(unused_batch):
      return transform_fn(updates, state, params)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    pmap_fn = jax.pmap(_update, axis_name='batch')

    updates, state = pmap_fn(jnp.array([1.0]))
    chex.assert_tree_all_finite((params, updates, state))
    if expected_value is not None:
      last_entry = updates[1][-1, -1, -1]
      self.assertLess(
          abs(last_entry - expected_value),
          1e-4,
          msg=f'{last_entry=}, {expected_value=}')
    for _ in range(5):
      updates, state = pmap_fn(jnp.array([1.0]))
      chex.assert_tree_all_finite((params, updates, state))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters([
      {
          'testcase_name': 'default',
      },
      {
          'testcase_name': 'no_training_metrics',
          'generate_training_metrics': False,
      },
  ])
  def test_distributed_shampoo_no_pmap(self, generate_training_metrics=True):
    params = self.init_params

    optim = distributed_shampoo.distributed_shampoo(
        0.1,
        32,
        batch_axis_name=None,
        preconditioning_compute_steps=2,
        generate_training_metrics=generate_training_metrics)
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)
    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    updates, state = transform_fn(self.per_step_updates, state, params)
    chex.assert_tree_all_finite((params, updates, state))

  @chex.all_variants(with_pmap=False)
  @parameterized.named_parameters([
      {
          'testcase_name': 'preconditioning_compute_steps_schedule',
          'preconditioning_compute_steps': 2,
          'end_preconditioning_steps': 100,
      },
      {
          'testcase_name': (
              'preconditioning_compute_steps_schedule_short_circuit'
          ),
          'preconditioning_compute_steps': 1,
          'end_preconditioning_steps': 1,
      },
  ])
  def test_distributed_shampoo_preconditioning_compute_steps_schedule(
      self, preconditioning_compute_steps, end_preconditioning_steps
  ):
    params = self.init_params

    base_lr = 0.1

    def lr_fn(t):
      decay_factor = (t + 1) ** -0.5
      return base_lr * decay_factor

    optim = distributed_shampoo.distributed_shampoo(
        lr_fn,
        32,
        batch_axis_name='batch',
        preconditioning_compute_steps=preconditioning_compute_steps,
        decay_preconditioning_compute_steps=True,
        end_preconditioning_compute_steps=end_preconditioning_steps,
    )
    init_fn = self.variant(optim.init)
    transform_fn = self.variant(optim.update)

    updates = self.per_step_updates

    def _update(unused_batch):
      return transform_fn(updates, state, params)

    state = init_fn(params)
    chex.assert_tree_all_finite(state)
    pmap_fn = jax.pmap(_update, axis_name='batch')

    updates, state = pmap_fn(jnp.array([1.0]))
    chex.assert_tree_all_finite((params, updates, state))
    for _ in range(5):
      updates, state = pmap_fn(jnp.array([1.0]))
      chex.assert_tree_all_finite((params, updates, state))

  def _gen_symmetrix_matrix(self, dim, condition_number):
    u = scipy.stats.ortho_group.rvs(
        dim=dim, random_state=self.rng).astype(np.float64)
    v = u.T
    diag = np.diag([condition_number**(-i / (dim - 1)) for i in range(dim)])
    return u @ diag @ v

  def test_matrix_inverse_root(self):
    """Test for matrix inverse pth root."""

    # Fails after it reaches a particular condition number.
    for e in range(2, 12):
      condition_number = 10**e
      ms = self._gen_symmetrix_matrix(16, condition_number)
      self.assertLess(
          np.abs(np.linalg.cond(ms) - condition_number),
          condition_number * 0.01)
      metrics = distributed_shampoo.matrix_inverse_pth_root(
          ms.astype(np.float32), 4, ridge_epsilon=1e-12)[1]
      error = metrics.inverse_pth_root_errors
      if e < 7:
        self.assertLess(error, 0.1)
      else:
        # No guarantee of success after e >= 7
        pass

  @parameterized.parameters([{'sz': sz} for sz in [4, 32]])
  def test_matrix_inverse_root_padding(self, sz):
    """Test padding does not affect result much."""

    # Note sz == 1 case will not pass tests here b/c the method
    # is exact for scalars (but padding triggers coupled iteration).

    condition_number = 1e3
    ms = self._gen_symmetrix_matrix(sz, condition_number).astype(np.float32)

    # Shift matrix norm down by some large factor, so that improper padding
    # handling results in an error by increasing the condition number.
    ms = jnp.array(ms) * 1e-3

    rt, metrics = distributed_shampoo.matrix_inverse_pth_root(
        ms, 4, ridge_epsilon=1e-3)
    err = metrics.inverse_pth_root_errors
    pad_ms = distributed_shampoo.pad_square_matrix(ms, sz * 2)
    pad_rt, metrics = distributed_shampoo.matrix_inverse_pth_root(
        pad_ms, 4, ridge_epsilon=1e-3, padding_start=sz)
    pad_err = metrics.inverse_pth_root_errors
    pad_rt_principal = pad_rt[:sz, :sz]
    np.testing.assert_allclose(
        rt,
        pad_rt_principal,
        # The fact that this is so large keeps vladf up at night,
        # but without padding_start argument it's even worse (>1).
        rtol=1e-2 if sz == 4 else 5e-2,
        err_msg=np.array2string(rt - pad_rt_principal))
    self.assertLessEqual(pad_err, 4 * err)
    self.assertEqual(np.abs(pad_rt[sz:]).sum(), 0)
    self.assertEqual(np.abs(pad_rt[:, sz:]).sum(), 0)

  def test_all_padding(self):
    """Test full padding matrix."""
    empty = jnp.zeros([0, 0])
    padded = distributed_shampoo.pad_square_matrix(empty, 10)
    rt, metrics = distributed_shampoo.matrix_inverse_pth_root(
        padded, 4, ridge_epsilon=1e-3, padding_start=0)
    err = metrics.inverse_pth_root_errors
    self.assertEqual(np.abs(rt).sum(), 0.0)
    self.assertEqual(np.abs(err).sum(), 0.0)

  def _make_pth_diff_message(self, w, alpha, beta, p):
    left = f'({w} + {alpha})^(-1.0 / {p}) - '
    right = f'({w} + {beta})^(-1.0 / {p})'
    return left + right

  @parameterized.parameters(_pth_root_difference_cases())
  def test_pth_root_difference(self, p, a, b, w):
    """Test stable difference computation."""
    pth_rt_diff = jax.jit(
        functools.partial(distributed_shampoo._pth_root_difference, p=p))
    actual = pth_rt_diff(w, a, b)
    # in float64
    exp = (-1.0 / p)
    expected = (w + a)**exp - (w + b)**exp

    self.assertAlmostEqual(
        actual,
        expected,
        msg=self._make_pth_diff_message(w, a, b, p),
        delta=1e-2)

  @parameterized.parameters([{'p': p} for p in [2, 4, 8]])
  def test_lobpcg_preconditioning(self, p):
    """Checks that root calculation is valid with top-k preconditioning."""
    rng = np.random.RandomState(seed=42)
    n = 11
    epsilon = jnp.float32(1e-4)
    a_asymm = jnp.array(rng.random((n, n)), jnp.float32)
    a = jnp.matmul(a_asymm.T, a_asymm, precision=jax.lax.Precision.HIGHEST)
    log2 = (p - 1).bit_length()
    assert 2**log2 == p, (p, log2)

    root = functools.partial(
        distributed_shampoo.matrix_inverse_pth_root, ridge_epsilon=epsilon, p=p)
    root_lobpcg = functools.partial(
        root, lobpcg_topk_precondition=2, lobpcg_max_iter=10)

    methods = {'default': root, 'precond': root_lobpcg}
    spectrum_err, entry_err = {}, {}
    for k, method in methods.items():
      rt = jax.jit(method)(a)[0]

      # Recover the inverse by repeated squaring of inverse p-th root.
      inv = np.asarray(rt).astype(np.float64)
      for _ in range(log2):
        inv = inv.dot(inv)

      approx_id = inv.dot(a)
      spectrum = np.linalg.eigvalsh(approx_id)
      spectrum_err[k] = np.abs(1 - spectrum)
      entry_err[k] = np.mean(np.abs(approx_id - np.eye(n)))

    with np.printoptions(precision=2):

      def print_dict(d):
        return '\n'.join(f'{k} {v}' for k, v in d.items())

      err_msg = (f'p={p} log2(p)={log2}\n'
                 f'spectrum error\n{print_dict(spectrum_err)}\n'
                 f'entry_err\n{print_dict(entry_err)}')

      self.assertLessEqual(
          np.median(spectrum_err['precond']),
          2 * np.median(spectrum_err['default']),
          msg=err_msg)

      self.assertLessEqual(
          entry_err['precond'], entry_err['default'] * 2, msg=err_msg)


class LowRankInverseRootTest(chex.TestCase):

  def test_dynamic_exponent(self):
    """Test that exponent for various 'p' is correct."""
    root = jax.jit(
        functools.partial(
            distributed_shampoo._low_rank_root,
            compression_rank=1,
            ridge_epsilon=0.0,
            relative_matrix_epsilon=False,
        ))
    for p in range(2, 9):
      # Requires padding (else why would we compress to low rank?)
      a = np.zeros([4, 4], jnp.float32)
      a[0, 0] = 2**p
      exact = jnp.float32(1 / 2)
      r, metrics = root(a, p)
      e = metrics.inverse_pth_root_errors
      error = jnp.abs(r[0, 1] - exact)
      self.assertLessEqual(error, 10 * np.finfo(np.float32).eps)
      self.assertLessEqual(e, 10 * np.finfo(np.float32).eps)

  @parameterized.parameters(
      {
          'size': 5,
          'padded_size': 5,
          'compression_rank': 2
      },
      {
          'size': 5,
          'padded_size': 8,
          'compression_rank': 2
      },
      {
          'size': 5,
          'padded_size': 5,
          'compression_rank': -2
      },
      {
          'size': 5,
          'padded_size': 8,
          'compression_rank': -2
      },
  )
  def test_basic(self, size, padded_size, compression_rank):
    """Validate low rank returned matrix."""
    assert size > abs(compression_rank) + 2
    eps = 0.1
    root = jax.jit(
        functools.partial(
            distributed_shampoo._low_rank_root,
            p=2,
            compression_rank=compression_rank,
            ridge_epsilon=eps,
            relative_matrix_epsilon=False,
            padding_start=size))
    rng = np.random.default_rng(1234)
    a = rng.standard_normal(size=[size, size])
    a = a.T.dot(a)
    s, v = np.linalg.eigh(a + eps * np.eye(size))
    padded_a = np.zeros([padded_size, padded_size])
    padded_a[:size, :size] = a
    r, metrics = root(padded_a)
    packing_dim = abs(compression_rank) + 2
    assert list(r.shape) == [padded_size, packing_dim]
    e = metrics.inverse_pth_root_errors
    rv, re, rc, _ = distributed_shampoo._low_rank_unpack(
        r[:size, :packing_dim], compression_rank)
    complement = np.eye(size) - rv.dot(rv.T)
    root = complement * rc + rv.dot(np.diag(re).dot(rv.T))
    s = s**(-0.5)
    if compression_rank > 0:
      s[:-compression_rank] = np.mean(s[:-compression_rank])
    else:
      s[-compression_rank:] = np.mean(s[-compression_rank:])
    exact = v.dot(np.diag(s).dot(v.T))
    error = np.max(np.abs(exact - root))
    self.assertLessEqual(error, 1e-2)
    self.assertLessEqual(e, 1e-2)

  @parameterized.parameters(True, False)
  def test_nonzero_epsilon(self, relative):
    """Tests that the proper epsilon is added."""
    root = jax.jit(
        functools.partial(
            distributed_shampoo._low_rank_root,
            compression_rank=1,
            ridge_epsilon=0.1,
            relative_matrix_epsilon=relative,
        ))
    p = 2
    a = np.zeros([4, 4], jnp.float32)
    a[0, 0] = 2**p
    eps = 0.1 * (a[0, 0] if relative else 1)
    corner = jnp.float32((a[0, 0] + eps)**(-1.0 / p))
    ridge = jnp.float32(eps**(-1.0 / p))
    r, metrics = root(a, p)
    v, e, c, _ = distributed_shampoo._low_rank_unpack(r, compression_rank=1)
    err = metrics.inverse_pth_root_errors
    assert list(v.shape) == [4, 1]
    assert list(e.shape) == [1], e.shape
    assert not c.shape
    error = np.max(np.abs(corner - e[0]))
    self.assertLessEqual(error, 10 * np.finfo(np.float32).eps)
    self.assertLessEqual(err, 10 * np.finfo(np.float32).eps)
    vec_error = np.max(np.abs(np.array([1, 0, 0, 0]) - v.ravel()))
    self.assertLessEqual(vec_error, 10 * np.finfo(np.float32).eps, msg=v)
    self.assertLessEqual(abs(c - ridge), 1e-4)


def _make_pack_unpack_cases():
  sizes = [4, 8]
  paddings = [0, 10]
  cases = []
  for zero_tail in [True, False]:
    for size in sizes:
      for padding in paddings:
        for null_rank in [0, 1]:
          nonzero_rank = size - 2 - null_rank - 1
          for has_zeros in [True, False]:
            cases.append({
                'zero_tail': zero_tail,
                'size': size,
                'padding': padding,
                'null_rank': null_rank,
                'nonzero_rank': nonzero_rank,
                'has_zeros': has_zeros,
            })
  return cases


class FDLowRankInverseRootTest(chex.TestCase):

  def setUp(self):
    super().setUp()
    self.rank = None
    self.p = None
    self.rng = np.random.default_rng(1234)

  def tearDown(self):
    self.rank = None
    self.p = None
    self.rng = None
    super().tearDown()

  def _fd_update(
      self,
      grad=None,
      prev=None,
      ridge_epsilon=0.0,
      relative_matrix_epsilon=False,
      padding_start=None,
  ):
    assert self.p is not None
    assert self.rank is not None
    assert grad is not None or prev is not None
    padded_size = len(grad) if grad is not None else len(prev)
    if grad is None:
      grad = np.zeros([padded_size, padded_size], np.float32)
    else:
      grad = np.array(grad).astype(np.float32)
      assert list(grad.shape) == [padded_size, padded_size], grad.shape
    if prev is None:
      prev = np.zeros([padded_size, self.rank + 2], np.float32)
    else:
      prev = np.array(prev).astype(np.float32)
      assert list(prev.shape) == [padded_size, self.rank + 2
                                 ], (prev.shape, [padded_size, self.rank + 2])
    if padding_start is None:
      padding_start = padded_size
    assert padding_start <= padded_size
    fd_update = jax.jit(
        functools.partial(
            distributed_shampoo._fd_update_root,
            rank=self.rank,
            ridge_epsilon=ridge_epsilon,
            relative_matrix_epsilon=relative_matrix_epsilon,
            error_tolerance=0.0,
            padding_start=padding_start,
        ))
    updated, _ = fd_update(grad, p=self.p, prev=prev)
    x = {}
    (x['eigvecs'], x['eigvals'], x['inverted_eigvals'], x['inverted_tail'],
     x['tail'], x['has_zeros']) = distributed_shampoo._fd_low_rank_unpack(
         updated, self.rank)
    assert list(x['eigvecs'].shape) == [padded_size, self.rank]
    assert list(x['eigvals'].shape) == [self.rank]
    assert list(x['inverted_eigvals'].shape) == [self.rank]
    assert not x['inverted_tail'].shape
    assert not x['tail'].shape
    if x['tail'] > 0:
      self.assertAlmostEqual(x['tail']**(-1 / self.p), x['inverted_tail'])
      np.testing.assert_allclose((x['eigvals'] + x['tail'])**(-1 / self.p),
                                 x['inverted_eigvals'])
    return x

  def _make_fd_state(self, eigvecs, eigs, start_tail, padded_size=None):
    size, dim = eigvecs.shape
    assert dim == self.rank
    assert self.rank is not None
    assert self.p is not None
    assert self.rank + 2 < size
    assert len(eigs) == self.rank
    if padded_size is None:
      padded_size = size
    assert size <= padded_size
    eigvecs = jnp.pad(eigvecs, ((0, padded_size - size), (0, 0)))
    eigs = np.array(eigs).astype(np.float32)
    inv_eigs = np.where(
        eigs == 0.0, 0.0, jnp.where(eigs <= 0, 1, eigs) ** (-1 / self.p)
    )
    inv_tail = 0.0 if start_tail == 0.0 else start_tail**(-1 / self.p)
    return distributed_shampoo._fd_low_rank_pack(
        eigvecs,
        eigs,
        inv_eigs,
        inv_tail,
        start_tail,
        False,
        self.rank,
    )

  def _make_eye_state(self, size, eigs, start_tail, padded_size=None):
    eigvecs = np.eye(size, self.rank, dtype=np.float32)
    return self._make_fd_state(eigvecs, eigs, start_tail, padded_size)

  @parameterized.parameters(range(2, 9, 2))
  def test_dynamic_exponent(self, p):
    """Test that exponent for various 'p' is correct."""
    self.rank = 1
    self.p = p
    size = 4
    prev = self._make_eye_state(size, [0], 0.0)
    grad = np.zeros([4, 4], np.float32)
    grad[0, 0] = 2**(p / 2)  # Grad will get squared.
    ret = self._fd_update(grad, prev)
    self.assertAlmostEqual(ret['inverted_eigvals'], 1 / 2, delta=1e-6)

    prev = self._make_eye_state(size, [2**p], 0.0)
    grad = np.zeros([4, 4], np.float32)
    ret = self._fd_update(grad, prev)
    self.assertAlmostEqual(ret['inverted_eigvals'], 1 / 2, delta=1e-6)

  @parameterized.parameters(True, False)
  def test_nonzero_epsilon(self, relative):
    """Tests that the proper epsilon is added."""
    self.rank = 1
    self.p = 2
    size = 4

    eig0 = 2**self.p
    prev = self._make_eye_state(size, [eig0], 0.0)
    grad = np.zeros([size, size])
    ridge_epsilon = 0.1
    ret = self._fd_update(
        grad,
        prev,
        ridge_epsilon=ridge_epsilon,
        relative_matrix_epsilon=relative,
    )
    eps = 0.1 * (eig0 if relative else 1)
    self.assertAlmostEqual(eig0 + eps, ret['eigvals'][0], delta=1e-5)
    v = ret['eigvecs'][:, 0]
    vec_error = np.max(np.abs(np.array([1, 0, 0, 0]) - v.ravel()))
    self.assertLessEqual(vec_error, 10 * np.finfo(np.float32).eps, msg=v)
    self.assertAlmostEqual(ret['tail'], 0.0)

  def _make_rand_state(self, size, eigs, start_tail, padded_size=None):
    b = self.rng.standard_normal(size=[size, size])
    b = b.dot(b.T)
    _, v = np.linalg.eigh(b)
    return self._make_fd_state(v[:, :self.rank], eigs, start_tail, padded_size)

  @parameterized.parameters(_make_pack_unpack_cases())
  def test_pack_unpack(self, size, padding, nonzero_rank, null_rank, zero_tail,
                       has_zeros):
    self.p = 2
    self.rank = nonzero_rank + null_rank
    eigs = np.concatenate(
        [self.rng.uniform(size=[nonzero_rank]),
         np.zeros([null_rank])])
    self.rng.shuffle(eigs)
    start_tail = 0.0 if zero_tail else self.rng.uniform()
    padded_size = size + padding
    packed = self._make_rand_state(size, eigs, start_tail, padded_size)
    packed = packed.at[-1, -2].set(jnp.asarray(has_zeros).astype(jnp.float32))
    ret = distributed_shampoo._fd_low_rank_unpack(packed, self.rank)
    repacked = distributed_shampoo._fd_low_rank_pack(*(ret + (self.rank,)))
    np.testing.assert_array_equal(packed, repacked)

  @parameterized.parameters(itertools.product([0, 1], [0, 3], [True, False]))
  def test_basic(self, axis, padding, use_fd_update):
    """Validate low rank returned matrix."""
    size = 5
    padded_size = size + padding
    self.rank = 2
    self.p = 2
    assert size > self.rank + 2

    grad = self.rng.standard_normal(size=[size, size])
    for i in range(size):
      grad[i] /= np.linalg.norm(grad[i])
    added_cov = grad.dot(grad.T) if axis == 0 else grad.T.dot(grad)
    top_added_eig = np.linalg.eigvalsh(added_cov).max()
    start_tail = 0.0
    eigs = np.array([top_added_eig * 4, top_added_eig / 4])
    prev = self._make_rand_state(size, eigs, start_tail, padded_size)
    prev_eigvecs, *_ = distributed_shampoo._fd_low_rank_unpack(prev, self.rank)

    if use_fd_update:
      grad = distributed_shampoo.frequent_directions_update(
          np.zeros([]),  # Ignored argument.
          grad,
          axis,
          0.0,  # Ignored argument.
          0.0,  # Ignored argument.
      )
    else:
      # Align the axis to extract covariance for to dimension 0.
      grad = grad if axis == 0 else grad.T
    grad = np.pad(grad, ((0, padding), (0, padding)))
    updated = self._fd_update(
        grad,
        prev,
        padding_start=size,
    )

    prev_eigvecs = prev_eigvecs[:size, :]
    self.assertEqual(np.abs(prev_eigvecs[size:]).sum(), 0.0)
    half = np.float64(prev_eigvecs) * np.sqrt(np.float64(eigs))
    full_cov = half.dot(half.T) + added_cov
    s, new_v = np.linalg.eigh(np.float64(full_cov))
    s = np.flip(s)  # Descending order.
    new_v = np.flip(new_v, axis=1)
    expected_v = new_v[:, :self.rank]
    rv, re, tail = (updated[k] for k in ['eigvecs', 'eigvals', 'tail'])
    self.assertEqual(np.abs(rv[size:]).sum(), 0.0)
    rv = rv[:size, :]

    cross_error = np.abs(rv.T.dot(expected_v))
    self.assertLessEqual(
        np.max(np.abs(np.eye(self.rank) - cross_error)), 1e-2, msg=cross_error)

    cutoff = s[self.rank]
    self.assertAlmostEqual(tail, start_tail + cutoff, delta=1e-2)

    s[:self.rank] -= cutoff
    np.testing.assert_allclose(s[:self.rank] + tail, re + tail, rtol=1e-2)

  @parameterized.parameters(itertools.product([0, 3], [True, False]))
  def test_basic_1(self, padding, use_fd_update):
    """Validate low rank returned matrix."""
    size = 5
    padded_size = size + padding
    self.rank = 2
    self.p = 2
    assert size > self.rank + 2

    grad = self.rng.standard_normal(size=[size])
    grad /= np.linalg.norm(grad)
    added_cov = np.multiply.outer(grad, grad)
    top_added_eig = 1.0  # Normalized vector outer product.
    start_tail = 0.0
    eigs = np.array([top_added_eig * 4, top_added_eig / 4])
    prev = self._make_rand_state(size, eigs, start_tail, padded_size)
    prev_eigvecs, *_ = distributed_shampoo._fd_low_rank_unpack(prev, self.rank)

    if use_fd_update:
      axis = 0
      grad = distributed_shampoo.frequent_directions_update(
          np.zeros([]),  # Ignored argument.
          grad,
          axis,
          0.0,  # Ignored argument.
          0.0,  # Ignored argument.
      )
    else:
      grad = np.pad(grad.reshape(-1, 1), ((0, 0), (0, size - 1)))
    grad = np.pad(grad, ((0, padding), (0, padding)))
    updated = self._fd_update(
        grad,
        prev,
        padding_start=size,
    )

    prev_eigvecs = prev_eigvecs[:size, :]
    self.assertEqual(np.abs(prev_eigvecs[size:]).sum(), 0.0)
    half = np.float64(prev_eigvecs) * np.sqrt(np.float64(eigs))
    full_cov = half.dot(half.T) + added_cov
    s, new_v = np.linalg.eigh(np.float64(full_cov))
    s = np.flip(s)  # Descending order.
    new_v = np.flip(new_v, axis=1)
    expected_v = new_v[:, :self.rank]
    rv, re, tail = (updated[k] for k in ['eigvecs', 'eigvals', 'tail'])
    self.assertEqual(np.abs(rv[size:]).sum(), 0.0)
    rv = rv[:size, :]

    cross_error = np.abs(rv.T.dot(expected_v))
    self.assertLessEqual(
        np.max(np.abs(np.eye(self.rank) - cross_error)), 1e-2, msg=cross_error)

    cutoff = s[self.rank]
    self.assertAlmostEqual(tail, start_tail + cutoff, delta=1e-2)

    s[:self.rank] -= cutoff
    np.testing.assert_allclose(s[:self.rank] + tail, re + tail, rtol=1e-2)


if __name__ == '__main__':
  absltest.main()
