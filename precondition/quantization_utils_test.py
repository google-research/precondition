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

"""Numerical coverage for integer quantization of low-precision values."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from precondition.quantization_utils import QuantizedValue
from precondition import sm3


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize('quantized_dtype', [jnp.int8, jnp.int16])
@pytest.mark.parametrize('scale', [1e-4, 1.0])
def test_low_precision_roundtrip_matches_float32_quantization(
    dtype, quantized_dtype, scale
):
  values = (
      jnp.array(
          [[-1.0, 0.3, 0.0], [0.6, -0.8, 0.0], [0.1, 1.0, 0.0]], dtype=dtype
      )
      * scale
  )
  reference = QuantizedValue.from_float_value(
      values.astype(jnp.float32), quantized_dtype
  )
  for quantize in (
      lambda x: QuantizedValue.from_float_value(x, quantized_dtype),
      jax.jit(lambda x: QuantizedValue.from_float_value(x, quantized_dtype)),
  ):
    actual = quantize(values)
    np.testing.assert_array_equal(actual.quantized, reference.quantized)
    np.testing.assert_allclose(
        actual.bucket_size, reference.bucket_size, rtol=1e-6, atol=0
    )
    np.testing.assert_allclose(
        actual.to_float(), reference.to_float(), rtol=1e-6, atol=0
    )
    assert actual.to_float().dtype == jnp.float32
    assert actual.quantized.dtype == quantized_dtype
    assert actual.shape == list(values.shape)
    buckets = float(jnp.iinfo(quantized_dtype).max)
    assert np.all(
        np.abs(np.asarray(actual.quantized, dtype=np.int64)) <= buckets
    )


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
def test_diagonal_is_preserved_while_offdiagonal_scale_is_float32(dtype):
  values = jnp.array([[2.0, 1e-4], [-1e-4, 3.0]], dtype=dtype)
  actual = QuantizedValue.from_float_value(
      values, jnp.int16, extract_diagonal=True
  )
  np.testing.assert_allclose(
      actual.to_float(), np.asarray(values, dtype=np.float32), rtol=2e-5, atol=0
  )
  np.testing.assert_array_equal(jnp.diag(actual.to_float()), jnp.diag(values))


def test_float32_result_and_floating_storage_paths_remain_unchanged():
  values = jnp.array([[1.0, -2.0], [3.0, 4.0]], dtype=jnp.float32)
  quantized = QuantizedValue.from_float_value(values, jnp.int16)
  expected_scale = jnp.max(jnp.abs(values), axis=0) / 32767
  np.testing.assert_array_equal(quantized.bucket_size, expected_scale)
  for dtype in [jnp.float32, jnp.bfloat16]:
    actual = QuantizedValue.from_float_value(values, dtype)
    np.testing.assert_array_equal(
        actual.to_float(), values.astype(dtype).astype(jnp.float32)
    )
  assert QuantizedValue.from_float_value([], jnp.int8).to_float() == []


@pytest.mark.parametrize('dtype', [jnp.float16, jnp.bfloat16])
def test_sm3_low_precision_initial_state_can_be_carried_through_jitted_steps(
    dtype,
):
  parameters = {'weight': jnp.array([[1.0, -1.0], [0.5, -0.5]], dtype=dtype)}
  optimizer = sm3.sm3(learning_rate=0.01)
  state = optimizer.init(parameters)
  gradients = jax.tree.map(
      lambda p: jnp.full(p.shape, 0.01, dtype=jnp.float32), parameters
  )

  def step(state, _):
    update, state = optimizer.update(gradients, state, parameters)
    return state, update

  final, updates = jax.jit(lambda s: jax.lax.scan(step, s, None, length=3))(
      state
  )
  assert int(final.count) == 3
  assert np.isfinite(updates['weight']).all()
  assert (
      final.stats['weight'].diagonal_momentum.bucket_size.dtype == jnp.float32
  )
