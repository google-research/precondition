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

from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
from precondition.tearfree import reshaper


def _make_invalid_cases() -> Sequence[dict[str, ...]]:
  """Generate invalid cases which should throw."""
  return [
      {
          'testcase_name': 'smallblock',
          'invalid_options': reshaper.Options(
              block_size=1,
          ),
      },
      {
          'testcase_name': 'smallmerge',
          'invalid_options': reshaper.Options(
              merge_dims=0,
          ),
      },
  ]


def _make_expected_shape_cases() -> Sequence[dict[str, ...]]:
  cases = [
      {'in_shape': [4], 'merge': 2, 'block': 3, 'out_shape': [6]},
      {'in_shape': [3], 'merge': 2, 'block': 3, 'out_shape': [3]},
      {'in_shape': [1, 3, 1], 'merge': 2, 'block': 3, 'out_shape': [3]},
      {'in_shape': [1, 3, 1], 'merge': 3, 'block': 3, 'out_shape': [3]},
      {'in_shape': [1, 3, 1], 'merge': 3, 'block': 4, 'out_shape': [3]},
      {'in_shape': [1, 3, 1, 2], 'merge': 2, 'block': 3, 'out_shape': [3, 2]},
      {'in_shape': [4, 1, 5], 'merge': 2, 'block': 3, 'out_shape': [6, 6]},
      {'in_shape': [1], 'merge': 2, 'block': 2, 'out_shape': []},
      {'in_shape': [1, 1, 1], 'merge': 2, 'block': 2, 'out_shape': []},
      {'in_shape': [1, 1, 1], 'merge': 2, 'block': 2, 'out_shape': []},
      {
          'in_shape': [3, 1, 5, 2, 2],
          'merge': 4,
          'block': 10,
          'out_shape': [3, 5, 4],
      },
      {'in_shape': [2, 3, 2], 'merge': 6, 'block': 10, 'out_shape': [6, 2]},
  ]
  for case in cases[:]:
    if all(i <= case['block'] for i in case['in_shape']):
      block0 = case.copy()
      block0['block'] = 0
      cases.append(block0)
  return cases


class ReshaperTest(parameterized.TestCase):
  """Basic test for shampoo implementation."""

  @parameterized.named_parameters(_make_invalid_cases())
  def test_invalid(self, invalid_options):
    with self.assertRaises(ValueError):
      reshaper.merge(invalid_options)

  @parameterized.parameters(_make_expected_shape_cases())
  def test_expected_shape(self, in_shape, merge, block, out_shape):
    options = reshaper.Options(merge_dims=merge, block_size=block)
    init_fn, update_fn = reshaper.merge(options)
    init = jnp.zeros(in_shape)
    out, _ = update_fn(init, init_fn(None), init)
    self.assertSequenceEqual(out.shape, out_shape)

  @parameterized.parameters(_make_expected_shape_cases())
  def test_inversion(self, in_shape, merge, block, out_shape):
    del out_shape
    options = reshaper.Options(merge_dims=merge, block_size=block)
    init_fn, update_fn = reshaper.merge(options)
    init = jax.random.normal(jax.random.PRNGKey(0), in_shape)
    out, _ = update_fn(init, init_fn(None), init)
    init_fn, update_fn = reshaper.unmerge(options)
    recover, _ = update_fn(out, init_fn(None), init)
    np.testing.assert_array_equal(init, recover)

  def test_tree(self):
    shapes = {
        'w': [[{'b': (3, 2)}]],
        'z': (
            1,
            2,
            1,
        ),
    }
    init = jax.tree.map(
        jnp.zeros, shapes, is_leaf=lambda x: isinstance(x, tuple)
    )
    options = reshaper.Options(merge_dims=2, block_size=2)
    init_fn, update_fn = reshaper.merge(options)
    out, _ = update_fn(init, init_fn(None), init)
    out_shapes = jax.tree.map(lambda x: tuple(x.shape), out)
    expected_shapes = {'w': [[{'b': (4, 2)}]], 'z': (2,)}

    self.assertEqual(out_shapes, expected_shapes)


if __name__ == '__main__':
  absltest.main()
