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

"""Simple test case for memory reallocation function."""

import json
import os

from absl.testing import absltest
from absl.testing import parameterized
from jax import numpy as jnp
from precondition.tearfree import reallocation


def dict_almost_equal(dict1, dict2, delta=1):
  """Helper function."""
  for key, value in dict1.items():
    assert key in dict2, key
    if isinstance(value, dict):
      dict_almost_equal(value, dict2[key], delta)
    else:
      for i in range(len(value)):
        assert jnp.abs(value[i] - dict2[key][i]) <= delta


class ReallocationTest(parameterized.TestCase):

  def test_create_redist_dict(self):
    chpt_path = ''
    data_dir = os.path.join(
        os.path.dirname(__file__), 'reallocation_test_data'
    )
    realloc_path = os.path.join(data_dir, 'gnn_realloc.json')
    states_path = os.path.join(data_dir, 'states.json')
    with open(states_path, 'r') as f:
      states = tuple(json.load(f))
    sketches = states[-1]['inner_state']['0']['direction']['1']['sketches']
    for layer in sketches:
      tmp = sketches[layer]['kernel']['axes']
      for axes in tmp:
        tmp[axes]['eigvals'] = jnp.array(
            tmp[axes]['eigvals'], dtype=jnp.float32
        )
      states[-1]['inner_state']['0']['direction']['1']['sketches'][layer][
          'kernel'
      ]['axes'] = tmp
    realloc_result = reallocation.create_redist_dict(
        chpt_path, [-1], 'sketch_trace', False, 256, states
    )
    with open(realloc_path, 'r') as f:
      realloc_dict = json.load(f)

    dict_almost_equal(realloc_result, realloc_dict, delta=1)

if __name__ == '__main__':
  absltest.main()

