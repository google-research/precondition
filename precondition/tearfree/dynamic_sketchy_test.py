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

"""Test case for dynamic sketchy."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
from precondition.tearfree import dynamic_sketchy
from precondition.tearfree import sketchy


class DynamicSketchyTest(parameterized.TestCase):

  @parameterized.parameters(
      [{'n': i, 'm': j} for i, j in zip(range(6, 11), range(6, 11))]
  )
  def test_proj(self, m, n):
    options = dynamic_sketchy.Options()
    delta = options.delta
    key = jax.random.PRNGKey(0)
    v = jax.random.uniform(key, (n,))
    projected = dynamic_sketchy._proj(v, m, delta)
    assert jnp.sum(projected) <= (1 - delta) * m
    assert jnp.all(projected) >= 0

  @parameterized.parameters([{'shape': {'a': (3, 2), 'b': (4, 3)}}])
  def test_initiliazation(self, shape):
    params = jax.tree_util.tree_map(
        jnp.zeros,
        shape,
        is_leaf=lambda x: isinstance(x, tuple)
        and all(isinstance(y, int) for y in x),
    )
    options = dynamic_sketchy.Options()
    dynamic_sketchy._init(options, params)
    return dynamic_sketchy._init(options, params)

  @parameterized.parameters([{'shape': {'a': (3, 2), 'b': (4, 3)}}])
  def test_update(self, shape):
    params = jax.tree_util.tree_map(
        jnp.zeros,
        shape,
        is_leaf=lambda x: isinstance(x, tuple)
        and all(isinstance(y, int) for y in x),
    )
    options = dynamic_sketchy.Options()
    state = dynamic_sketchy._init(options, params)
    updates = jax.tree_util.tree_map(
        jnp.ones,
        shape,
        is_leaf=lambda x: isinstance(x, tuple)
        and all(isinstance(y, int) for y in x),
    )
    updated = dynamic_sketchy._update(options, updates, state)
    return updated

  def _generate_params_grads(self, shape):
    """Helper function to generate test params and gradients."""

    key = jax.random.PRNGKey(0)
    grads = {}
    grads['a'] = jax.random.normal(key, shape['a'], dtype=jnp.float32)
    grads['b'] = jax.random.normal(key, shape['b'], dtype=jnp.float32)
    params = jax.tree_map(
        jnp.zeros,
        shape,
        is_leaf=lambda x: isinstance(x, tuple)
        and all(isinstance(y, int) for y in x),
    )
    return params, grads

  def _unroll(self, nsteps, tx, options, grads, state):
    """Helper function to unroll gradients."""

    for _ in range(nsteps):
      grads, state = tx(options, grads, state)

    return grads, state

  @parameterized.parameters([{'shape': {'a': (3, 2), 'b': (4, 3)}}])
  def test_precondition(self, shape):
    params, grads = self._generate_params_grads(shape)
    group_maps = dynamic_sketchy._create_groups(params)
    sketchy_options = sketchy.Options()
    dynamic_options = dynamic_sketchy.Options()
    sketchy_state = sketchy._init(sketchy_options, params)
    dynamic_state = dynamic_sketchy._init(dynamic_options, params)
    is_tensor_state = lambda x: isinstance(x, sketchy._TensorState)
    sketchy_new_grads = jax.tree_util.tree_map_with_path(
        functools.partial(sketchy._precondition, sketchy_options),
        grads,
        sketchy_state.sketches,
        is_leaf=is_tensor_state,
    )
    dynamic_new_grads = jax.tree_util.tree_map_with_path(
        functools.partial(
            dynamic_sketchy._precondition,
            group_states=dynamic_state.group_states,
            layer2dimidx=group_maps.layer2dimidx,
        ),
        grads,
    )
    assert jnp.isclose(sketchy_new_grads['a'], dynamic_new_grads['a']).all(), (
        sketchy_new_grads['a'],
        dynamic_new_grads['a'],
    )
    assert jnp.isclose(sketchy_new_grads['b'], dynamic_new_grads['b']).all(), (
        sketchy_new_grads['b'],
        dynamic_new_grads['b'],
    )

  @parameterized.parameters(
      {'n': 3, 'shape': {'a': (i, i), 'b': (i, i)}, 'rank': i}
      for i in range(3, 4)
  )
  def test_reduction_to_sketchy(self, n, shape, rank):
    params, grads = self._generate_params_grads(shape)
    sketchy_options = sketchy.Options(rank=rank)
    dynamic_options = dynamic_sketchy.Options(rank=rank)
    sketchy_state = sketchy._init(sketchy_options, params)
    dynamic_state = dynamic_sketchy._init(dynamic_options, params)
    sketchy_nsteps, _ = self._unroll(
        n, sketchy._update, sketchy_options, grads, sketchy_state
    )
    dynamic_nsteps, dynamic_state = self._unroll(
        n,
        dynamic_sketchy._update,
        dynamic_options,
        grads,
        dynamic_state,
    )
    assert jnp.isclose(sketchy_nsteps['a'], dynamic_nsteps['a']).all(), (
        sketchy_nsteps['a'],
        dynamic_nsteps['a'],
        dynamic_state.group_states[rank].memory_pi,
    )
    assert jnp.isclose(
        sketchy_nsteps['b'], dynamic_nsteps['b'], atol=1e-4
    ).all(), (
        sketchy_nsteps['b'],
        dynamic_nsteps['b'],
    )


if __name__ == '__main__':
  absltest.main()
