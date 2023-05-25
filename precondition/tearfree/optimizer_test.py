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

"""Tests for tearfree optimizer."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
from jax import numpy as jnp
import numpy as np
import optax
from praxis import optimizers
from precondition.tearfree import grafting
from precondition.tearfree import momentum
from precondition.tearfree import optimizer
from precondition.tearfree import second_order
from precondition.tearfree import shampoo


class OptimizerTest(parameterized.TestCase):
  """Basic test for optimizer configurations."""

  def setUp(self):
    super().setUp()
    jax.config.update('jax_debug_nans', True)

  def _unroll(self, options, shape, transform=None, lr=0.1, n=4):
    """Generate states and grad updates n times."""
    rng = jax.random.PRNGKey(0)
    params = jnp.zeros(shape)
    grads = jax.random.normal(rng, (n, *shape))

    if transform is not None:
      params = transform(params)
      grads = jnp.stack([transform(g) for g in grads])

    if isinstance(options, optimizer.TearfreeOptions):
      tx = optimizer.tearfree(lr, options)
    else:
      tx = options
    init = tx.init(params)

    def reduce(state, grad):
      new_grad, new_state = tx.update(grad, state, params)
      return new_state, new_grad

    _, out_grads = jax.lax.scan(reduce, init, grads)
    return out_grads

  def _no_graft_no_momentum(self):
    return optimizer.TearfreeOptions(
        grafting_options=grafting.Options(
            grafting_type=grafting.GraftingType.NONE, second_moment_decay=0.0
        ),
        momentum_options=momentum.Options(momentum_decay=0.0),
    )

  def test_merge_dims(self):
    shape = (2, 2)
    options = dataclasses.replace(
        self._no_graft_no_momentum(),
        second_order_options=second_order.Options(merge_dims=4),
    )
    transform = lambda x: x.reshape(4)
    actual = self._unroll(options, shape)
    expected = self._unroll(options, shape, transform)
    np.testing.assert_allclose(actual.reshape(-1, 4), expected)

  def test_block_size(self):
    shape = (4,)
    options = dataclasses.replace(
        self._no_graft_no_momentum(),
        second_order_options=second_order.Options(
            shampoo_options=shampoo.Options(block_size=3)
        ),
    )
    actual = self._unroll(options, shape)
    expected = self._unroll(options, shape)
    np.testing.assert_allclose(actual, expected)

  @parameterized.parameters(
      momentum.Options(),  # Default is 0.9, active momentum.
      momentum.Options(momentum_decay=0.0),
      momentum.Options(weight_decay=0.01),
      momentum.Options(weight_decay=0.01, weight_decay_after_momentum=False),
      momentum.Options(nesterov=False),
      momentum.Options(ema=True),
      momentum.Options(ema=True, nesterov=True),
  )
  def test_momentum_no_graft(self, momentum_options):
    shape = (4,)
    options = self._no_graft_no_momentum()
    options.momentum_options = momentum_options
    tx = optimizers.sharded_chain(
        second_order.apply(options.second_order_options),
        momentum.apply(momentum_options),
        optax.scale(-0.1),
    )
    actual = self._unroll(options, shape)
    expected = self._unroll(tx, shape)
    np.testing.assert_allclose(actual, expected)

  def _grafting_tx(
      self, grafting_options
  ) -> optimizers.ShardedGradientTransformation:
    id_tx = optax.identity()
    id_tx_shard = optimizers.ShardedGradientTransformation(
        id_tx.init,
        id_tx.update,
        lambda _: optax.EmptyState(),
    )
    return grafting.graft(grafting_options, id_tx_shard)

  def _grafting_tx_with_momentum(
      self, grafting_options, momentum_options, lr=0.1
  ):
    return optimizers.sharded_chain(
        self._grafting_tx(grafting_options),
        momentum.apply(momentum_options),
        optax.scale(-lr),
    )

  @parameterized.parameters(
      grafting.Options(),
      grafting.Options(
          grafting_type=grafting.GraftingType.SGD, second_moment_decay=0.0
      ),
      grafting.Options(second_moment_decay=1.0),
  )
  def test_momentum_yes_graft(self, grafting_options):
    shape = (4,)
    nsteps = 4
    options = self._no_graft_no_momentum()
    options.momentum_options.momentum_decay = 0.9
    options.grafting_options = grafting_options
    grafting_options.start_preconditioning_step = nsteps + 1
    tx = self._grafting_tx_with_momentum(
        grafting_options, options.momentum_options
    )
    expected = self._unroll(tx, shape, n=nsteps)
    actual = self._unroll(options, shape, n=nsteps)
    np.testing.assert_allclose(actual, expected)

  def _precondition_at(self, i):
    """Return optimizer with momentum, grafting, and start precon at step i."""
    return optimizer.TearfreeOptions(
        grafting_options=grafting.Options(start_preconditioning_step=i)
    )

  @parameterized.parameters(
      dict(shape=(1, 1, 1)),
      dict(shape=(1,)),
      dict(shape=tuple()),
  )
  def test_scalar_is_grafting(self, shape):
    nsteps = 4
    options = self._precondition_at(2)
    tx = self._grafting_tx_with_momentum(
        options.grafting_options, options.momentum_options
    )
    expected = self._unroll(tx, shape, n=nsteps)
    actual = self._unroll(options, shape, n=nsteps)
    np.testing.assert_allclose(actual, expected)

  def test_lr(self):
    shape = (3,)
    options = self._precondition_at(2)
    nsteps = 4

    def schedule(count):
      return (count + 1) * 0.1

    actual = self._unroll(options, shape, lr=schedule, n=nsteps)
    expected = self._unroll(options, shape, lr=0.1, n=nsteps)
    expected *= (jnp.arange(nsteps) + 1).reshape(-1, 1)
    np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
  absltest.main()
