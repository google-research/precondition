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

"""A collection of functionally-expressed OCO algorithms."""

import dataclasses
import enum
from typing import Callable, Union
import jax
import jax.numpy as jnp
import numpy as np


class Algorithm(enum.Enum):
  OGD = enum.auto()
  ADA = enum.auto()
  RFD_SON = enum.auto()
  FD_SON = enum.auto()
  ADA_FD = enum.auto()
  S_ADA = enum.auto()


RuntimeScalar = Union[float, jax.Array]


@dataclasses.dataclass
class HParams:
  """A union of all hyperparmeters across algorithms in this file."""

  # Initial diagonal regularization delta * I.
  delta: RuntimeScalar

  # Learning rate.
  lr: RuntimeScalar

  # Sketch size. (0 for non-sketched).
  sketch_size: int

  # Sketch 'style' disambiguates which of the sketched second-order methods
  # we should apply.
  algorithm: Algorithm


State = dict[str, jax.Array]


NpState = dict[str, np.ndarray]


def as_np(state: State) -> NpState:
  return {k: np.asarray(v, dtype=v.dtype) for k, v in state.items()}


InitFn = Callable[[], State]


UpdateFn = Callable[[State, jax.Array, jax.Array], State]


def generate_init_update(
    w_shape: tuple[int], hparams: HParams
) -> tuple[InitFn, UpdateFn]:
  """Bind parameters to appropriate init/update functions for OCO."""
  if hparams.algorithm == Algorithm.OGD:
    assert hparams.sketch_size == 0, hparams.sketch_size
    init, update = _ogd_init_fn, _ogd_update_fn
  elif hparams.algorithm == Algorithm.ADA:
    assert hparams.sketch_size == 0, hparams.sketch_size
    init, update = _diag_adagrad_init_fn, _diag_adagrad_update_fn
  else:
    assert hparams.sketch_size > 1, hparams.sketch_size
    init, update = _fd_init_fn, _fd_update_fn

  bound_init_fn = lambda: init(w_shape, hparams)
  bound_update_fn = lambda state, loss, grad: update(state, loss, grad, hparams)
  return bound_init_fn, bound_update_fn


def _ogd_init_fn(w_shape: tuple[int], hparams: HParams) -> State:
  """Initialize OGD state."""
  del hparams
  return {
      'w': jnp.zeros(w_shape, dtype=jnp.float64),
      't': jnp.array(0.0, jnp.float64),
  }


def _ogd_update_fn(
    state: State, loss: jax.Array, grad: jax.Array, hparams: HParams
) -> State:
  """Update OGD state."""
  del loss
  assert state['w'].shape == grad.shape, (state['w'].shape, grad.shape)
  state['t'] += 1.0
  state['w'] -= hparams.lr * grad * jax.lax.rsqrt(state['t'] + hparams.delta)
  return state


def _diag_adagrad_init_fn(w_shape: tuple[int], hparams: HParams) -> State:
  """Initialize adagrad state."""
  return {
      'w': jnp.zeros(w_shape, dtype=jnp.float64),
      'diag_h': jnp.ones(w_shape, dtype=jnp.float64) * hparams.delta,
  }


def _diag_adagrad_update_fn(
    state: State, loss: jax.Array, grad: jax.Array, hparams: HParams
) -> State:
  """Update adagrad state."""
  del loss
  state['diag_h'] = state['diag_h'] + grad**2
  rsqrt = jax.lax.rsqrt(jnp.where(state['diag_h'] == 0, 1, state['diag_h']))
  state['w'] -= rsqrt * grad * hparams.lr
  return state


def _fd_init_fn(w_shape: tuple[int], hparams: HParams) -> State:
  """Initialize sketch algorithm state."""
  state = {
      'w': jnp.zeros(w_shape, dtype=jnp.float64),
      't': jnp.array(0.0, dtype=jnp.float64),
  }
  grad_size = state['w'].size
  sketch_size = hparams.sketch_size
  assert grad_size >= sketch_size
  assert sketch_size >= 2
  state['alpha'] = jnp.array(hparams.delta, dtype=jnp.float64)
  # Preconditioner eigenvectors.
  state['P'] = jnp.zeros((sketch_size, grad_size), dtype=jnp.float64)
  # Covariance root eigenvalues (i.e., from the sketch, NOT covariance eigs)
  state['e'] = jnp.zeros((sketch_size,), dtype=jnp.float64)
  return state


# RFD0 for Online Newton Step, Algorithm 3 in Section 5.
# https://arxiv.org/pdf/1705.05067.pdf
# \mu_t, the exp-convexity constant, is zero.
# Unique parts: screen/ZxejS5fYRAB7tG8
#   - updates sketch with sqrt(eta_t) ~ 1/sqrt(t), up to a constant, the LR.
#   - increments alpha by rho/2
#   - no LR in the update
#   - RFD0 has no hparams besides LR
#
# The "alpha0" hparam for RFD (not RFD0) is exactly hparams['delta']. Note that
# RFD and RFD0 can also be given a learning rate which scales eta_t; we choose
# a parameterization consistent with other algorithms' uses of hparams['lr'].
def _rfd(state: State, hparams: HParams) -> ...:
  """Create RFD-specific algorithmic changes."""
  sketch_update_factor = jax.lax.rsqrt(state['t'] * hparams.lr)
  alpha_update_factor = 0.5
  lr = 1.0
  eig_inversion = jnp.reciprocal
  return sketch_update_factor, alpha_update_factor, lr, eig_inversion


# FD-SON - Algorithm 1 and 3 in Sections 2 and 3.
# https://arxiv.org/pdf/1602.02202.pdf
# Per screen/BqHhFBN3qt4zrLa
#  - update the scetch with sqrt(eta_t), like in RFD0
#  - eta_t has itself 1/sqrt(t) decay, unlike RFD0
#  - we use hparams['delta'] instead of alpha
#  - we again consolidate all constants on eta_t into hparams['lr']
#
# Per screen/AB4SSFXvAh6PYe7, alpha does not update.
def _fdson(state: State, hparams: HParams) -> ...:
  """Create FD-SON-specific algorithmic changes."""
  sketch_update_factor = jax.lax.rsqrt(jnp.sqrt(state['t']) * hparams.lr)
  alpha_update_factor = 0.0
  lr = 1.0
  eig_inversion = jnp.reciprocal
  return sketch_update_factor, alpha_update_factor, lr, eig_inversion


# Ada-FD - https://www.ijcai.org/proceedings/2018/0381.pdf
# Per screen/BMwyfTHcwnbpexn
#  - Similar to S-Adagrad, except no updating alpha (dynamic diagonal).
#  - However, they add delta to the *square rooted* eigenvalues, unlike
#    all the other works.
def _adafd(state: State, hparams: HParams) -> ...:
  """Create Ada-FD-specific algorithmic changes."""
  del state
  sketch_update_factor = 1.0
  alpha_update_factor = 0.0
  lr = hparams.lr
  eig_inversion = 'Ada-FD requires special handling'
  return sketch_update_factor, alpha_update_factor, lr, eig_inversion


# S-Adagrad - https://arxiv.org/pdf/2302.03764.pdf
def _sada(state: State, hparams: HParams) -> ...:
  """Create S-Adagrad-specific algorithmic changes."""
  del state
  sketch_update_factor = 1.0
  alpha_update_factor = 1.0
  lr = hparams.lr
  eig_inversion = jax.lax.rsqrt
  return sketch_update_factor, alpha_update_factor, lr, eig_inversion


def _fd_method_factors(state: State, hparams: HParams) -> ...:
  """Create algorithmic variants specific to hparams."""
  return {
      Algorithm.RFD_SON: _rfd(state, hparams),
      Algorithm.FD_SON: _fdson(state, hparams),
      Algorithm.ADA_FD: _adafd(state, hparams),
      Algorithm.S_ADA: _sada(state, hparams),
  }[hparams.algorithm]


def _fd_update_fn(
    state: State, loss: jax.Array, grad: jax.Array, hparams: HParams
) -> State:
  """Update FD algorithm state given loss and grad."""
  del loss
  state['t'] += 1.0
  sketch_update_factor, alpha_update_factor, lr, eig_inversion = (
      _fd_method_factors(state, hparams)
  )
  grad_input = grad.ravel() * sketch_update_factor

  B = state['P'] * state['e'].reshape(-1, 1)  # pylint: disable=invalid-name
  B = B.at[-1].set(grad_input)  # pylint: disable=invalid-name
  _, s, vt = jnp.linalg.svd(B, full_matrices=False)
  rho = s[-1]
  s = (s - rho) * (s + rho)
  P = state['P'] = vt  # pylint: disable=invalid-name
  state['e'] = jnp.sqrt(s)
  state['alpha'] += alpha_update_factor * rho**2

  mm = lambda x, y: jnp.dot(x, y, precision=jax.lax.Precision.HIGHEST)
  g = grad.ravel()
  alpha = state['alpha']

  eps = 0.0

  def safe_invert(x, inversion=eig_inversion):
    return jnp.where(x <= eps, 0.0, inversion(x))

  if eig_inversion == 'Ada-FD requires special handling':
    e = state['e']
    d = e / (alpha + e)
    update = g - mm(P.T, d * mm(P, g))
    update *= safe_invert(alpha, inversion=jnp.reciprocal)
  else:
    e = alpha + s
    inv_s = safe_invert(e)
    inv_alpha = safe_invert(alpha)
    outside_sketch_g = g - mm(P.T, mm(P, g))
    sketched_precond = mm(P.T, inv_s * mm(P, g))
    update = sketched_precond + inv_alpha * outside_sketch_g

  state['w'] -= lr * update.reshape(state['w'].shape)
  return state
