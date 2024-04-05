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

"""Module with sparse datasets on CNS."""

import dataclasses
import os
from typing import Callable

from absl import flags
import jax
import jax.numpy as jnp
import joblib
import numpy as np
import sklearn.datasets

_DATA_DIR = flags.DEFINE_string(
    'data_dir',
    None,
    'load data: your directory needs to contain the benchmark datasets'
    ' (a9a, a9a.t, cifar10, cifar10.t, gisette_scale, gisette_scale.t,'
    ' where .t stands for the testing set)'
    ' in libsvm format, available at'
    ' https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/',
)

SUPPORTED_DATASETS: list[str] = [
    'a9a',
    'a9a.t',
    'cifar10',
    'cifar10.t',
    'gisette_scale',
    'gisette_scale.t',
]


def _logistic_loss(w: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
  """Compute logistic loss."""
  # assumes y is binary
  pred = jnp.dot(w, x, precision=jax.lax.Precision.HIGHEST)
  lse = lambda x: jax.nn.logsumexp(jnp.array(x))
  return y * lse([0, -pred]) + (1 - y) * lse([0, pred])


def incorrect(w: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
  """Compute binary 0-1 loss."""
  pred = jnp.dot(w, x, precision=jax.lax.Precision.HIGHEST)
  return (pred > 0) != (y > 0)


Loss = Callable[[jax.Array, jax.Array, jax.Array], jax.Array]


@dataclasses.dataclass
class SimpleDataset:
  """Simple dense supervised learning dataset for linear learners."""

  x: np.ndarray
  y: np.ndarray
  loss: Loss
  w_shape: tuple[int, ...]


def _load_dataset_uncached(name: str) -> SimpleDataset:
  """Generate a dataset with an intercept added."""
  assert name in SUPPORTED_DATASETS, name
  if not _DATA_DIR.value:
    raise ValueError('must specify directory where datasets are stored')
  filename = os.path.join(_DATA_DIR.value, name)
  with open(filename, 'rb') as f:
    x, y = sklearn.datasets.load_svmlight_file(f)

  x = x.todense()
  x = np.concatenate([x, np.ones((len(x), 1))], axis=1)
  y = y > 0
  return SimpleDataset(x, y, _logistic_loss, (x.shape[1],))


def load_dataset(name: str, cache: str = '/tmp/cache') -> SimpleDataset:
  memory = joblib.Memory(cache, verbose=0)
  cached_fn = memory.cache(_load_dataset_uncached)
  return cached_fn(name)
