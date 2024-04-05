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

"""Python binary to run and evaluate OCO sweeps.

This can be used to either run the best hparams from a previous run on a test
set of data, or to sweep hparams on a training set.
"""

import concurrent.futures as concurrent_futures
import datetime
import itertools
import os
from typing import Optional, Sequence, Union

from absl import app
from absl import flags
from absl import logging
import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from precondition.oco import algorithms
from precondition.oco import datasets
from precondition.oco import train

jax.config.update('jax_enable_x64', True)


_SKETCH_SIZE = flags.DEFINE_integer(
    'sketch_size', 0, 'sketch size for approximate full-matrix algorithms'
)


_PARALLEL = flags.DEFINE_integer(
    'parallel', 16, 'number of threads for launching jax programs'
)


_DATASET = flags.DEFINE_enum(
    'dataset', 'a9a', datasets.SUPPORTED_DATASETS, 'dataset to run on'
)


_ALGS = flags.DEFINE_multi_enum_class(
    'alg',
    list(algorithms.Algorithm),
    algorithms.Algorithm,
    'which algorithms to evaluate',
)


_DELTA = flags.DEFINE_multi_float(
    'delta',
    [],
    'initial diagonal regularization hparams. Must be unset if --use_best_from'
    ' is set (and vice versa).',
)


_LR = flags.DEFINE_multi_float(
    'lr',
    [],
    'lr hparams. Must be unset if --use_best_from is set (and vice versa).',
)


_USE_BEST_FROM = flags.DEFINE_multi_string(
    'use_best_from',
    None,
    'initialize best hparams for each algorithm using results from these'
    ' directories. If set, --delta and --lr must not be.',
)


_DIR = flags.DEFINE_string(
    'save_dir',
    None,
    'save results to this directory suffix, by default, HH:MM:SS',
)


_SKETCHING_ALGS = [
    algorithms.Algorithm.RFD_SON,
    algorithms.Algorithm.FD_SON,
    algorithms.Algorithm.ADA_FD,
    algorithms.Algorithm.S_ADA,
]


def _validate_flags() -> bool:
  """Raises if flags improperly set, returns whether sweeping."""
  if (
      any(alg in _SKETCHING_ALGS for alg in _ALGS.value)
      and _SKETCH_SIZE.value <= 1
  ):
    raise ValueError('sketch size must be at least 2')
  hparams_set = _DELTA.value or _LR.value
  if hparams_set and not (_DELTA.value and _LR.value):
    raise ValueError('if one of --delta/--lr is set, the other must be')
  if _USE_BEST_FROM.value and hparams_set:
    raise ValueError(
        '--delta/--lr may only be set if --use_best_from is not '
        '(and vice versa)'
    )
  if not _USE_BEST_FROM.value and not hparams_set:
    raise ValueError(
        'at least one of --delta/--lr or --use_best_from must be set'
    )
  if not _DIR.value:
    raise ValueError('require a directory to save outputs')
  return bool(hparams_set)


def _make_directory():
  """Derive which directory to save in, ensure it doesn't exist, make it."""
  suffix = _DIR.value
  now = datetime.datetime.now()
  time = now.strftime('%H:%M:%S')
  date = now.date()
  directory = f'{suffix}/{date}/{time}'
  os.makedirs(directory)
  return directory


def main(argv: ...) -> None:
  del argv

  is_sweep = _validate_flags()
  directory = _make_directory()
  flagfile = os.path.join(directory, 'flagfile.txt')
  with open(flagfile, 'w') as f:
    f.write(flags.FLAGS.flags_into_string())

  dataset_name = _DATASET.value
  # Load dataset to print top-level info and for joblib to cache it in /tmp
  dataset = datasets.load_dataset(dataset_name)
  logging.info('loaded dataset %s with dims %s', dataset_name, dataset.x.shape)

  sketch_size = _SKETCH_SIZE.value

  hparams = []
  if is_sweep:
    for alg, lr, delta in itertools.product(
        _ALGS.value, _LR.value, _DELTA.value
    ):
      hparam = algorithms.HParams(
          delta,
          lr,
          sketch_size if alg in _SKETCHING_ALGS else 0,
          alg,
      )
      hparams.append(hparam)
  else:
    dfs = []
    for previous_directory in _USE_BEST_FROM.value:
      dfs.append(_read_pandas(previous_directory, dataset_name, sketch_size))
    df = pd.concat(dfs, axis=0)
    df.sort_values('loss', inplace=True)
    df.drop_duplicates('alg', inplace=True)
    algs = df.alg.unique()
    hparams = []
    for alg in _ALGS.value:
      if alg.name not in algs:
        raise ValueError(f'missing {alg} in --use_best_from')
      row = df.loc[df.alg == alg.name].T.squeeze()
      hparam = algorithms.HParams(
          row.delta,
          row.lr,
          sketch_size if alg in _SKETCHING_ALGS else 0,
          alg,
      )
      hparams.append(hparam)

  assert hparams

  nobs = 100
  logging.info('generated %s trials to run with %s obs', len(hparams), nobs)
  logging.info('%s distinct algs: %s', len(_ALGS.value), _ALGS.value)
  total = len(hparams)
  args = [
      {'idx': i, 'total': total, 'dataset': dataset_name, 'hparam': hparam,
       'nobs': nobs,}
      for i, hparam in enumerate(hparams)
  ]
  # python threading suffices b/c Jax itself is async
  executor = concurrent_futures.ThreadPoolExecutor(_PARALLEL.value)
  histories = list(executor.map(lambda kwargs: _run_oco(**kwargs), args))
  for e in histories:
    if isinstance(e, Exception):
      raise e
  logging.info('all %s hparam tunings complete', total)

  result_df = _make_pandas(hparams, dataset_name, sketch_size, histories)
  result_df.sort_values('loss', inplace=True)
  _save_pandas(directory, result_df)
  best_df = result_df.drop_duplicates('alg', inplace=False)
  assert best_df is not None
  best_txt = best_df.drop(columns='history').to_string(index=False)
  logging.info('completed runs, results\n%s', best_txt)

  with open(os.path.join(directory, 'best.txt'), 'w') as f:
    print(best_txt, file=f)

  cs = itertools.cycle('rbcgk')
  lss = itertools.cycle(['--', '-', ':'])
  for loss_type in ['loss', 'extra_loss']:
    assert best_df is not None
    for h, alg, ls, c in zip(best_df.history, best_df.alg, lss, cs):
      if h is None:
        continue
      plt.plot(h['n'][1:], h[loss_type][1:] / h['n'][1:], label=alg, ls=ls, c=c)

    loss_name = '0-1 loss' if loss_type == 'extra_loss' else 'logloss'
    plt.xlabel('examples')
    plt.ylabel(f'cumulative {loss_name}')
    plt.title(f'cumulative {loss_name}')
    plt.legend()
    ln = loss_name.replace(' ', '-')
    with open(os.path.join(directory, f'plot-{ln}.pdf'), 'wb') as f:
      plt.savefig(f, format='pdf', bbox_inches='tight')
    plt.clf()

  logging.info(
      'all results saved in %s/{results.pkl,plot*.pdf,flagfile.txt,best.txt}',
      directory,
  )


def _run_oco(
    idx: int, total: int, dataset: str, hparam: algorithms.HParams, nobs: int
) -> Union[Optional[algorithms.NpState], Exception]:
  """Run online convex algorithm on a worker process."""
  logging.info('job %04d of %04d starting', idx, total)
  try:
    history = train.run_dataset(dataset, nobs, hparam, datasets.incorrect)
    logging.info('job %04d of %04d done', idx, total)
    return algorithms.as_np(history)
  except FloatingPointError:
    logging.info('job %04d of %04d inf', idx, total)
    return None
  except Exception as e:  # pylint: disable=broad-exception-caught
    logging.info('job %04d of %04d errored', idx, total)
    return e

# Disable for Numpy and Pandas containers.
# pylint: disable=g-explicit-length-test


def _read_pandas(
    path: str, dataset_name: str, sketch_size: int
) -> pd.DataFrame:
  """Read and validate dataframe from previous run."""
  path = os.path.join(path, 'results.pkl')
  with open(path, 'rb') as f:
    df = pd.read_pickle(f)
  assert len(df) > 0, (path, len(df))
  expected = [
      'alg',
      'lr',
      'delta',
      'loss',
      'acc',
      'dataset',
      'sketch_size',
      'history',
  ]
  assert set(df.columns) == set(expected), df.columns
  assert df.dataset.nunique(dropna=False) == 1, (path, df.dataset.unique())
  stored_dataset = list(df.dataset.unique())[0]
  assert df.sketch_size.nunique(dropna=True) == 1, (
      path,
      df.sketch_size.unique(),
  )
  sketch_sizes = [x for x in df.sketch_size.unique() if not pd.isnull(x)]
  if sketch_sizes:
    stored_sketch_size = sketch_sizes[0]
    assert sketch_size == stored_sketch_size, (
        path,
        sketch_size,
        stored_sketch_size,
    )
  assert dataset_name in stored_dataset or stored_dataset in dataset_name, (
      path,
      stored_dataset,
      dataset_name,
  )
  logging.info(
      'extracted %s runs from %s with dataset %s matching %s at sketch size %s',
      len(df),
      path,
      stored_dataset,
      dataset_name,
      sketch_size,
  )
  return df


def _make_pandas(
    hparams: Sequence[algorithms.HParams],
    dataset_name: str,
    sketch_size: int,
    histories: Sequence[Optional[algorithms.NpState]],
) -> pd.DataFrame:
  """Generate a dataframe with all serializable run information."""
  records = []
  for hparam, history in zip(hparams, histories):
    if history is None:
      loss = np.inf
      acc = 0.0
    else:
      loss = history['loss'][-1] / history['n'][-1]
      acc = 1.0 - history['extra_loss'][-1] / history['n'][-1]
    sketch_size_used = (
        sketch_size if hparam.algorithm in _SKETCHING_ALGS else np.nan
    )
    record = {
        'alg': hparam.algorithm.name,
        'lr': hparam.lr,
        'delta': hparam.delta,
        'loss': loss,
        'acc': acc,
        'dataset': dataset_name,
        'sketch_size': sketch_size_used,
        'history': history,
    }
    records.append(record)
  return pd.DataFrame.from_records(records)


def _save_pandas(path: str, df: pd.DataFrame) -> None:
  """Read and validate dataframe from previous run."""
  path = os.path.join(path, 'results.pkl')
  with open(path, 'wb') as f:
    df.to_pickle(f)


if __name__ == '__main__':
  app.run(main)
