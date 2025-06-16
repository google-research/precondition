# Copyright 2025 The precondition Authors.
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

"""Random baseline."""

import copy

from absl import logging
import jax
import numpy as np
from precondition.datamix_gemma import training_loop
from precondition.datamix_gemma.evals import eval as eval_lib
from precondition.datamix_gemma.training_batch_generators import training_batch_generator


def random_simplex(n):
  """Return uniformly random vector in the n-simplex."""
  k = np.random.exponential(scale=1.0, size=n)
  return k / np.sum(k)


def random_baseline(
    eval_obj: eval_lib.Eval,
    train_obj: training_loop.TrainingLoop,
    training_batch_generator_obj: training_batch_generator.TrainingBatchGenerator,
    init_params,
    num_iterations=100,
):
  """Random baseline."""
  for _ in range(num_iterations):
    random_weights = random_simplex(len(training_batch_generator_obj.train_ds_builders))
    cur_params = copy.deepcopy(init_params)
    cur_params = jax.device_get(cur_params)
    trained_params = train_obj.train_loop(
        params={'params': cur_params}, get_next_batch_fn=training_batch_generator_obj.get_next_batch
    )
    score = eval_obj.evaluate(trained_params['params'])
    logging.info(f'score: {score}')
    for i in range(len(random_weights)):
      logging.info(f'weights_{str(i)}: {random_weights[i]}')
