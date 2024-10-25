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

"""ImportanceWeightingTrainingBatchGenerator."""

import itertools

from absl import logging
import numpy as np
from precondition.datamix_gemma.training_batch_generators import training_batch_generator
import tensorflow_datasets as tfds


class ImportanceWeightingTrainingBatchGenerator(
    training_batch_generator.TrainingBatchGenerator
):
  """ImportanceWeightingTrainingBatchGenerator."""

  def __init__(
      self, train_ds_builders, batch_size, num_weights=2, num_iterations=100
  ):
    super().__init__(train_ds_builders, batch_size, num_weights, num_iterations)
    self.training_iters_lists = []
    for _ in range(self.num_weights):
      self.training_iters_lists.append([])

    for dataset_builder_obj in self.train_ds_builders:
      cur_iter = iter(
          tfds.as_numpy(
              dataset_builder_obj.get_train_dataset(
                  batch_size=batch_size, num_epochs=1
              )
          )
      )
      iter_list = itertools.tee(cur_iter, self.num_weights)
      for i in range(self.num_weights):
        self.training_iters_lists[i].append(iter_list[i])
    #self.avg_weights = np.zeros(len(self.weights_list[0]))
    self.avg_weights = []
    self.weights_list = []
    self.sample_choices = []

  def prepare_for_training(self, weights_list, new_unnormalized_weights):
    """Prepare for training."""
    self.weights_list = weights_list
    self.avg_weights = np.zeros(len(self.weights_list[0]))
    for i in range(len(self.weights_list)):
      self.avg_weights += self.weights_list[i]
    self.avg_weights /= len(self.weights_list)

    logging.info(f'Avg weights: {self.avg_weights}')
    self.sample_choices = np.random.choice(
        len(self.avg_weights),
        size=self.batch_size,
        p=self.avg_weights,
    )
    return 1

  def get_next_batch(self, index):
    logging.info('Getting next batch')
    training_iters = self.training_iters_lists[index]
    input_tokens_batch = []
    input_mask_batch = []
    factors = np.zeros(self.batch_size)
    for i in range(self.batch_size):
      sample_choice = self.sample_choices[i]
      try:
        cur_example = next(training_iters[sample_choice])
      except StopIteration:
        training_iters[sample_choice] = iter(
            tfds.as_numpy(
                self.train_ds_builders[sample_choice].get_train_dataset(
                    batch_size=self.batch_size, num_epochs=1
                )
            )
        )
        cur_example = next(training_iters[sample_choice])
      factors[i] = self.weights_list[index][sample_choice] / self.avg_weights[sample_choice] #pytype: disable=attribute-error
      input_tokens_batch.append(np.asarray([cur_example.input_tokens]))
      input_mask_batch.append(np.asarray([cur_example.target_mask]))
    factors *= len(factors)/np.sum(factors)

    return factors, input_tokens_batch, input_mask_batch
