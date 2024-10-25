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

"""FixedDatasetImportanceWeightingTrainingBatchGenerator."""

from absl import logging
import numpy as np
from precondition.datamix_gemma.training_batch_generators import training_batch_generator
import tensorflow_datasets as tfds


class FixedDatasetImportanceWeightingTrainingBatchGenerator(
    training_batch_generator.TrainingBatchGenerator
):
  """FixedDatasetImportanceWeightingTrainingBatchGenerator."""

  def __init__(self, train_ds_builders, batch_size, num_weights=2, num_iterations=100):
    super().__init__(train_ds_builders, batch_size, num_weights, num_iterations)
    self.training_iters = []
    for dataset_builder_obj in self.train_ds_builders:
      cur_iter = iter(
          tfds.as_numpy(
              dataset_builder_obj.get_train_dataset(
                  batch_size=batch_size, num_epochs=1
              )
          )
      )
      self.training_iters.append(cur_iter)
    self.examples = []
    num_datasets = len(self.training_iters)
    self.sample_choices = np.random.choice(
        num_datasets,
        size=self.batch_size * self.num_iterations,
        p=np.ones(num_datasets)/num_datasets,
    )
    for i in range(self.batch_size * num_iterations):
      try:
        self.examples.append(next(self.training_iters[self.sample_choices[i]]))
      except StopIteration:
        self.training_iters[self.sample_choices[i]] = iter(
            tfds.as_numpy(
                self.train_ds_builders[self.sample_choices[i]].get_train_dataset(
                    batch_size=self.batch_size, num_epochs=1
                )
            )
        )
        self.examples.append(next(self.training_iters[self.sample_choices[i]]))

    #self.input_tokens_batch = np.asarray([[example.input_tokens] for example in self.examples])
    #self.input_mask_batch = np.asarray([[example.target_mask] for example in self.examples])
    self.weights_list = []
    self.indices = []
    self.factors = []
    #self.avg_weights = np.zeros(len(self.weights_list[0]))

  def prepare_for_training(self, weights_list, new_unnormalized_weights):
    """Prepare for training."""
    self.weights_list = weights_list
    self.indices = [0 for _ in range(self.num_weights)]
    self.factors = [np.zeros(self.batch_size * self.num_iterations) for _ in range(self.num_weights)]
    for i in range(self.num_weights):
      for j in range(self.batch_size * self.num_iterations):
        self.factors[i][j] = self.weights_list[i][self.sample_choices[j]]
      self.factors = (self.factors[i] / np.sum(self.factors[i])) * len(self.factors[i])
    return 1

  def get_next_batch(self, index):
    logging.info(f'Getting next batch, batch_size={self.batch_size}, weights_list={self.weights_list}')
    logging.info(f'sample choices len: {len(self.sample_choices)}, examples len: {len(self.examples)}')
    cur_factors = self.factors[index][self.indices[index]:(self.indices[index]+self.batch_size)]
    cur_examples = self.examples[self.indices[index]:(self.indices[index]+self.batch_size)]
    cur_input_tokens_batch = np.asarray([[example.input_tokens] for example in cur_examples])
    cur_input_mask_batch = np.asarray([[example.target_mask] for example in cur_examples])
    self.indices[index] += self.batch_size
    return cur_factors, cur_input_tokens_batch, cur_input_mask_batch
