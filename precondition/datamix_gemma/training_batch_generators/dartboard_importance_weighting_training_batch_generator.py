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

"""DartboardImportanceWeightingTrainingBatchGenerator."""
import copy
import itertools
from absl import logging

import tensorflow_datasets as tfds
import numpy as np
from precondition.datamix_gemma.training_batch_generators import training_batch_generator


class DartboardImportanceWeightingTrainingBatchGenerator(
    training_batch_generator.TrainingBatchGenerator
):
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
    self.num_datasets = len(self.train_ds_builders)
    self.sample_choices = np.random.choice(
        self.num_datasets,
        size=self.batch_size * num_iterations,
        p=np.ones(self.num_datasets)/self.num_datasets,
    )
    self.examples = []
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
    self.unnormalized_weights = np.ones(self.num_datasets)/self.num_datasets
    self.avg_weights = np.ones(self.num_datasets)/self.num_datasets
    self.weights_list = [np.ones(self.num_datasets)/self.num_datasets for _ in range(num_weights)]
    self.indices = []
    self.factors = []

  def prepare_for_training(self, weights_list, new_unnormalized_weights):
    """Prepare for training."""
    self.indices = [0 for _ in range(self.num_weights)]
    logging.info(f'new_unnormalized_weights: {new_unnormalized_weights}')
    logging.info(f'avg_weights: {self.avg_weights}')
    nochange_prob = new_unnormalized_weights / self.avg_weights
    nochange_prob = np.minimum(nochange_prob, 1)
    logging.info(f'nochange_prob: {nochange_prob}')
    self.unnormalized_weights = new_unnormalized_weights
    self.weights_list = weights_list
    self.avg_weights = np.zeros(len(self.weights_list[0]))
    for i in range(len(self.weights_list)):
      self.avg_weights += self.weights_list[i]
    self.avg_weights /= len(self.weights_list)
    for i in range(self.batch_size * self.num_iterations):
      change = np.random.choice(2, p=[nochange_prob[self.sample_choices[i]], 1-nochange_prob[self.sample_choices[i]]])
      if change:
        self.sample_choices[i] = np.random.choice(
            len(self.avg_weights),
            p=self.avg_weights,
        )
        try:
          self.examples[i] = next(self.training_iters[self.sample_choices[i]])
        except StopIteration:
          self.training_iters[self.sample_choices[i]] = iter(
              tfds.as_numpy(
                  self.train_ds_builders[self.sample_choices[i]].get_train_dataset(
                      batch_size=self.batch_size, num_epochs=1
                  )
              )
          )
          self.examples[i] = next(self.training_iters[self.sample_choices[i]])
    self.factors = [np.zeros(self.batch_size * self.num_iterations) for _ in range(self.num_weights)]
    for i in range(self.num_weights):
      for j in range(self.batch_size * self.num_iterations):
        self.factors[i][j] = self.weights_list[i][self.sample_choices[j]] / self.avg_weights[self.sample_choices[j]]
      self.factors[i] = (self.factors[i] / np.sum(self.factors[i])) * len(self.factors[i])
    return 1

  def get_next_batch(self, index):
    logging.info(f'Getting next batch, batch_size={self.batch_size}, weights_list={self.weights_list}')
    cur_factors = self.factors[index][self.indices[index]:(self.indices[index]+self.batch_size)]
    cur_examples = self.examples[self.indices[index]:(self.indices[index]+self.batch_size)]
    cur_input_tokens_batch = np.asarray([[example.input_tokens] for example in cur_examples])
    cur_input_mask_batch = np.asarray([[example.target_mask] for example in cur_examples])
    self.indices[index] += self.batch_size
    logging.info(f'cur_factors: {cur_factors}')
    return cur_factors, cur_input_tokens_batch, cur_input_mask_batch
