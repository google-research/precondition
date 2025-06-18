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

import copy
import itertools
from typing import cast
from absl import logging

import tensorflow_datasets as tfds
import numpy as np

class DartboardDeterministicTrainingBatchGenerator:
  def __init__(self, train_ds_builders, batch_size, num_weights=2, num_iterations=100):
    self.train_ds_builders = train_ds_builders
    self.batch_size = batch_size
    self.num_weights = num_weights
    self.num_iterations = num_iterations
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
    self.avg_index = 0
    self.indices = []

  def prepare_for_training(self, avg_weights, new_unnormalized_weights):
    """Prepare for training."""
    self.indices = [0 for _ in range(self.num_weights)]
    logging.info(f'new_unnormalized_weights: {new_unnormalized_weights}')
    logging.info(f'avg_weights: {self.avg_weights}')
    nochange_prob = new_unnormalized_weights / self.avg_weights
    nochange_prob = np.minimum(nochange_prob, 1)
    logging.info(f'nochange_prob: {nochange_prob}')
    self.unnormalized_weights = new_unnormalized_weights
    self.avg_weights = avg_weights
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
    return 1

  def get_next_batch(self):
    logging.info(f'Getting next batch, batch_size={self.batch_size}, weights_list={self.weights_list}')
    factors = np.ones(self.batch_size)
    cur_examples = self.examples[self.avg_index:(self.avg_index+self.batch_size)]
    cur_input_tokens_batch = np.asarray([[example.input_tokens] for example in cur_examples])
    cur_input_mask_batch = np.asarray([[example.target_mask] for example in cur_examples])
    self.avg_index += self.batch_size
    return factors, cur_input_tokens_batch, cur_input_mask_batch

  def get_next_batch_special(self, index, delta):
    logging.info(f'Getting next batch, batch_size={self.batch_size}, weights_list={self.weights_list}')
    cur_batch = []
    while len(cur_batch) < self.batch_size and self.indices[index] < len(self.examples):
      if self.sample_choices[self.indices[index]] == index:
        cur_batch.append(self.examples[self.indices[index]])
        self.indices[index] += 1
    if len(cur_batch) == 0:
      return False
    cur_ind = 0
    while(len(cur_batch) < self.batch_size):
      cur_batch.append(cur_batch[cur_ind])
      cur_ind += 1
    cur_input_tokens_batch = np.asarray([[example.input_tokens] for example in cur_batch])
    cur_input_mask_batch = np.asarray([[example.target_mask] for example in cur_batch])

    factors = np.ones(self.batch_size) * delta/np.sqrt(self.batch_size)
    return factors, cur_input_tokens_batch, cur_input_mask_batch
