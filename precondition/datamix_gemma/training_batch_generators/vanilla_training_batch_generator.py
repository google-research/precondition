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

"""VanillaTrainingBatchGenerator."""

from absl import logging
import numpy as np
from precondition.datamix_gemma.training_batch_generators import training_batch_generator
import tensorflow_datasets as tfds


class VanillaTrainingBatchGenerator(
    training_batch_generator.TrainingBatchGenerator
):
  """VanillaTrainingBatchGenerator."""

  def __init__(self, train_ds_builders, batch_size, num_weights=2, num_iterations=100):
    super().__init__(train_ds_builders, batch_size, num_weights, num_iterations)
    self.training_iters = []
    for dataset_builder_obj in self.train_ds_builders:
      self.training_iters.append(
          iter(
              tfds.as_numpy(
                  dataset_builder_obj.get_train_dataset(
                      batch_size=batch_size, num_epochs=1
                  )
              )
          )
      )
    self.weights_list = []

  def prepare_for_training(self, weights_list, new_unnormalized_weights):
    """Prepare for training."""
    self.weights_list = weights_list
    #gradient discount factor
    return 1

  def get_next_batch(self, index):
    weights = self.weights_list[index]
    input_tokens_batch = []
    input_mask_batch = []
    factors = []
    for _ in range(self.batch_size):
      cur_ind = np.random.choice(len(self.training_iters), p=weights)
      logging.info(f'cur_ind: {cur_ind}')
      try:
        cur_example = next(self.training_iters[cur_ind])
      except StopIteration:
        self.training_iters[cur_ind] = iter(
            tfds.as_numpy(
                self.train_ds_builders[cur_ind].get_train_dataset(
                    batch_size=self.batch_size, num_epochs=1
                )
            )
        )
        cur_example = next(self.training_iters[cur_ind])
      input_tokens_batch.append(np.asarray([cur_example.input_tokens]))
      input_mask_batch.append(np.asarray([cur_example.target_mask]))
      factors.append(1)
    return factors, input_tokens_batch, input_mask_batch
