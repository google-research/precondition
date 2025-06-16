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

"""TrainingBatchGenerator."""

class TrainingBatchGenerator:
  """TrainingBatchGenerator."""

  def __init__(self, train_ds_builders, batch_size, num_weights=2, num_iterations=100):
    self.train_ds_builders = train_ds_builders
    self.batch_size = batch_size
    self.num_weights = num_weights
    self.num_iterations = num_iterations

  #def prepare_for_training(self, weights_1, weights_2):
  #  """Prepare for training."""
  #  raise NotImplementedError()
  def prepare_for_training(self, weights_list, new_unnormalized_weights):
    """Prepare for training."""
    raise NotImplementedError()

  def get_next_batch(self, index):
    raise NotImplementedError()
