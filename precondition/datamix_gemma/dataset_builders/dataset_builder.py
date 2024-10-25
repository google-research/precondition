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

"""Base class for dataset builders."""

import chex
import jax
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
import tensorflow as tf


@chex.dataclass(frozen=True)
class TrainingInput:
  # Input tokens provided to model
  input_tokens: jax.Array

  # A mask that determines which tokens contribute to the target loss
  # calculation
  target_mask: jax.Array


class DatasetBuilder:
  """Base class for dataset builders.

  This class provides the interface for dataset builders.
  """

  def __init__(self, tokenizer: gemma_tokenizer.GemmaTokenizer,
               max_seq_len: int):
    """Constructor.

    Args:
      tokenizer: Gemma tokenizer to use.
      max_seq_len: size of each sequence in a given batch.
    """
    self._tokenizer = tokenizer
    self._max_seq_len = max_seq_len

  def _pad_up_to_max_len(
      self, input_tensor: tf.Tensor, pad_value: int | bool
  ) -> tf.Tensor:
    """Pads the given tensor up to max_seq_len."""
    seq_len = tf.shape(input_tensor)[0]
    to_pad = tf.maximum(0, self._max_seq_len - seq_len)
    return tf.pad(
        input_tensor,
        [[0, to_pad]],
        mode='CONSTANT',
        constant_values=pad_value
    )

  def get_train_dataset(self, batch_size: int, num_epochs: int):
    raise NotImplementedError()

  def get_validation_dataset(self, batch_size: int):
    raise NotImplementedError()
