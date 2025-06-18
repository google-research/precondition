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

"""Dataset builder for the MTNT dataset."""

import enum as Enum

import jax.dlpack
from precondition.datamix_gemma.dataset_builders import dataset_builder
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetSplit(Enum.Enum):

  TRAIN = 'train'
  VALIDATION = 'validation'


class MTNTDatasetBuilder(dataset_builder.DatasetBuilder):
  """Dataset builder for the MTNT dataset."""

  N_ITEMS = {DatasetSplit.TRAIN: 35_692, DatasetSplit.VALIDATION: 811}

  BUFFER_SIZE_SHUFFLE = 10_000
  TRANSLATION_PREFIX = 'Translate this into French:\n'
  TRANSLATION_SUFFIX = '\n'

  def __init__(
      self, tokenizer: gemma_tokenizer.GemmaTokenizer, max_seq_len: int
  ):
    """Constructor.

    Args:
      tokenizer: Gemma tokenizer to use.
      max_seq_len: size of each sequence in a given batch.
    """
    self._tokenizer = tokenizer
    self._base_data = {
        DatasetSplit.TRAIN: tfds.load('mtnt/en-fr', split='train'),
        DatasetSplit.VALIDATION: tfds.load('mtnt/en-fr', split='valid'),
    }
    self._max_seq_len = max_seq_len

  def _tokenize_source(self, example: tf.Tensor) -> tf.Tensor:
    """Tokenization function for the source."""
    res = self._tokenizer.tokenize_tf_op(
        example,
        prefix=self.TRANSLATION_PREFIX,
        suffix=self.TRANSLATION_SUFFIX,
        add_eos=False,
    )
    return res

  def _tokenize_destination(self, example: tf.Tensor):
    """Tokenization function for the French translation."""
    return self._tokenizer.tokenize_tf_op(example, add_eos=True)

  def _to_training_input(
      self,
      src_tokens: jax.Array,
      dst_tokens: jax.Array,
  ):
    """Build a training input from a tuple of source and destination tokens."""

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = tf.concat([src_tokens, dst_tokens], axis=0)

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    q_mask = tf.zeros_like(src_tokens, dtype=tf.bool)
    a_mask = tf.ones_like(dst_tokens, dtype=tf.bool)
    mask = tf.concat([q_mask, a_mask], axis=0)

    # If the output tokens sequence is smaller than the target sequence size,
    # then pad it with pad tokens.
    tokens = self._pad_up_to_max_len(tokens, self._tokenizer.pad_id)

    # Don't want to perform the backward pass on the pad tokens.
    mask = self._pad_up_to_max_len(mask, False)
    return dataset_builder.TrainingInput( #type: ignore
        input_tokens=tokens, #type:ignore
        target_mask=mask,  #type:ignore
    )# type: ignore

  def get_train_dataset(self, batch_size: int, num_epochs: int):
    """Build the training dataset."""

    ds = self._base_data[DatasetSplit.TRAIN].map(
        lambda x: (
            self._tokenize_source(x['src']),
            self._tokenize_destination(x['dst']),
        )
    )
    ds = ds.map(self._to_training_input)
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    ds = ds.shuffle(buffer_size=self.BUFFER_SIZE_SHUFFLE)
    ds = ds.repeat(num_epochs)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def get_validation_dataset(self, batch_size: int):
    """Build the validation dataset."""

    ds = self._base_data[DatasetSplit.VALIDATION].map(
        lambda x: (
            self._tokenize_source(x['src']),
            self._tokenize_destination(x['dst']),
        )
    )
    ds = ds.map(self._to_training_input)
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds
