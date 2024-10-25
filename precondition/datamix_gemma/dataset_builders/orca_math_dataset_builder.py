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

"""Dataset builder for the Orca Math dataset."""

import enum as Enum

from absl import logging
import jax.dlpack
from precondition.datamix_gemma.dataset_builders import dataset_builder
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetSplit(Enum.Enum):
  TRAIN = 'train'


class OrcaMathDatasetBuilder(dataset_builder.DatasetBuilder):
  """Dataset builder for the Orca Math dataset."""

  N_ITEMS = {DatasetSplit.TRAIN: 200035}

  #BUFFER_SIZE_SHUFFLE = 10_000
  BUFFER_SIZE_SHUFFLE = 100
  QUESTION_PREFIX = 'Question: \n'
  QUESTION_SUFFIX = '\n'
  #TRANSLATION_PREFIX = 'Translate this into French:\n'
  #TRANSLATION_SUFFIX = '\n'

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
        DatasetSplit.TRAIN: tfds.load(
            'huggingface:microsoft__orca_math_word_problems_200k', split='train'
        ),
    }
    logging.info(
        'orca math size: %s',
        self._base_data[DatasetSplit.TRAIN].cardinality().numpy(),
    )
    self._max_seq_len = max_seq_len

  def _tokenize_question(self, example: tf.Tensor):
    """Tokenization function for the Question."""
    return self._tokenizer.tokenize_tf_op(
        example,
        prefix=self.QUESTION_PREFIX,
        suffix=self.QUESTION_SUFFIX,
        add_eos=False,
    )

  def _tokenize_response(self, example: tf.Tensor):
    """Tokenization function for the Response."""
    return self._tokenizer.tokenize_tf_op(
        example,
        add_eos=True,
    )

  def _to_training_input(
      self,
      question_tokens: jax.Array,
      answer_tokens: jax.Array,
  ):
    """Build a training input from a tuple of source and destination tokens."""

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = tf.concat(
        [question_tokens, answer_tokens], axis=0
    )

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    question_mask = tf.zeros_like(question_tokens, dtype=tf.bool)
    answer_mask = tf.ones_like(answer_tokens, dtype=tf.bool)
    mask = tf.concat([question_mask, answer_mask], axis=0)

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
            self._tokenize_question(x['question']),
            self._tokenize_response(x['answer'])
        )
    )
    ds = ds.map(self._to_training_input)
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    ds = ds.shuffle(buffer_size=self.BUFFER_SIZE_SHUFFLE)
    #ds = ds.repeat(num_epochs)
    #ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def get_validation_dataset(self, batch_size: int):
    """Build the validation dataset."""

    ds = self._base_data[DatasetSplit.TRAIN].map(
        lambda x: (
            self._tokenize_question(x['question']),
            self._tokenize_response(x['answer'])
        )
    )
    ds = ds.map(self._to_training_input)
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    return ds
