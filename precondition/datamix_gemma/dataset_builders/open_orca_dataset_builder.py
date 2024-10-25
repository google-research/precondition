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

"""Dataset builder for the Open Orca dataset."""

import enum as Enum

from absl import logging
import jax.dlpack
from precondition.datamix_gemma.dataset_builders import dataset_builder
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetSplit(Enum.Enum):
  TRAIN = 'train'


class OpenOrcaDatasetBuilder(dataset_builder.DatasetBuilder):
  """Dataset builder for the Open Orca dataset."""

  N_ITEMS = {DatasetSplit.TRAIN: 2914896}

  #BUFFER_SIZE_SHUFFLE = 10_000
  BUFFER_SIZE_SHUFFLE = 100
  SYSTEM_PREFIX = 'System: \n'
  SYSTEM_SUFFIX = '\n'
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
            'huggingface:open_orca__openorca', split='train'
        ),
    }
    logging.info(
        'open orca size: %s',
        self._base_data[DatasetSplit.TRAIN].cardinality().numpy(),
    )
    self._max_seq_len = max_seq_len

  def _tokenize_system(self, example: tf.Tensor) -> tf.Tensor:
    """Tokenization function for the system prompt."""
    res = self._tokenizer.tokenize_tf_op(
        example,
        prefix=self.SYSTEM_PREFIX,
        suffix=self.SYSTEM_SUFFIX,
        add_eos=False,
    )
    return res

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
      system_tokens: jax.Array,
      question_tokens: jax.Array,
      response_tokens: jax.Array,
  ):
    """Build a training input from a tuple of source and destination tokens."""

    # The input sequence fed to the model is simply the concatenation of the
    # source and the destination.
    tokens = tf.concat(
        [system_tokens, question_tokens, response_tokens], axis=0
    )

    # To prevent the model from updating based on the source (input)
    # tokens, add a target mask to each input.
    system_mask = tf.zeros_like(system_tokens, dtype=tf.bool)
    question_mask = tf.zeros_like(question_tokens, dtype=tf.bool)
    response_mask = tf.ones_like(response_tokens, dtype=tf.bool)
    mask = tf.concat([system_mask, question_mask, response_mask], axis=0)

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
            self._tokenize_system(x['system_prompt']),
            self._tokenize_question(x['question']),
            self._tokenize_response(x['response'])
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(self._to_training_input,
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    ds = ds.shuffle(buffer_size=self.BUFFER_SIZE_SHUFFLE)
    #ds = ds.repeat(num_epochs)
    #ds = ds.batch(batch_size, drop_remainder=True)
    return ds

  def get_validation_dataset(self, batch_size: int):
    """Build the validation dataset."""

    # Same steps as in `get_train_dataset`, but without shuffling and
    # repetition.
    # ds = self._base_data[DatasetSplit.VALIDATION].map(
    #    lambda x: (self._tokenize_source(x['src']),
    #               self._tokenize_destination(x['dst'])))
    ds = self._base_data[DatasetSplit.TRAIN].map(
        lambda x: (
            self._tokenize_system(x['system_prompt']),
            self._tokenize_question(x['question']),
            self._tokenize_response(x['response']),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(self._to_training_input, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    # ds = ds.batch(batch_size, drop_remainder=True)
    return ds
    # ds = [self._to_training_input(x, y) for x, y in ds]
    # print('here3:', ds)
    # ds = [x for x in ds if tf.shape(x.input_tokens)[0] <= self._max_seq_len]
    # ds = [ds[i : i + batch_size] for i in range(0, len(ds), batch_size)]
