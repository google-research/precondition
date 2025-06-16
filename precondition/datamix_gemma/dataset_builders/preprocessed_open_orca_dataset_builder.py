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

"""Dataset builder for the Open Orca dataset."""

import enum as Enum

from precondition.datamix_gemma.dataset_builders import dataset_builder
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
import tensorflow as tf


open_orca_path = '/home/shivguptashi/open_orca/open_orca_data.tfrecord'

class DatasetSplit(Enum.Enum):
  TRAIN = 'train'


class PreprocessedOpenOrcaDatasetBuilder(dataset_builder.DatasetBuilder):
  """Dataset builder for the Open Orca dataset."""

  def __init__(
      self, tokenizer: gemma_tokenizer.GemmaTokenizer, max_seq_len: int
  ):
    """Constructor.

    Args:
      tokenizer: Gemma tokenizer to use.
      max_seq_len: size of each sequence in a given batch.
    """
    self._tokenizer = tokenizer
    self._base_data = tf.data.TFRecordDataset(
        [open_orca_path], num_parallel_reads=tf.data.AUTOTUNE
    )
    self._max_seq_len = max_seq_len

  def _to_training_input(
      self,
      input_tokens,
      target_mask,
  ):
    return dataset_builder.TrainingInput(  # type: ignore
        input_tokens=input_tokens,  # type:ignore
        target_mask=target_mask,  # type:ignore
    )  # type: ignore

  def _decode_fn(self, record_bytes):
    parsed_features = tf.io.parse_example(
        record_bytes,
        {
            'input_tokens': tf.io.FixedLenFeature((), tf.string),
            'target_mask': tf.io.FixedLenFeature((), tf.string),
        },
    )
    decoded = {
        'input_tokens': tf.io.decode_raw(
            parsed_features['input_tokens'], out_type=tf.int32
        ),
        'target_mask': tf.io.decode_raw(
            parsed_features['target_mask'], out_type=tf.bool
        ),
    }
    return {
        'input_tokens': self._pad_up_to_max_len(
            decoded['input_tokens'], self._tokenizer.pad_id
        ),
        'target_mask': self._pad_up_to_max_len(
            decoded['target_mask'], False
        ),
    }

  def get_train_dataset(self, batch_size: int, num_epochs: int):
    """Build the training dataset."""
    ds = self._base_data.map(
        self._decode_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(
        lambda x: self._to_training_input(x['input_tokens'], x['target_mask']),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds

  def get_validation_dataset(self, batch_size: int):
    ds = self._base_data.map(
        self._decode_fn, num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(
        lambda x: self._to_training_input(x['input_tokens'], x['target_mask']),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds
