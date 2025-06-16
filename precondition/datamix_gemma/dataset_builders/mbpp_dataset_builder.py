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

"""MBPP dataset builder."""

import enum as Enum

from precondition.datamix_gemma.dataset_builders import dataset_builder
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
import tensorflow as tf
import tensorflow_datasets as tfds


class DatasetSplit(Enum.Enum):
  TRAIN = 'train'
  TEST = 'test'
  PROMPT = 'prompt'


class MBPPDatasetBuilder(dataset_builder.DatasetBuilder):
  """Dataset builder for the MBPP dataset."""

  #BUFFER_SIZE_SHUFFLE = 10_000
  BUFFER_SIZE_SHUFFLE = 100

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
            'huggingface:mbpp/full', split='train'
        ),
        DatasetSplit.TEST: tfds.load(
            'huggingface:mbpp/full', split='test'
        ),
        DatasetSplit.PROMPT: tfds.load(
            'huggingface:mbpp/full', split='prompt'
        ),
    }
    self._max_seq_len = max_seq_len
    #train_ds = self._base_data[DatasetSplit.TEST]
    prompt_ds = self._base_data[DatasetSplit.PROMPT]
    #train_ds = train_ds.filter(
    #    lambda x: 2 <= tf.cast(x['task_id'], tf.int32) <= 4
    #)
    prompt_ds = prompt_ds.map(
        lambda x: (x['text'],
                   self._generate_tests_string(x['test_list']), x['code']))
    prompt_ds = prompt_ds.map(
        lambda x, y, z: tf.py_function(
            self._generate_training_prompt, [x, y, z], [tf.string]))
    individual_prompts = []
    for prompt in prompt_ds:
      individual_prompts.append(prompt[0].numpy().decode('utf-8'))
    self._train_prompt = '\n'.join(individual_prompts)

  def _generate_eval_prompt(self, prompt, tests_str):
    full_prompt = f'You are an expert Python programmer, and here is your task: {prompt.numpy().decode("utf-8")} Your code should pass these tests:\n\n{tests_str.numpy().decode("utf-8")}\n'  # pylint: disable=line-too-long
    return full_prompt, tests_str.numpy().decode('utf-8')

  def _generate_training_prompt(self, prompt, tests_str, code):
    return f'You are an expert Python programmer, and here is your task: {prompt.numpy().decode("utf-8")} Your code should pass these tests:\n\n{tests_str.numpy().decode("utf-8")}\n[BEGIN]\n{code.numpy().decode("utf-8")}\n[DONE]'  # pylint: disable=line-too-long

  def _generate_tests_string(self, tests_list):
    return tf.strings.reduce_join(tests_list, separator='\n')

  def _generate_full_eval_prompt(self, eval_prompt, tests_str):
    full_prompt = '\n'.join(
        [self._train_prompt, eval_prompt.numpy().decode('utf-8')]
    )
    return full_prompt, tests_str.numpy().decode('utf-8')

  def get_test_dataset(self):
    ds = self._base_data[DatasetSplit.TEST].filter(
        lambda x: 11 <= x['task_id'] <= 510
    )
    ds = ds.map(
        lambda x: (x['text'], self._generate_tests_string(x['test_list']))
    )
    ds = ds.map(
        lambda x, y: tf.py_function(
            self._generate_eval_prompt, [x, y], [tf.string, tf.string]
        )
    )
    ds = ds.map(
        lambda x, y: tf.py_function(
            self._generate_full_eval_prompt, [x, y], [tf.string, tf.string]
        )
    )
    ds = ds.map(
        lambda x, y: tf.py_function(
            self._generate_eval_prompt, [x, y], [tf.string, tf.string]
        )
    )
    ds = ds.map(
        lambda x, y: tf.py_function(
            self._generate_full_eval_prompt, [x, y], [tf.string, tf.string]
        )
    )
    return ds
