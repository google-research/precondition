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

"""GSM8k dataset builder."""

import enum as Enum

import jax.dlpack
from precondition.datamix_gemma.dataset_builders import dataset_builder
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
import tensorflow as tf
import tensorflow_datasets as tfds


PREAMBLE = """As an expert problem solver solve step by step the following mathematical questions."""

# The default gsm8k prompt from the CoT paper
# https://arxiv.org/pdf/2201.11903.pdf page 35.

PROMPT = """Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
A: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.

Q: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
A: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
A: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8."""


class DatasetSplit(Enum.Enum):
  TRAIN = 'train'
  TEST = 'test'


class GSM8KDatasetBuilder(dataset_builder.DatasetBuilder):
  """Dataset builder for the GSM8k dataset."""

  N_ITEMS = {DatasetSplit.TRAIN: 7473}

  #BUFFER_SIZE_SHUFFLE = 10_000
  BUFFER_SIZE_SHUFFLE = 100
  ANSWER_PREFIX = 'A: '
  ANSWER_SUFFIX = '\n'
  QUESTION_PREFIX = 'Q: '
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
            'huggingface:gsm8k/main', split='train'
        ),
        DatasetSplit.TEST: tfds.load(
            'huggingface:gsm8k/main', split='test'
        ),
    }
    self._max_seq_len = max_seq_len

  def _tokenize_question(self, example: tf.Tensor):
    """Tokenization function for the Question."""
    return self._tokenizer.tokenize_tf_op(
        example,
        prefix=self.QUESTION_PREFIX,
        suffix=self.QUESTION_SUFFIX,
        add_eos=False,
    )

  def _tokenize_answer(self, example: tf.Tensor):
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
            self._tokenize_answer(x['answer']),
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
    ds = self._base_data[DatasetSplit.TEST].map(
        lambda x: (
            self._tokenize_question(x['question']),
            self._tokenize_answer(x['answer']),
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.map(
        self._to_training_input,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.filter(lambda x: tf.shape(x.input_tokens)[0] <= self._max_seq_len)
    # ds = ds.batch(batch_size, drop_remainder=True)
    return ds
    # ds = [self._to_training_input(x, y) for x, y in ds]
    # print('here3:', ds)
    # ds = [x for x in ds if tf.shape(x.input_tokens)[0] <= self._max_seq_len]
    # ds = [ds[i : i + batch_size] for i in range(0, len(ds), batch_size)]

  def get_question_answer_dataset(self):
    #ds = self._base_data[DatasetSplit.TEST]
    return self._base_data[DatasetSplit.TEST]
