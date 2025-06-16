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

# We import JAX and some related packages.
# Finally, we import Gemma.
import jax
import jax.numpy as jnp
from sentencepiece import sentencepiece_processor as spm
# We will use tensorflow to handle the dataset
import tensorflow as tf


"""Custom tokenizer for Gemma."""
class GemmaTokenizer:
  """Custom wrapper around a SentencePieceProcessor for tensorflow."""

  def __init__(self,
               spm_processor: spm.SentencePieceProcessor):
    self._spm_processor = spm_processor

  @property
  def pad_id(self) -> int:
    """Fast access to the pad id."""
    return self._spm_processor.pad_id()

  def tokenize(self,
               example: str | bytes,
               prefix: str = '',
               suffix: str = '',
               add_eos: bool = True) -> jax.Array:
    """Tokenization function.

    Args:
      example: input string to tokenize.
      prefix:  prefix to add to the input string.
      suffix:  suffix to add to the input string.
      add_eos: if True, add an end of sentence token at the end of the output
               sequence.
    Returns:
      Tokens corresponding to the input string.
    """
    int_list = [self._spm_processor.bos_id()]
    int_list.extend(self._spm_processor.EncodeAsIds(prefix))
    int_list.extend(self._spm_processor.EncodeAsIds(example))
    int_list.extend(self._spm_processor.EncodeAsIds(suffix))
    if add_eos:
      int_list.append(self._spm_processor.eos_id())

    #return tf.convert_to_tensor(int_list, dtype=tf.int32)
    return jnp.array(int_list, dtype=jnp.int32)

  def tokenize_tf_op(self,
                     str_tensor: tf.Tensor,
                     prefix: str = '',
                     suffix: str = '',
                     add_eos: bool = True) -> tf.Tensor:
    """Tensforflow operator for the tokenize function."""
    encoded = tf.numpy_function(
        self.tokenize,
        [str_tensor, prefix, suffix, add_eos],
        tf.int32)
    encoded.set_shape([None])
    return encoded

  # def to_string(self, tokens: jax.Array) -> str:
  #  """Convert an array of tokens to a string."""
  #  return self._spm_processor.EncodeIds(tokens.tolist())
