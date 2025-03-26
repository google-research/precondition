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

# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sampler for Gemma transformer.

An example of a sampling class for a Gemma model.
"""

from collections.abc import Sequence
import dataclasses
import logging

import chex
from gemma.deprecated import modules
from gemma.deprecated import params as params_lib
from gemma.deprecated import transformer as transformer_lib
import jax
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from sentencepiece import sentencepiece_processor as spm


def _compute_attention_masks(
    time_step: jax.Array, seq_len: int, input_mask: jax.Array
) -> jax.Array:
  """Computes causal attention mask."""
  bsz = input_mask.shape[0]
  batch_time_step = jnp.full((bsz, 1), time_step, dtype=jnp.uint32)
  causal_padding = jnp.greater(
      jnp.expand_dims(jnp.arange(seq_len), 0), batch_time_step
  )
  max_seq_len = min(input_mask.shape[-1], seq_len)
  input_mask = jax.lax.dynamic_slice(
      input_mask,
      (0, jnp.maximum(time_step - seq_len + 1, 0)),
      (bsz, max_seq_len),
  )
  input_mask = (
      jnp.zeros((bsz, seq_len), dtype=jnp.bool_)
      .at[:, :max_seq_len]
      .set(input_mask)
  )

  causal_padding = jnp.logical_or(causal_padding, input_mask)
  attention_mask = causal_padding[:, jnp.newaxis, :].astype(jnp.bool_)

  return ~attention_mask


@chex.dataclass
class _SamplingState:
  """Internal sampling state."""

  # Decoding step.
  decoding_step: jnp.int32

  # Number of tokens in the prompt.
  num_input_tokens: jnp.ndarray  # [B]

  # Fixed-size buffer for accumulating the output tokens.
  token_buffer: jnp.ndarray  # [B, L]

  # Position indices, based on ignoring pad tokens.
  positions: jnp.ndarray  # [B, L]

  # Model state for conditioning the model on autoregressively.
  cache: dict[str, modules.LayerCache]

  # Is decoding done on the given sequence?
  done: jnp.ndarray  # [B]

  # Total sampling steps (including the prompt).
  total_sampling_steps: int

  # Fixed-size buffer for accumulating the output logits.
  logits_buffer: jnp.ndarray | None = None  # [B, L, V]

  # List of tokens that are forbidden to be generated.
  forbidden_token_ids: Sequence[int] | None = None

def tree_map_helper(cur_obj, sharding):
  def callback_func(idx):
    try:
      return cur_obj[idx]
    except:
      return cur_obj
  if type(cur_obj) == int or type(cur_obj) == float:
    return cur_obj
  else:
    try:
      return jax.make_array_from_callback(
          cur_obj.shape, sharding, callback_func
      )
    except:
      return cur_obj

@dataclasses.dataclass
class SamplerOutput:

  # Decoded samples from the model.
  text: list[str]

  # Per-step logits used during sampling.
  logits: list[list[float]]

  # Tokens corresponding to the generated samples.
  tokens: list[list[int]]


class DeconstructedSampler:
  """Sampler for gemma transformer."""

  def __init__(
      self,
      transformer: transformer_lib.Transformer,
      vocab: spm.SentencePieceProcessor,
      params: params_lib.Params,
  ):
    """Initializes a sampler for a Gemma model.

    Args:
      transformer: an instance of the Gemma transformer.
      vocab: vocabulary of the given model.
      params: weights of the model.
    """
    self.transformer = transformer
    self.vocab = vocab
    self.params = params
    self._compiled_sample_fn = jax.jit(self._sample_fn)
    devices = jax.experimental.mesh_utils.create_device_mesh((jax.device_count(),))
    mesh = Mesh(devices, axis_names=('ax'))
    self.sharding = NamedSharding(mesh, P('ax'))
    self.compiled_init_cache_fn = jax.jit(self.init_cache, static_argnames='bsz', out_shardings=self.sharding)
    self.cache_ = None

  @property
  def dtype(self) -> jnp.dtype:
    return jax.tree_util.tree_leaves(self.params)[0].dtype

  def update_params(self, params: params_lib.Params):
    self.params = params

  def _sample_step(
      self, params, sampler_state: _SamplingState
  ) -> _SamplingState:
    """Performs a single sampling step."""
    batch_size = sampler_state.token_buffer.shape[0]
    logging.info(f'sample_state battch size: {batch_size}')
    decoding_step = jnp.asarray(sampler_state.decoding_step, dtype=jnp.int32)
    last_token = sampler_state.token_buffer[:, decoding_step]
    input_mask = sampler_state.token_buffer == self.vocab.pad_id()
    attention_mask = _compute_attention_masks(
        decoding_step, self.transformer.config.max_cache_length, input_mask
    )
    step_positions = jnp.expand_dims(
        sampler_state.positions[:, decoding_step], -1
    )
    last_token = last_token.reshape((batch_size, 1))

    logits, cache = self.transformer.apply(
        {'params': params},
        last_token,
        step_positions,
        sampler_state.cache,
        attention_mask,
    )
    if sampler_state.forbidden_token_ids:
      logits = logits.at[:, :, sampler_state.forbidden_token_ids].set(-jnp.inf)

    next_token_candidate = jnp.argmax(logits, axis=-1)  # [B, 1]
    next_token_candidate = next_token_candidate[:, 0]  # [B,]

    next_token_candidate = jnp.where(
        decoding_step < sampler_state.num_input_tokens - 1,
        sampler_state.token_buffer[:, decoding_step + 1],
        next_token_candidate,
    )

    token_buffer = sampler_state.token_buffer.at[:, decoding_step + 1].set(
        next_token_candidate
    )

    if sampler_state.logits_buffer is not None:
      next_logits = jnp.squeeze(logits, 1)
      logits_buffer = sampler_state.logits_buffer.at[:, decoding_step + 1].set(
          next_logits
      )
    else:
      logits_buffer = sampler_state.logits_buffer

    done = sampler_state.done | jnp.equal(
        token_buffer[:, decoding_step + 1], self.vocab.eos_id()
    )

    return _SamplingState(
        decoding_step=sampler_state.decoding_step + 1,
        num_input_tokens=sampler_state.num_input_tokens,
        token_buffer=token_buffer,
        positions=sampler_state.positions,
        logits_buffer=logits_buffer,
        cache=cache,
        done=done,
        total_sampling_steps=sampler_state.total_sampling_steps,
        forbidden_token_ids=sampler_state.forbidden_token_ids,
    )

  def init_cache(self, bsz: int) -> dict[str, modules.LayerCache]:
    """Initializes the attention cache for each layer."""
    return self.transformer.config.init_cache(bsz, dtype=self.dtype)

  def init_sample_state(
      self,
      all_input_ids: list[jax.Array],
      total_sampling_steps: int,
      include_logits: bool = False,
      forbidden_token_ids: Sequence[int] | None = None,
  ) -> _SamplingState:
    """Initializes the sampling state given input prompts."""
    bsz = len(all_input_ids)
    num_input_tokens = [len(input_ids) for input_ids in all_input_ids]
    buffer_size = total_sampling_steps + 1
    logging.info(f'buffer_size: {buffer_size}, bsz: {bsz}')

    token_buffer = jnp.full(
        (
            bsz,
            buffer_size,
        ),
        self.vocab.pad_id(),
        dtype=jnp.int32,
    )
    input_mask = jnp.ones_like(token_buffer, dtype=jnp.bool_)
    for i, (input_ids, num_tokens) in enumerate(
        zip(all_input_ids, num_input_tokens)
    ):
      logging.info(f'i: {i}, num_tokens: {num_tokens}, input_ids shape: {input_ids.shape}, len of all_input_ids: {len(all_input_ids)}')
      token_buffer = token_buffer.at[i, :num_tokens].set(input_ids)
      input_mask = input_mask.at[i, :num_tokens].set(
          input_ids != self.vocab.pad_id()
      )
    logging.info(f'Done setting token buffer and input mask')
    positions = transformer_lib.build_positions_from_mask(input_mask)
    logging.info(f'Done building positions from mask')

    done = jnp.zeros((bsz,), dtype=jnp.bool_)

    if include_logits:
      logits_buffer = jnp.zeros(
          (bsz, buffer_size, self.transformer.config.num_embed),
          dtype=jnp.float32,
      )
    else:
      logits_buffer = None
    logging.info(f'Done setting logits buffer')
    return _SamplingState(
        decoding_step=0,
        num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
        token_buffer=token_buffer,
        positions=positions,
        logits_buffer=logits_buffer,
        cache={},
        #cache=self.init_cache(bsz),
        done=done,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
    )

  def tokenize(self, input_string: str) -> jax.Array:
    """Tokenizes the input string."""
    input_ids = self.vocab.EncodeAsIds(input_string)
    #print('tokenized input_ids:', jnp.array(input_ids).tolist())
    input_ids = jnp.array(
        [self.vocab.bos_id()] + jnp.array(input_ids).tolist(), dtype=jnp.int32
    )
    return input_ids

  def mask_tokens_after_eos_ids(self, token_buffer):
    """Mask token IDs after the EOS token with the padding ID."""
    eos_id = self.vocab.eos_id()
    eos_exists = jnp.any(jnp.equal(token_buffer, eos_id), axis=-1)
    eos_indices = jnp.where(
        eos_exists,
        jnp.argmax(jnp.equal(token_buffer, eos_id), axis=-1),
        token_buffer.shape[-1],
    )
    mask = jnp.less_equal(
        jnp.arange(token_buffer.shape[-1]), eos_indices[:, None]
    )
    masked_token_buffer = token_buffer * mask + self.vocab.pad_id()*(1 - mask)

    return masked_token_buffer

  def _sample_fn(
      self,
      params: params_lib.Params,
      initial_sampling_state: _SamplingState,
  ) -> _SamplingState:
    """Internal sampling function (to be jitted)."""

    def sample_with_params(sampler_state: _SamplingState):
      return self._sample_step(params, sampler_state)

    def cond_fn(sampler_state: _SamplingState):
      return (
          sampler_state.decoding_step < sampler_state.total_sampling_steps
      ) & jnp.any(jnp.logical_not(sampler_state.done))

    return jax.lax.while_loop(
        cond_fn, sample_with_params, initial_sampling_state
    )

  def generate_tokenized_inputs(
      self,
      input_strings: Sequence[str],
      total_generation_steps: int,
      forbidden_tokens: Sequence[str] | None = None,
  ):
    forbidden_token_ids = None
    if forbidden_tokens is not None:
      forbidden_token_ids = []
      for token in forbidden_tokens:
        token_id = self.vocab.EncodeAsIds(token)
        if len(token_id) != 1:
          raise ValueError(
              'Forbidden tokens must map to single token ids in the vocab.'
          )
        forbidden_token_ids.extend(token_id)
      forbidden_token_ids = tuple(forbidden_token_ids)
    all_input_ids = [self.tokenize(x) for x in input_strings]
    max_input_length = max(len(input_ids) for input_ids in all_input_ids)
    total_sampling_steps = max_input_length + total_generation_steps
    #initial_sampling_state = self.init_sample_state(
    #    device=jax.devices('cpu')[0],
    #    all_input_ids=all_input_ids,
    #    include_logits=return_logits,
    #    total_sampling_steps=total_sampling_steps,
    #    forbidden_token_ids=forbidden_token_ids,
    #)
    return all_input_ids, forbidden_token_ids, total_sampling_steps

  def generate_initial_sampling_state(
      self,
      all_input_ids: list[jax.Array],
      forbidden_token_ids: list[int] | None,
      total_sampling_steps: int,
      return_logits: bool = False,
      ):
    initial_sampling_state = self.init_sample_state(
        all_input_ids=all_input_ids,
        include_logits=return_logits,
        total_sampling_steps=total_sampling_steps,
        forbidden_token_ids=forbidden_token_ids,
    )
    return initial_sampling_state

  def post_process_sampling_state(
      self,
      token_buffer,
      num_input_tokens,
      total_sampling_steps,
      return_logits: bool = False,
      echo: bool = False,
  ):
    print('token buffer shape:',token_buffer.shape)
    masked_token_buffer = self.mask_tokens_after_eos_ids(
        token_buffer
    )
    out_tokens = []
    out_logits = []
    for i, (token_buffer, num_tokens) in enumerate(
        zip(
            masked_token_buffer,
            num_input_tokens,
        )
    ):
      start_idx = 0 if echo else num_tokens
      # out_tokens.append(token_buffer[start_idx:total_sampling_steps].tolist())
      out_tokens.append(
          token_buffer.tolist()[start_idx : total_sampling_steps]
      )
    return out_tokens

  def decode_outputs(self, token_buffer, num_input_tokens, total_sampling_steps, echo=False):
    masked_token_buffer = self.mask_tokens_after_eos_ids(
        token_buffer
    )
    out_tokens = []
    out_logits = []
    for i, (token_buffer, num_tokens) in enumerate(
        zip(
            masked_token_buffer,
            num_input_tokens,
        )
    ):
      start_idx = 0 if echo else num_tokens
      #start_idx = 0
      #out_tokens.append(token_buffer[start_idx:total_sampling_steps].tolist())
      out_tokens.append(token_buffer[start_idx:total_sampling_steps].tolist())
    #print('decoding tokens:', out_tokens)
    decoded_outputs = [self.vocab.DecodeIds(tokens) for tokens in out_tokens]
    #logging.info('decoded outputs:', decoded_outputs)
    return decoded_outputs

  def __call__(
      self,
      # input_strings: Sequence[str],
      initial_sampling_state: _SamplingState,
      # all_input_ids: list[jax.Array],
      # forbidden_token_ids: list[int] | None,
      # total_sampling_steps: int,
      num_devices: int,
      echo: bool = False,
      return_logits: bool = False,
      forbidden_tokens: Sequence[str] | None = None,
  ):
    """Samples a completion of the input string.

    Args:
      input_strings: input prompts to feed to the model for sampling.
      total_generation_steps: number of generation steps. will correspond to the
        longest prompt in the batch.
      echo: whether to return the prompt as part of the output sample.
      return_logits: whether to return per-step logits used during generation.
      forbidden_tokens: list of tokens that are forbidden to be generated. Each
        token must map to a single token id in the vocab.

    Returns:
      sampler_output: A SamplerOutput object containing the generated samples.
    """
    #forbidden_token_ids = None
    #if forbidden_tokens is not None:
    #  forbidden_token_ids = []
    #  for token in forbidden_tokens:
    #    token_id = self.vocab.EncodeAsIds(token)
    #    if len(token_id) != 1:
    #      raise ValueError(
    #          'Forbidden tokens must map to single token ids in the vocab.'
    #      )
    #    forbidden_token_ids.extend(token_id)
    #  forbidden_token_ids = tuple(forbidden_token_ids)
    #all_input_ids = [self.tokenize(x) for x in input_strings]
    #max_input_length = max(len(input_ids) for input_ids in all_input_ids)
    #total_sampling_steps = max_input_length + total_generation_steps
    #initial_sampling_state = self.init_sample_state(
    #    all_input_ids,
    #    include_logits=return_logits,
    #    total_sampling_steps=total_sampling_steps,
    #    forbidden_token_ids=forbidden_token_ids,
    #)

    #initial_sampling_state = self.init_sample_state(
    #    all_input_ids=all_input_ids,
    #    include_logits=return_logits,
    #    total_sampling_steps=total_sampling_steps,
    #    forbidden_token_ids=forbidden_token_ids,
    #)
    if self.cache_ is None:
      #self.cache_ = self.compiled_init_cache_fn(initial_sampling_state.token_buffer.shape[0])
      print(f'initial_sampling_state.token_buffer.shape[0]: {initial_sampling_state.token_buffer.shape[0]}')
      self.cache_ = self.compiled_init_cache_fn(initial_sampling_state.token_buffer.shape[0])
    initial_sampling_state.cache = self.cache_
    if self.sharding is None:
      devices = jax.experimental.mesh_utils.create_device_mesh((num_devices,))
      mesh = Mesh(devices, axis_names=('ax'))
      self.sharding = NamedSharding(mesh, P('ax'))
    #sharding = PositionalSharding(devices)

    #sharded_arr = jax.device_put(a, sharding)
    #sharded_arr = jax.make_array_from_callback(a.shape, sharding, lambda idx: a[idx])
    print('Created sharded array')
    #exit()
    sharded_initial_sampling_state = jax.tree_util.tree_map(
        lambda arr: tree_map_helper(arr, sharding=self.sharding),
        initial_sampling_state)

    sampling_state = self._compiled_sample_fn(
        self.params, sharded_initial_sampling_state
    )
    sampling_state = jax.experimental.multihost_utils.process_allgather(sampling_state, tiled=True)
    logging.info('returning token buffer with shape:', sampling_state.token_buffer.shape)
    return sampling_state.token_buffer, sampling_state.num_input_tokens
