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

"""Training loop for Gemma."""
import queue
import threading

from absl import logging
import chex
from flax import jax_utils
from gemma import transformer as transformer_lib
import jax
import jax.numpy as jnp
import numpy as np
import optax


class BackgroundGenerator(threading.Thread):
  """Background generator for training batches."""

  def __init__(self, next_batch_fn, num_batches):
    threading.Thread.__init__(self)
    self.queue = queue.Queue(1)
    self.next_batch_fn = next_batch_fn
    self.num_batches = num_batches
    self.daemon = True
    self.batch = None

  def run(self):
    for _ in range(self.num_batches):
      item = self.next_batch_fn()
      self.queue.put(item)
    self.queue.put(None)

  def next(self):
    next_item = self.queue.get()
    if next_item is None:
      raise StopIteration
    return next_item


def forward_and_loss_fn(
    params,
    *,
    model: transformer_lib.Transformer,
    input_tokens: jax.Array,  # Shape [B, L]
    input_mask: jax.Array,  # Shape [B, L]
    positions: jax.Array,  # Shape [B, L]
    attention_mask: jax.Array,  # [B, L, L]
    factors: jax.Array,  # Shape [B]
) -> jax.Array:
  """The forward pass and the loss function.

  Args:
    params: Model's input parameters.
    model: The Gemma transformer model to call.
    input_tokens: Input tokens sequence, shape [B, L].
    input_mask: Tokens to ignore when computing the loss, shape [B, L].
    positions: Relative position of each token, shape [B, L].
    attention_mask: Input attention mask, shape [B, L].

  Returns:
    The softmax cross-entropy loss for the next-token prediction task.
  """

  # The forward pass on the input data.
  # No attention cache is needed here.
  logits, _ = model.apply(
      params,
      input_tokens,
      positions,
      None,  # Attention cache is None.
      attention_mask)

  # Exclude the last step as it does not appear in the targets.
  logits = logits[0, :-1]

  # Similarly, the first token cannot be predicted.
  target_tokens = input_tokens[0, 1:]
  target_mask = input_mask[0, 1:]

  # Convert the target labels to one-hot encoded vectors.
  one_hot = jax.nn.one_hot(target_tokens, logits.shape[-1])

  # Don't update on unwanted tokens.
  one_hot = one_hot * target_mask.astype(one_hot.dtype)[..., None]

  # Define the normalization factor.
  norm_factor = 1 / (jnp.sum(target_mask) + 1e-8)

  # Return the negative log likelihood (NLL) loss.
  return (
      -jnp.sum(
          jax.nn.log_softmax(logits)
          * jnp.expand_dims(factors, axis=-1)
          * one_hot
      )
      * norm_factor
  )


def get_attention_mask_and_positions(
    example: jax.Array,
    pad_id: int,
) -> tuple[jax.Array, jax.Array]:
  """Builds the position and attention mask vectors from the given tokens."""
  pad_mask = example != pad_id
  current_token_position = transformer_lib.build_positions_from_mask(pad_mask)
  attention_mask = transformer_lib.make_causal_attn_mask(pad_mask)
  return current_token_position, attention_mask


def train_step(
    model: transformer_lib.Transformer,
    params,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    pad_id: int,
    example_tokens: jax.Array,
    example_masks: jax.Array,
    factors: jax.Array,
):
  """Train step.

  Args:
    model: The Gemma transformer model.
    params: The model's input parameters.
    optimizer: The Optax optimizer to use.
    opt_state: The input optimizer's state.
    pad_id: ID of the pad token.
    example_tokens: Input tokens sequence, shape [B, L].
    example_masks: Tokens to ignore when computing the loss, shape [B, L].

  Returns:
    The training loss, the updated parameters, and the updated optimizer state.
  """

  # Build the position and attention mask vectors.
  positions, attention_mask = get_attention_mask_and_positions(
      example_tokens, pad_id
  )

  # The forward and backward passes.
  train_loss, grads = jax.value_and_grad(forward_and_loss_fn)(
      params,
      model=model,
      input_tokens=example_tokens,
      input_mask=example_masks,
      positions=positions,
      attention_mask=attention_mask,
      factors=factors,
  )
  grads = jax.lax.pmean(grads, axis_name='batch')
  # Update the parameters.
  updates, opt_state = optimizer.update(grads, opt_state)
  params = optax.apply_updates(params, updates)

  return train_loss, params, opt_state


def validation_step(
    model: transformer_lib.Transformer,
    params,
    pad_id: int,
    example_tokens: jax.Array,
    example_masks: jax.Array,
):
  """Validation step.

  Args:
    model: The Gemma transformer model.
    params: The model's input parameters.
    pad_id: ID of the pad token.
    example_tokens: Input tokens sequence, shape [B, L].
    example_masks: Tokens to ignore when computing the loss, shape [B, L].

  Returns:
    The validation loss.
  """
  positions, attention_mask = get_attention_mask_and_positions(
      example_tokens, pad_id
  )
  val_loss = forward_and_loss_fn(
      params,
      model=model,
      input_tokens=example_tokens,
      input_mask=example_masks,
      positions=positions,
      attention_mask=attention_mask,
      factors=jnp.ones(example_tokens.shape[0]),
  )
  val_loss = jax.lax.pmean(val_loss, axis_name='batch')
  return val_loss


@chex.dataclass(frozen=True)
class TrainingConfig:
  learning_rate: float
  batch_size: int


class TrainingLoop:
  """Training loop for Gemma."""

  def __init__(
      self,
      model: transformer_lib.Transformer,
      pad_id: int,
      training_cfg: TrainingConfig,
      num_training_steps: int = 100,
      optimization_alg: str = 'sgd',
  ):
    self.model = model
    if optimization_alg == 'sgd':
      self.optimizer = optax.sgd(training_cfg.learning_rate)
    else:
      self.optimizer = optax.adam(training_cfg.learning_rate)
    self.pad_id = pad_id
    self.batch_size = training_cfg.batch_size
    self.learning_rate = training_cfg.learning_rate
    self.training_cfg = training_cfg
    self.num_training_steps = num_training_steps

  def train_loop(
      self,
      params,
      get_next_batch_fn,
  ):
    """Train loop."""

    opt_state = self.optimizer.init(params)
    pmapped_train_step = jax.pmap(
        lambda params, opt_state, example_tokens, example_masks, factors: train_step(
            self.model,
            params,
            self.optimizer,
            opt_state,
            self.pad_id,
            example_tokens,
            example_masks,
            factors
        ),
        axis_name='batch',
        donate_argnums=(0, 1)
    )

    n_steps = 0
    avg_loss = 0

    train_params = jax_utils.replicate(params)
    opt_state = jax_utils.replicate(opt_state)


    background_generator = BackgroundGenerator(
        get_next_batch_fn,
        self.num_training_steps,
    )
    background_generator.start()

    for _ in range(self.num_training_steps):
      try:
        factors, train_batch_input_tokens, train_batch_target_mask = (
            background_generator.next()
        )
      except:
        break
      train_batch_input_tokens = np.asarray(train_batch_input_tokens)
      train_batch_input_tokens = np.reshape(
          train_batch_input_tokens,
          (
              jax.process_count(),
              jax.local_device_count(),
              *train_batch_input_tokens.shape[1:],
          ),
      )
      train_batch_target_mask = np.asarray(train_batch_target_mask)
      train_batch_target_mask = np.reshape(
          train_batch_target_mask,
          (
              jax.process_count(),
              jax.local_device_count(),
              *train_batch_target_mask.shape[1:],
          ),
      )

      factors = np.asarray(factors)
      factors = np.reshape(
          factors,
          (
              jax.process_count(),
              jax.local_device_count(),
          ),
      )
      logging.info(f'factors shape: {factors.shape}')
      train_loss, train_params, opt_state = pmapped_train_step(
          train_params,
          opt_state,
          # tokenizer_pad_id,
          train_batch_input_tokens[jax.process_index()],
          train_batch_target_mask[jax.process_index()],
          factors[jax.process_index()],
      )

      n_steps += 1
      train_loss = jax.experimental.multihost_utils.process_allgather(
          train_loss
      )
      avg_loss += train_loss
      if n_steps % 20 == 0:
        logging.info(
            'STEP %d',
            n_steps,
        )
        avg_loss = 0
    jax_utils.unreplicate(opt_state)
    return jax_utils.unreplicate(train_params)
