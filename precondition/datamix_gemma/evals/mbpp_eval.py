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

"""MBPP eval class."""

from concurrent.futures import _base
import functools
import pickle
import re
import signal

from absl import logging
import file
import jax
import jax.numpy as jnp
import numpy as np
from precondition.datamix_gemma import deconstructed_sampler
from precondition.datamix_gemma.dataset_builders import mbpp_dataset_builder
from precondition.datamix_gemma.evals import eval as eval_lib
import tensorflow as tf


def handler(signum, frame): # pylint: disable=unused-argument
  """Handler for timeout."""
  raise TimeoutError('Timed out!')

pickle_file_prefix = '/home/shivguptashi/mbpp_pickle'
CLEANCODE_REGEX = re.compile(
    r'.*\[BEGIN\](.*)\[DONE\].*', re.MULTILINE | re.DOTALL
)


class MBPPEval(eval_lib.Eval):
  """MBPP eval class."""

  def __init__(self, model, tokenizer, vocab, eval_batch_size):
    super().__init__(model, tokenizer, vocab, eval_batch_size)
    self.mbpp_dataset_builder = mbpp_dataset_builder.MBPPDatasetBuilder(
        self.tokenizer, 1000
    )
    self.sampler = deconstructed_sampler.DeconstructedSampler(
        transformer=model,
        vocab=vocab,
        params={'params': None},
        )

    self.eval_batch_size = eval_batch_size
    self.eval_ds = self.mbpp_dataset_builder.get_test_dataset()
    self.eval_ds = self.eval_ds.batch(
        eval_batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    self.eval_ds = self.eval_ds.prefetch(tf.data.AUTOTUNE)
    self.sampling_states, self.total_sampling_steps = self.unpickle_sampling_states() # pylint: disable=line-too-long

  def generate_and_pickle_sampling_states(self):
    all_sampling_states = []
    all_total_sampling_steps = []
    for batch in self.eval_ds:
      prompt_batch = []
      for val in batch[0]:
        prompt_batch.append(val.numpy().decode('utf-8'))
      while len(prompt_batch) < self.eval_batch_size:
        prompt_batch.append('')
      logging.info('mbpp eval batch: %s', batch)
      all_input_ids, forbidden_token_ids, total_sampling_steps= self.sampler.generate_tokenized_inputs( #pylint: disable=line-too-long
          prompt_batch, total_generation_steps=2)
      sampling_states = self.sampler.generate_initial_sampling_state(
          all_input_ids=all_input_ids,
          forbidden_token_ids=forbidden_token_ids,
          total_sampling_steps=total_sampling_steps,
      )
      sampling_states = jax.tree_util.tree_map(
          lambda arr: jax.device_put(arr, jax.local_devices(backend='cpu')[0]),
          sampling_states)
      all_sampling_states.append(sampling_states)
      all_total_sampling_steps.append(total_sampling_steps)
    cur_pickle_file = pickle_file_prefix + '_' + str(self.eval_batch_size) + '_all_sampling_states' + '.pkl' # pylint: disable=line-too-long
    with file.Open(cur_pickle_file, 'wb+') as f:
        pickle.dump(all_sampling_states, f)
    cur_pickle_file = pickle_file_prefix + '_' + str(self.eval_batch_size) + '_all_total_sampling_steps' + '.pkl'  # pylint: disable=line-too-long
    with file.Open(cur_pickle_file, 'wb+') as f:
        pickle.dump(all_total_sampling_steps, f)

  def unpickle_sampling_states(self):
    cur_pickle_file = pickle_file_prefix + '_' + str(self.eval_batch_size) + '_all_sampling_states' + '.pkl' # pylint: disable=line-too-long
    with file.Open(cur_pickle_file, 'rb') as f:
        all_sampling_states = pickle.load(f)
    cur_pickle_file = pickle_file_prefix + '_' + str(self.eval_batch_size) + '_all_total_sampling_steps' + '.pkl' # pylint: disable=line-too-long
    with file.Open(cur_pickle_file, 'rb') as f:
        all_total_sampling_steps = pickle.load(f)
    return all_sampling_states, all_total_sampling_steps

  def _generate_responses(self, initial_sampling_state):
    responses = self.sampler(
        #all_input_ids=initial_sampling_state[0], # nprompts x seq_len
        #forbidden_token_ids=initial_sampling_state[1], #
        #total_sampling_steps=initial_sampling_state[2],
        initial_sampling_state=initial_sampling_state,
        num_devices=jax.device_count(),
        echo=False,
        return_logits=False,
    )
    return responses

  def _evaluate_helper(self, data, tests_batch, device):
    token_buffer, num_input_tokens = data[0], data[1]
    sampling_steps = data[2]
    logging.info(f'running evaluate helper with device: {device}')
    decoded_responses = self.sampler.decode_outputs(token_buffer, num_input_tokens, sampling_steps)
    print(f'running mbpp eval, decoded responses: {decoded_responses}')
    correct_count = 0
    total_count = len(tests_batch)
    for i in range(len(tests_batch)):
      if self.is_correct(decoded_responses[i], tests_batch[i]):
        correct_count += 1
      total_count += 1
    return np.array([correct_count, total_count])

  def _evaluate_host_callback(self):
    correct_count = 0
    total_count = 0
    for sampling_state, sampling_steps, batch in zip(
        self.sampling_states, self.total_sampling_steps, self.eval_ds
    ):
      # total_sampling_steps = question_tokens_batch.shape[1] + 256
      tests_batch = []
      for val in batch[1]:
        tests_batch.append(val.numpy().decode('utf-8'))
      token_buffer, num_input_tokens = self._generate_responses(sampling_state)
      logging.info('Done generating responses!')
      token_buffer = jnp.reshape(
          token_buffer,
          (
              jax.local_device_count(),
              self.eval_batch_size // jax.local_device_count(),
              token_buffer.shape[1],
          ),
      )
      num_input_tokens = jnp.reshape(
          num_input_tokens,
          (
              jax.local_device_count(),
              self.eval_batch_size // jax.local_device_count(),
          ),
      )
      sampling_steps_arr = (
          jnp.ones((jax.local_device_count(), 1), dtype=jnp.int32)
          * sampling_steps
      )
      results = jax.pmap(
          lambda x, y, z: jax.experimental.host_callback.call(
              functools.partial(self._evaluate_helper, tests_batch=tests_batch),
              [x, y, z],
              result_shape=jax.ShapeDtypeStruct(
                  shape=(jax.local_device_count(), 2), dtype=np.float32
              ),
              call_with_device=True,
          )
      )(token_buffer, num_input_tokens, sampling_steps_arr)
      for i in range(len(results)):
        correct_count += results[i][0]
        total_count += results[i][1]
    return correct_count / total_count

  def _evaluate_vanilla(self):
    correct_count = 0
    total_count = 0
    for sampling_state, sampling_steps, batch in zip(
        self.sampling_states, self.total_sampling_steps, self.eval_ds
    ):
      # total_sampling_steps = question_tokens_batch.shape[1] + 256
      tests_batch = []
      for val in batch[1]:
        tests_batch.append(val.numpy().decode('utf-8'))
      token_buffer, num_input_tokens = self._generate_responses(sampling_state)
      logging.info('Done generating responses!')
      data = [token_buffer, num_input_tokens, sampling_steps]
      results = self._evaluate_helper(data, tests_batch, device=None)
      for _ in range(len(results)):
        correct_count += results[0]
        total_count += results[1]
    return correct_count / total_count

  def evaluate(self, params):
    self.sampler.update_params(params)
    #self._evaluate_host_callback()
    return self._evaluate_vanilla()

  def extract_answer(self, completion):
    pass

  def is_correct(self, model_completion, tests_str):
    program = model_completion.strip('[BEGIN]').strip('[DONE]')
    full_program = '\n'.join([program, tests_str])
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(100)
    try:
      exec(full_program)  # pylint: disable=exec-used
      signal.alarm(0)
      return True
    except:  # pylint: disable=bare-except
      print('Program failed!')
      signal.alarm(0)
      return False
    print('Correct program!')
    return True
