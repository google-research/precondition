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

"""Deterministic strategy bandit loop."""

import copy
import functools

from absl import logging
import jax
import numpy as np
from precondition.datamix_gemma import bandit_loop
from precondition.datamix_gemma import training_loop
from precondition.datamix_gemma.evals import eval as eval_lib
from precondition.datamix_gemma.training_batch_generators import dartboard_deterministic_training_batch_generator


_TEST_FN_FLAG = False
_STEP_SIZE = 0.1

def run_deterministic_strategy_bandit_loop(
    eval_obj: eval_lib.Eval,
    train_obj: training_loop.TrainingLoop,
    training_batch_generator_obj: dartboard_deterministic_training_batch_generator.DartboardDeterministicTrainingBatchGenerator,
    init_weights=None,
    num_iterations=1000,
    step_size=0.001,
    delta=0.001,
    warm_start=False,
    init_params=None,
    static_weights=False,
    step_size_decay=False,
    step_size_decay_rate=0.95,
    momentum=False,
    momentum_beta=0.1,
    use_adagrad=False,
    use_adagrad_avg=False,
    use_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.99,
    gradient_clipping=False,
    gradient_clipping_norm=30000,
):
  """Run the bandit loop.

  Args:
    eval_obj: the evaluation object.
    train_obj: the training object.
    init_weights: the initial weights.
    num_iterations: the number of iterations to run.
    step_size: the step size for the gradient update.
    delta: the magnitude of the perturbation for the gradient estimate.
    warm_start: whether to warm start the training.
    init_params: the initial parameters.
    static_weights: whether to use static weights.
    step_size_decay: whether to decay the step size.
    step_size_decay_rate: the rate of decay.
    momentum: whether to use momentum.
    momentum_beta: the beta for momentum.

  Returns:
    The final weights.
  """
  adagrad_matrix= None
  adam_matrix = None
  adam_first_moment = None
  assert not (use_adagrad and use_adam)
  if use_adam:
    adam_matrix = np.ones(len(training_batch_generator_obj.train_ds_builders)) * 1
    adam_first_moment = np.zeros(len(training_batch_generator_obj.train_ds_builders))
  elif use_adagrad:
    adagrad_matrix = np.ones(len(training_batch_generator_obj.train_ds_builders)) * 1e6
  init_weights = init_weights
  if init_weights is None:
    init_weights = np.ones(
        len(training_batch_generator_obj.train_ds_builders)
    ) / len(training_batch_generator_obj.train_ds_builders)
  momentum_vec = np.zeros(len(training_batch_generator_obj.train_ds_builders))

  next_params = init_params
  #print(f'init eval score: {eval_obj.evaluate(params=next_params)}')
  #logging.info('Done running init_eval')

  weights = init_weights
  rng = np.random.default_rng(seed=0)

  unnormalized_weights = copy.deepcopy(weights)
  for it in range(num_iterations):
    if static_weights:
      weights = init_weights
    logging.info('[WEIGHTS]: %s', weights)
    #next_cands = _generate_candidates_random_sign(weights, rng, delta=delta)
    logging.info('Going to train!')
    #Prepare for training.
    gradient_discount_factor = training_batch_generator_obj.prepare_for_training(
        weights, unnormalized_weights
    )

    if not warm_start:
      cur_params = copy.deepcopy(init_params)
    else:
      cur_params = copy.deepcopy(next_params)
    training_operations = []
    init_trained_params = train_obj.train_loop(
        params={'params': cur_params},
        get_next_batch_fn=functools.partial(
            training_batch_generator_obj.get_next_batch
        ),
    )
    init_trained_params = jax.tree_util.tree_map(
        lambda arr: jax.device_put(
            arr, jax.local_devices(backend='cpu')[0]
        ),
        init_trained_params,
    )
    for i in range(len(training_batch_generator_obj.train_ds_builders)):
      trained_params = copy.deepcopy(init_trained_params)
      trained_params = train_obj.train_loop(
          params=trained_params,
          get_next_batch_fn=functools.partial(
              training_batch_generator_obj.get_next_batch_special, index=i, delta=delta
          ),
      )
      trained_params = jax.tree_util.tree_map(
          lambda arr: jax.device_put(
              arr, jax.local_devices(backend='cpu')[0]
          ),
          trained_params,
      )
      training_operations.append(trained_params)
    logging.info('Done training!')
    init_score = eval_obj.evaluate(init_trained_params['params'])
    scores = []
    for trained_params in training_operations:
      trained_params = jax.device_get(trained_params)
      scores.append(
          eval_obj.evaluate(trained_params['params'])
      )
    if warm_start:
      next_params = training_operations[0]['params']
    logging.info('iteration: %d', it)
    logging.info('[SCORES]: %s', scores)
    for i in range(weights.shape[0]):
      logging.inf(f'weights_{str(i)}: {weights[i]}')
    logging.info(f'average_score: {(scores[0] + scores[1]) / 2.0}')
    logging.info(f'score_1: {scores[0]}')
    logging.info(f'score_2: {scores[1]}')
    grad = np.zeros(len(weights))
    for i in range(len(weights)):
      grad[i] = (scores[i] - init_score)/delta
    logging.info('[GRAD]: %s', grad)
    if momentum:
      momentum_vec = momentum_beta * momentum_vec + grad
      unnormalized_weights = bandit_loop._exponentiated_gradient(weights, momentum_vec, step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
    elif use_adagrad:
      adagrad_matrix += grad * grad
      truncated_adagrad_matrix = np.maximum(adagrad_matrix, 1e-8)
      unnormalized_weights= bandit_loop._exponentiated_gradient(weights, grad / np.sqrt(truncated_adagrad_matrix), step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
      for i in range(weights.shape[0]):
        logging.info(f'adagrad_matrix_{str(i)}: {adagrad_matrix[i]}')
    elif use_adam:
      adam_first_moment = adam_beta1 * adam_first_moment + (1 - adam_beta1) * grad
      bias_corrected_first_moment = adam_first_moment / (1 - adam_beta1 ** (it + 1))
      logging.info(f'bias_corrected_first_moment: {bias_corrected_first_moment}')
      adam_matrix = (1 - adam_beta2) * grad * grad + adam_beta2 * adam_matrix
      logging.info(f'adam_matrix: {adam_matrix}')
      bias_corrected_adam_matrix = adam_matrix / (1 - adam_beta2 ** (it + 1))
      truncated_bias_corrected_adam_matrix = np.maximum(bias_corrected_adam_matrix, 1e-8)
      unnormalized_weights = bandit_loop._exponentiated_gradient(weights, bias_corrected_first_moment/ np.sqrt(truncated_bias_corrected_adam_matrix), step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
      for i in range(weights.shape[0]):
        logging.info(f'adam_matrix_{str(i)}: {adam_matrix[i]}')
        logging.info(f'adam_first_moment_{str(i)}: adam_first_moment[i]')
    elif use_adagrad_avg:
      adagrad_matrix += np.square(grad)
      unnormalized_weights = bandit_loop._exponentiated_gradient(weights, grad / np.mean(np.sqrt(adagrad_matrix + 1e-8)), step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
      for i in range(weights.shape[0]):
        logging.info(f'adagrad_matrix_{str(i)}: {adagrad_matrix[i]}')
    else:
      unnormalized_weights = bandit_loop._exponentiated_gradient(weights, grad, step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
    if step_size_decay:
      step_size *= step_size_decay_rate


  return weights
