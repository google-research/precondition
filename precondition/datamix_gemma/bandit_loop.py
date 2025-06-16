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

"""Bandit loop."""

import copy
import functools

from absl import logging
import numpy as np
from precondition.datamix_gemma import training_loop
from precondition.datamix_gemma.evals import eval as eval_lib
from precondition.datamix_gemma.training_batch_generators import training_batch_generator


_TEST_FN_FLAG = False
_STEP_SIZE = 0.1

def _compute_gradient(cands, delta, scores):
  cands_diff = cands[0] - cands[1]
  cands_diff_norm = np.linalg.norm(cands_diff)
  scores_diff = scores[0] - scores[1]
  return cands_diff.shape[0] * cands_diff * scores_diff / (cands_diff_norm ** 2)

#def _compute_gradient_random_sign(x1_y1, x2_y2):
#  """Compute gradient estimator according to Eq.
#
#  (3) of
#
#    https://arxiv.org/pdf/1507.08752.
#
#    gradient = d/(2 * delta)(f(x + delta * u) - f(x - delta * u))u
#    where delta is the perturbation magnitude, u is a vector sampled uniformly
#    from the unit sphere, and d is the dimension of x.
#
#  Args:
#    x1_y1: the first mixture input and the objective value, assume the input is
#      x + delta * u for some delta and u.
#    x2_y2: the second mixture input and the objective value, assume the input is
#      x - delta * u for some delta and u.
#
#  Returns:
#    The estimated gradient at x.
#  """
#
#  x1, y1 = x1_y1
#  x2, y2 = x2_y2
#  y_diff = y1 - y2
#  x_diff = x1 - x2
#
#  if np.absolute(x_diff[0]) > 1e-8:
#    delta = np.absolute(x_diff[0]) / 2.0 * np.sqrt(x_diff.shape[0])
#  else:
#    delta = np.absolute(x_diff[1]) / 2.0 * np.sqrt(x_diff.shape[0])
#
#  logging.info('[DELTA]: %s', delta)
#  logging.info('[X_DIFF]: %s', x_diff)
#  u = x_diff / (2 * delta)
#  return x1.shape[0] / (2 * delta) * y_diff * u
#

def _generate_candidates_random_sign(
    weights: np.ndarray, rng: np.random.Generator, delta: float = 0.1
):
  """Generate candidates for two-point evaluation.

  Args:
    weights: input at which to estimate the gradient, should be a distribution.
    rng: rng for generating the random perturbation.
    delta: magnitude of perturbation.

  Returns:
    Candidate mixtures to evaluate on.
  """
  # Sample random noise.
  u = np.zeros(weights.shape)
  half_coordinates = rng.choice(
      weights.shape[0], size=weights.shape[0] // 2, replace=False
  )
  other_coordinates = [
      i for i in range(weights.shape[0]) if i not in half_coordinates
  ]
  u[half_coordinates] = 1.0
  u[other_coordinates] = -1.0
  if weights.shape[0] % 2 != 0:
    zero_coordinate = other_coordinates[0]
    u[zero_coordinate] = 0.0

  u = u / np.sqrt(u.shape[0])

  # Ensure weights is in the capped simplex: each coordinate is between delta
  # and 1 - delta, and the coordinates sum to 1.
  weights *= (1 - delta * weights.shape[0])
  weights += delta * np.ones(weights.shape)

  weights_a = weights.copy()
  weights_a += delta * u
  weights_b = weights.copy()
  weights_b -= delta * u

  # Normalize to be sure they are distributions.
  weights_a /= np.linalg.norm(weights_a, ord=1)
  weights_b /= np.linalg.norm(weights_b, ord=1)
  return [weights_a, weights_b]

def _generate_gaussian_candidates(
    weights: np.ndarray, rng: np.random.Generator, delta: float
):
  """Generate candidates for two-point evaluation."""
  weights_cpy = weights.copy()
  weights_cpy *= 1 - delta * weights_cpy.shape[0]
  weights_cpy += delta * np.ones(weights_cpy.shape)
  u = rng.normal(size=weights.shape)
  u = u/np.linalg.norm(u)
  weights_a = weights_cpy + delta * u
  weights_b = weights_cpy - delta * u
  weights_a /= np.linalg.norm(weights_a, ord=1)
  weights_b /= np.linalg.norm(weights_b, ord=1)
  return [weights_a, weights_b]

def _generate_dirichlet_candidates(
    weights: np.ndarray, rng: np.random.Generator, delta: float
):
  """Generate candidates for two-point evaluation."""
  weights_a = weights.copy()
  weights_b = weights.copy()
  u = rng.dirichlet(alpha=np.ones(weights.shape[0]))
  weights_a += delta * u
  weights_b -= delta * u
  weights_a = np.maximum(weights_a, 0.0)
  # weights_b = np.


def _update(x_current, cands, delta, scores, step_size=_STEP_SIZE):
  return x_current - step_size * _compute_gradient(
      cands, delta, scores
  )


def _exponentiated_gradient(x_current, grad, step_size=_STEP_SIZE):
  """The exponentiated gradient update.

  x_{t+1} = x_t * exp(step_size * gradient) / Z
  where Z is the normalizing constant so x_{t+1} stays a distribution.

  Args:
    x_current: current mixture.
    grad: gradient estimate.
    step_size: update step size.

  Returns:
    The updated mixture.
  """
  grad_cpy = copy.deepcopy(grad)
  grad_cpy -= np.max(grad_cpy)
  unnormalized_x = x_current * np.exp(step_size * grad_cpy)
  return unnormalized_x


def run_bandit_loop(
    eval_obj: eval_lib.Eval,
    train_obj: training_loop.TrainingLoop,
    training_batch_generator_obj: training_batch_generator.TrainingBatchGenerator,
    init_weights=None,
    num_iterations=10000,
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
    adagrad_beta=1.0,
    use_adagrad_avg=False,
    use_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.99,
    gradient_clipping=False,
    gradient_clipping_norm=30000,
    candidate_generator_fn=_generate_gaussian_candidates,
    num_grad_evals=2,
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
  adagrad_matrix = None
  adam_matrix = None
  adam_first_moment = None
  assert not (use_adagrad and use_adam)
  if use_adam:
    adam_matrix = (
        np.ones(len(training_batch_generator_obj.train_ds_builders)) * 1
    )
    adam_first_moment = np.zeros(
        len(training_batch_generator_obj.train_ds_builders)
    )
  elif use_adagrad:
    adagrad_matrix = np.ones(
        len(training_batch_generator_obj.train_ds_builders)
    )
  init_weights = init_weights
  if init_weights is None:
    init_weights = np.ones(
        len(training_batch_generator_obj.train_ds_builders)
    ) / len(training_batch_generator_obj.train_ds_builders)
  momentum_vec = np.zeros(len(training_batch_generator_obj.train_ds_builders))

  next_params = init_params
  #logging.info('Starting init eval')
  #next_params = jax.device_get(next_params)
  #score = eval_obj.evaluate(params=next_params)
  #logging.info('Done running init_eval')
  #exit()
  #print(f'init eval score: {score}')

  weights = init_weights
  rng = np.random.default_rng(seed=0)

  unnormalized_weights = copy.deepcopy(weights)
  for it in range(num_iterations):
    if static_weights:
      weights = init_weights
    logging.info('[WEIGHTS]: %s', weights)
    # next_cands = _generate_candidates_random_sign(state.weights, rng,
    # delta=delta)
    next_cands = []
    for _ in range(num_grad_evals):
      cur_next_cands = candidate_generator_fn(weights, rng, delta=delta)
      next_cands.append(cur_next_cands[0])
      next_cands.append(cur_next_cands[1])
    logging.info('Going to train!')
    # Prepare for training.
    gradient_discount_factor = (
        training_batch_generator_obj.prepare_for_training(
            next_cands, unnormalized_weights
        )
    )

    if not _TEST_FN_FLAG:
      scores = []
      trained_params = None
      for cand_it in range(len(next_cands)):
        if not warm_start:
          cur_params = copy.deepcopy(init_params)
        else:
          cur_params = copy.deepcopy(next_params)
        trained_params = train_obj.train_loop(
            params={'params': cur_params},
            get_next_batch_fn=functools.partial(
                training_batch_generator_obj.get_next_batch, index=cand_it
            ),
        )
        scores.append(
            eval_obj.evaluate(trained_params['params'])
        )
        #trained_params = jax.tree_util.tree_map(
        #    lambda arr: jax.device_put(
        #        arr, jax.local_devices(backend='cpu')[0]
        #    ),
        #    trained_params,
        #)
        #training_operations.append(trained_params)
      #logging.info('Done training!')
      #for trained_params in training_operations:
      #  trained_params = jax.device_get(trained_params)
      #  scores.append(
      #      eval_obj.evaluate(trained_params['params'])
      #  )
      if warm_start:
        next_params = trained_params['params']
    else:
      scores = [-((x[0] - 0.8) ** 2 + (x[1] - 0.2) ** 2) for x in next_cands]
    logging.info('iteration: %d', it)
    logging.info('[SCORES]: %s', scores)
    for i in range(weights.shape[0]):
      logging.info(f'weights_{str(i)}: {weights[i]}')
    logging.info(f'average_score: {(scores[0] + scores[1]) / 2.0}')
    logging.info(f'score_1: {scores[0]}')
    logging.info(f'score_2: {scores[1]}')
    num_grad_evals_per = num_grad_evals // 2
    grad1 = np.zeros(weights.shape)
    grad2 = np.zeros(weights.shape)
    for i in range(num_grad_evals_per):
      grad1 += _compute_gradient(next_cands[(2 * i):(2 * i + 2)], delta, scores[(2 * i):(2 * i + 2)]) * gradient_discount_factor
      grad2 += _compute_gradient(next_cands[(2 * (i+num_grad_evals_per)):(2 * (i+num_grad_evals_per)+2)], delta, scores[(2 * (i+num_grad_evals_per)):(2 * (i+num_grad_evals_per)+2)]) * gradient_discount_factor
    grad1 = grad1 / num_grad_evals_per
    grad2 = grad2 / num_grad_evals_per
    if gradient_clipping and np.linalg.norm(grad1) > gradient_clipping_norm:
      grad1 = grad1 / np.linalg.norm(grad1)
      grad1 *= gradient_clipping_norm
    if gradient_clipping and np.linalg.norm(grad2) > gradient_clipping_norm:
      grad2 = grad2 / np.linalg.norm(grad2)
      grad2 *= gradient_clipping_norm
    grad = (grad1 + grad2)/2
    logging.info('[GRAD]: %s', grad)
    if momentum:
      momentum_vec = momentum_beta * momentum_vec + grad
      unnormalized_weights = _exponentiated_gradient(weights, momentum_vec, step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
    elif use_adagrad:
      adagrad_matrix += grad1 * grad2 * adagrad_beta
      truncated_adagrad_matrix = np.maximum(adagrad_matrix, 1e-3)
      unnormalized_weights= _exponentiated_gradient(weights, grad / np.sqrt(truncated_adagrad_matrix), step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
      for i in range(weights.shape[0]):
        logging.info('adagrad_matrix_{str(i)}: {adagrad_matrix[i]}')
    elif use_adam:
      adam_first_moment = adam_beta1 * adam_first_moment + (1 - adam_beta1) * grad
      bias_corrected_first_moment = adam_first_moment / (1 - adam_beta1 ** (it + 1))
      logging.info(f'bias_corrected_first_moment: {bias_corrected_first_moment}')
      adam_matrix = (1 - adam_beta2) * grad1 * grad2 + adam_beta2 * adam_matrix
      logging.info(f'adam_matrix: {adam_matrix}')
      bias_corrected_adam_matrix = adam_matrix / (1 - adam_beta2 ** (it + 1))
      truncated_bias_corrected_adam_matrix = np.maximum(bias_corrected_adam_matrix, 1e-8)
      unnormalized_weights = _exponentiated_gradient(weights, bias_corrected_first_moment/ np.sqrt(truncated_bias_corrected_adam_matrix), step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
      for i in range(weights.shape[0]):
        logging.info(f'adam_matrix_{i}: {adam_matrix[i]}')
        logging.info(f'adam_first_moment_{str(i)}: {adam_first_moment[i]}')
    elif use_adagrad_avg:
      adagrad_matrix += np.square(grad)
      unnormalized_weights = _exponentiated_gradient(weights, grad / np.mean(np.sqrt(adagrad_matrix + 1e-8)), step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
      for i in range(weights.shape[0]):
        logging.info(f'adagrad_matrix_{str(i)}: adagrad_matrix[i]')
    else:
      unnormalized_weights = _exponentiated_gradient(weights, grad, step_size)
      weights = unnormalized_weights/np.linalg.norm(unnormalized_weights, ord=1)
    if step_size_decay:
      step_size *= step_size_decay_rate



  return weights
