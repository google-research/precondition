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



SEQ_SIZE = 1000
VARIANT = '2b-it'  # @param ['2b', '2b-it'] {type:"string"}
CKPT_PATH = '/tfhub/prod/g_mini/2b_it_v1p1_orbax/1'
VOCAB_PATH = '/home/mriviere/g_mini/tokenizer/gemini_bpe_256k_v5_no_tags_cleared_v1.model'

#def estimate_gradient(weights, delta, rng, params, train_obj, training_batch_generator_obj, eval_obj, writer):
#  cands = bandit_loop._generate_gaussian_candidates(weights, rng, delta=delta)
#  gradient_discount_factor = training_batch_generator_obj.prepare_for_training(
#      cands[0], cands[1]
#  )
#  training_operations = []
#  for cand_it in range(len(cands)):
#    cur_params = copy.deepcopy(params)
#    trained_params = train_obj.train_loop(
#        params={'params': cur_params},
#        get_next_batch_fn=functools.partial(
#            training_batch_generator_obj.get_next_batch, index=cand_it
#        ),
#    )
#    trained_params = jax.tree_util.tree_map(
#        lambda arr: jax.device_put(
#            arr, jax.local_devices(backend='cpu')[0]
#        ),
#        trained_params,
#    )
#    training_operations.append(trained_params)
#  logging.info('Done training!')
#  scores = []
#  for trained_params in training_operations:
#    trained_params = jax.device_get(trained_params)
#    scores.append(
#        eval_obj.evaluate(trained_params['params'])
#    )
#  logging.info('[SCORES]: %s', scores)
#  for i in range(weights.shape[0]):
#    writer.write({'weights_' + str(i): weights[i]})
#  writer.write({'average_score': (scores[0] + scores[1]) / 2.0})
#  writer.write({'score_1': scores[0]})
#  writer.write({'score_2': scores[1]})
#  #grad = bandit_loop._compute_gradient_random_sign(*zip(cands, scores)) * gradient_discount_factor
#  grad = bandit_loop._compute_gradient(cands, delta, scores)
#  logging.info('[GRAD]: %s', grad)
#  return grad
#
#def snr_calculation():
#  model_2b, tokenizer, vocab, params = finetune_utils.setup_model()
#
#  params = jax.tree_util.tree_map(
#      lambda arr: jax.device_put(
#          arr, jax.local_devices(backend='cpu')[0]
#      ),
#      params,
#  )
#
#  train_batch_size = jax.local_device_count()
#  logging.info('train_batch_size: %s', train_batch_size)
#  mmlu_eval_batch_size = 8192
#  mbpp_eval_batch_size = 1024
#
#
#  training_cfg = TrainingConfig(
#      learning_rate=1e-4,
#      batch_size=train_batch_size,
#  )
#
#  train_loop_obj = TrainingLoop(
#      model=model_2b,
#      pad_id=tokenizer.pad_id,
#      training_cfg=training_cfg,
#      num_training_steps=100,
#  )
#
#  all_dataset_builders = finetune_utils.get_dataset_builders(tokenizer, [0, 2])
#  importance_weighting_training_batch_generator_obj = importance_weighting_training_batch_generator.ImportanceWeightingTrainingBatchGenerator(
#      train_ds_builders=all_dataset_builders,
#      batch_size=jax.device_count(),
#  )
#
#  gsm8k_eval_obj = GSM8KEval(
#      model=model_2b,
#      tokenizer=tokenizer,
#      vocab=vocab,
#      eval_batch_size=mmlu_eval_batch_size
#      )
#
#  rng = np.random.default_rng(seed=0)
#  data_id = xdata.get_auto_data_id()
#  writer = xdata.bt.writer(data_id, 'scores')
#  num_iterations = 100
#  running_outer_sum = np.zeros((2, 2))
#  state.running_sum = np.zeros(2)
#  state.running_outer_sum = np.zeros((2,2))
#  for i in range(num_iterations):
#    ckpt.restore_or_save()
#    grad_estimate = estimate_gradient(
#        weights=np.array([0.5, 0.5]),
#        delta=0.0000001,
#        rng=rng,
#        params=params,
#        train_obj=train_loop_obj,
#        training_batch_generator_obj=importance_weighting_training_batch_generator_obj,
#        eval_obj=gsm8k_eval_obj,
#        writer=writer,
#    )
#    grad_estimate = np.array(grad_estimate)
#    state.running_sum += grad_estimate
#    running_avg = state.running_sum / (i + 1)
#    state.running_outer_sum += np.outer(grad_estimate, grad_estimate)
#    running_cov_avg = (state.running_outer_sum/(i+1) - np.outer(running_avg, running_avg))
#    #writer.write({'running_avg': running_avg, 'running_cov_avg': running_cov_avg, 'trace of running_cov_avg': np.trace(running_cov_avg)})
#    if i > 10:
#      writer.write({'running_avg_0': running_avg[0]})
#      writer.write({'running_avg_1': running_avg[1]})
#      writer.write({'running_cov_avg_00': running_cov_avg[0, 0]})
#      writer.write({'running_cov_avg_11': running_cov_avg[1, 1]})
#      writer.write({'running grads SNR': (np.linalg.norm(running_avg) ** 2) / np.trace(running_cov_avg)})
#    ckpt.save()
#
#  writer.close()
#