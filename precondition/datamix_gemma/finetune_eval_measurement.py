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

import functools

from absl import logging
# Finally, we import Gemma.
import jax
import numpy as np
from precondition.datamix_gemma import finetune_utils
from precondition.datamix_gemma.evals.gsm8k_eval import GSM8KEval
from precondition.datamix_gemma.training_batch_generators import vanilla_training_batch_generator
from precondition.datamix_gemma.training_loop import TrainingConfig
from precondition.datamix_gemma.training_loop import TrainingLoop


SEQ_SIZE = 1000
VARIANT = '2b-it'  # @param ['2b', '2b-it'] {type:"string"}
CKPT_PATH = '/tfhub/prod/g_mini/2b_it_v1p1_orbax/1'
VOCAB_PATH = '/home/mriviere/g_mini/tokenizer/gemini_bpe_256k_v5_no_tags_cleared_v1.model'

def compute_eval_score(params, train_obj, training_batch_generator_obj, eval_obj):
  training_batch_generator_obj.prepare_for_training([1.,], [1.,])
  trained_params = train_obj.train_loop(
      params={'params': params},
      get_next_batch_fn=functools.partial(
          training_batch_generator_obj.get_next_batch, index=0
      ),
  )
  eval_score = eval_obj.evaluate(trained_params['params'])
  return eval_score

def finetune_eval_measurement():
  model_2b, tokenizer, vocab, params = finetune_utils.setup_model()

  params = jax.tree_util.tree_map(
      lambda arr: jax.device_put(
          arr, jax.local_devices(backend='cpu')[0]
      ),
      params,
  )
  train_batch_size = jax.local_device_count()
  logging.info('train_batch_size: %s', train_batch_size)
  mmlu_eval_batch_size = 2048


  training_cfg = TrainingConfig(
      learning_rate=1e-4,
      batch_size=train_batch_size,
  )

  train_loop_obj = TrainingLoop(
      model=model_2b,
      pad_id=tokenizer.pad_id,
      training_cfg=training_cfg,
      num_training_steps=1,
      #optimization_alg='adam',
  )

  all_dataset_builders = finetune_utils.get_dataset_builders(tokenizer)

  vanilla_training_batch_generator_obj = vanilla_training_batch_generator.VanillaTrainingBatchGenerator(
      train_ds_builders=[all_dataset_builders[0],],
      batch_size=jax.device_count(),
      num_weights=1,
      num_iterations=100,
  )
  gsm8k_eval_obj = GSM8KEval(
      model=model_2b,
      tokenizer=tokenizer,
      vocab=vocab,
      eval_batch_size=mmlu_eval_batch_size
      )
  vanilla_training_batch_generator_obj.prepare_for_training([[1.,]], [[1.,]])
  trained_params = {'params': params}
  for i in range(100):
    trained_params = train_loop_obj.train_loop(
        params=trained_params,
        get_next_batch_fn=functools.partial(
            vanilla_training_batch_generator_obj.get_next_batch, index=0
        ),
    )
    score = gsm8k_eval_obj.evaluate(trained_params['params'])
    logging.info(f'Score at index {i}: {score}')

  exit()

  gsm8k_eval_obj = GSM8KEval(
      model=model_2b,
      tokenizer=tokenizer,
      vocab=vocab,
      eval_batch_size=mmlu_eval_batch_size
      )

  rng = np.random.default_rng(seed=0)
  num_iterations = 100
  scores = []
  for i in range(len(all_dataset_builders)):
    vanilla_training_batch_generator_obj = vanilla_training_batch_generator.VanillaTrainingBatchGenerator(
        train_ds_builders=[all_dataset_builders[i],],
        batch_size=jax.device_count(),
    )
    score = compute_eval_score(
        params,
        train_obj=train_loop_obj,
        training_batch_generator_obj=vanilla_training_batch_generator_obj,
        eval_obj=gsm8k_eval_obj,
    )
    logging.info(f'Score at index {i}: {score}')
    scores.append(score)
