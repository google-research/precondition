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

from copy import deepcopy

from absl import flags
from absl import logging
from gemma import params as params_lib
from gemma import transformer as transformer_lib
# Finally, we import Gemma.
import jax
import numpy as np
from precondition.datamix_gemma import deterministic_strategy_bandit_loop
from precondition.datamix_gemma.dataset_builders import preprocessed_codealpaca_dataset_builder
from precondition.datamix_gemma.dataset_builders import preprocessed_dolly_dataset_builder
from precondition.datamix_gemma.dataset_builders import preprocessed_gsm8k_dataset_builder
from precondition.datamix_gemma.dataset_builders import preprocessed_metamath_dataset_builder
from precondition.datamix_gemma.dataset_builders import preprocessed_open_orca_dataset_builder
from precondition.datamix_gemma.dataset_builders import preprocessed_orca_math_dataset_builder
from precondition.datamix_gemma.dataset_builders import preprocessed_sciq_dataset_builder
from precondition.datamix_gemma.dataset_builders import preprocessed_wikipedia_dataset_builder
from precondition.datamix_gemma.evals.gsm8k_eval import GSM8KEval
from precondition.datamix_gemma.tokenizers import gemma_tokenizer
from precondition.datamix_gemma.training_batch_generators import dartboard_deterministic_training_batch_generator
from precondition.datamix_gemma.training_loop import TrainingConfig
from precondition.datamix_gemma.training_loop import TrainingLoop
from sentencepiece import sentencepiece_processor as spm


SEQ_SIZE = 1000
VARIANT = '2b-it'  # @param ['2b', '2b-it'] {type:"string"}
#VARIANT='2b'
CKPT_PATH = '/tfhub/prod/g_mini/2b_it_v1p1_orbax/1'
#CKPT_PATH = '/tfhub/prod/g_mini/2b_final_pt_orbax/1'
VOCAB_PATH = '/home/mriviere/g_mini/tokenizer/gemini_bpe_256k_v5_no_tags_cleared_v1.model'



def eval_after_finetuning_on_single_dataset(
    train_obj,
    eval_obj,
    init_params,
):
  cur_params = deepcopy(init_params)
  cur_params = jax.device_get(cur_params)
  trained_params = train_obj.train_loop(params={'params': cur_params}, weights=np.array([1.]))
  return eval_obj.evaluate(trained_params['params'])

def setup_model():
  """Setup model."""
  vocab = spm.SentencePieceProcessor()
  vocab.Load(VOCAB_PATH)
  tokenizer = gemma_tokenizer.GemmaTokenizer(vocab)
  if flags.FLAGS.debug:
    config_2b = transformer_lib.TransformerConfig.gemma_2b(1024)
    model_2b = transformer_lib.Transformer(config=config_2b)
    params = model_2b.init(
        jax.random.PRNGKey(0),
        last_tokens=np.ones((1, 1), dtype=np.int32),
        positions=np.ones((1, 1), dtype=np.int32),
        cache=None,
        attention_mask=np.ones((1, 1, 1), dtype=np.bool_),
    )['params']
  else:
    params = params_lib.load_and_format_params(CKPT_PATH)
    config_2b = transformer_lib.TransformerConfig.from_params(params, cache_size=1024)
    model_2b = transformer_lib.Transformer(config=config_2b)
    params = params['transformer']
  return model_2b, tokenizer, vocab, params

def get_dataset_builders(tokenizer, dataset_indices = range(71)):
  preprocessed_gsm8k_dataset_builder_ = preprocessed_gsm8k_dataset_builder.PreprocessedGSM8KDatasetBuilder(
      tokenizer, SEQ_SIZE
  )

  preprocessed_open_orca_dataset_builder_ = preprocessed_open_orca_dataset_builder.PreprocessedOpenOrcaDatasetBuilder(
      tokenizer, SEQ_SIZE
  )
  preprocessed_orca_math_dataset_builder_ = preprocessed_orca_math_dataset_builder.PreprocessedOrcaMathDatasetBuilder(
      tokenizer, SEQ_SIZE
  )


  preprocessed_sciq_dataset_builder_ = preprocessed_sciq_dataset_builder.PreprocessedSciQDatasetBuilder(
      tokenizer, SEQ_SIZE
  )

  preprocessed_codealpaca_dataset_builder_ = preprocessed_codealpaca_dataset_builder.PreprocessedCodeAlpacaDatasetBuilder(
      tokenizer, SEQ_SIZE
  )

  preprocessed_metamath_dataset_builder_ = preprocessed_metamath_dataset_builder.PreprocessedMetaMathDatasetBuilder(
      tokenizer, SEQ_SIZE
  )

  preprocessed_dolly_dataset_builder_ = preprocessed_dolly_dataset_builder.PreprocessedDollyDatasetBuilder(
      tokenizer, SEQ_SIZE
  )

  wiki_dataset_builders = [preprocessed_wikipedia_dataset_builder.PreprocessedWikipediaDatasetBuilder(tokenizer, SEQ_SIZE, i) for i in range(64)]
  dataset_builders = [
      preprocessed_gsm8k_dataset_builder_,
      preprocessed_orca_math_dataset_builder_,
      preprocessed_open_orca_dataset_builder_,
      preprocessed_codealpaca_dataset_builder_,
      preprocessed_metamath_dataset_builder_,
      preprocessed_dolly_dataset_builder_,
      preprocessed_sciq_dataset_builder_,
      #gsm8k_dataset_builder_,
      #wiki_test_dataset_builder,
  ]
  dataset_builders.extend(wiki_dataset_builders)
  return [dataset_builders[i] for i in dataset_indices]

def finetune():
  """Finetune Gemma."""

  num_training_steps = 40
  model_2b, tokenizer, vocab, params = setup_model()

  train_batch_size = jax.local_device_count()
  logging.info('train_batch_size: %s', train_batch_size)
  mmlu_eval_batch_size = 8192
  mbpp_eval_batch_size = 1024


  training_cfg = TrainingConfig(
      learning_rate=1e-4,
      batch_size=train_batch_size,
  )

  train_loop_obj = TrainingLoop(
      model=model_2b,
      pad_id=tokenizer.pad_id,
      training_cfg=training_cfg,
      num_training_steps=num_training_steps,
  )

  #vanilla_training_batch_generator_obj = vanilla_training_batch_generator.VanillaTrainingBatchGenerator(
  #    train_ds_builders=all_dataset_builders,
  #    batch_size=jax.device_count(),
  #)
  #reweight_training_batch_generator_obj = reweight_training_batch_generator.ReweightTrainingBatchGenerator(
  #    train_ds_builders=all_dataset_builders,
  #    batch_size=jax.device_count(),
  #)
  #all_dataset_builders = get_dataset_builders(tokenizer, range(7))
  all_dataset_builders = get_dataset_builders(tokenizer, range(40))
  #all_dataset_builders = get_dataset_builders(tokenizer, range(71))
  #importance_weighting_training_batch_generator_obj = importance_weighting_training_batch_generator.ImportanceWeightingTrainingBatchGenerator(
  #    train_ds_builders=all_dataset_builders,
  #    batch_size=jax.device_count(),
  #    num_weights=4,
  #)
  #fixed_dataset_importance_weighting_training_batch_generator_obj = fixed_dataset_importance_weighting_training_batch_generator.FixedDatasetImportanceWeightingTrainingBatchGenerator(
  #    train_ds_builders=all_dataset_builders,
  #    batch_size=jax.device_count(),
  #    num_weights=4,
  #)

  #dartboard_importance_weighting_training_batch_generator_obj = dartboard_importance_weighting_training_batch_generator.DartboardImportanceWeightingTrainingBatchGenerator(
  #    train_ds_builders=all_dataset_builders,
  #    batch_size=jax.device_count(),
  #    num_weights=16,
  #)

  dartboard_deterministic_training_batch_generator_obj = dartboard_deterministic_training_batch_generator.DartboardDeterministicTrainingBatchGenerator(
      train_ds_builders=all_dataset_builders,
      batch_size=jax.device_count(),
      num_weights=4,
  )

  # option_tokens = generate_letter_token_dict(vocab)
  # subjects, prompts, labels = generate_all_prompts()
  # partialed_eval_fn = partial(
  #    eval_fn, option_tokens=option_tokens
  # )
  # partialed_eval_fn(params['transformer'], subjects, prompts, labels)
  #mmlu_eval_obj = MMLUEval(model_2b, tokenizer, vocab, mmlu_eval_batch_size)
  gsm8k_eval_obj = GSM8KEval(
      model=model_2b,
      tokenizer=tokenizer,
      vocab=vocab,
      eval_batch_size=2048
      )
  #mbpp_eval_obj = MBPPEval(
  #    model=model_2b,
  #    tokenizer=tokenizer,
  #    vocab=vocab,
  #    eval_batch_size=mbpp_eval_batch_size
  #)

  params = jax.tree_util.tree_map(
      lambda arr: jax.device_put(
          arr, jax.local_devices(backend='cpu')[0]
      ),
      params,
  )
  ###all_dataset_builders= [preprocessed_gsm8k_dataset_builder_]
  ##for i in range(len(all_dataset_builders)):
  #scores = np.zeros(100)
  #for i in range(100):
  #  cur_validation_ds = [all_dataset_builders[i].get_validation_dataset(batch_size=train_batch_size)]
  #  cur_train_loop_obj = TrainingLoop(
  #      model=model_2b,
  #      train_ds_builders=[all_dataset_builders[i]],
  #      validation_ds=cur_validation_ds,
  #      pad_id=tokenizer.pad_id,
  #      batch_size=train_batch_size,
  #      eval_every_n=training_cfg.eval_every_n,
  #      learning_rate=training_cfg.learning_rate,
  #      training_cfg=training_cfg,
  #      num_training_steps=0,
  #      num_validation_steps=1,
  #      #optimization_alg='adam',
  #  )
  #  eval_score = eval_after_finetuning_on_single_dataset(
  #      cur_train_loop_obj, gsm8k_eval_obj, params
  #  )
  #  #logging.info(f'dataset: {i}, eval: gsm8k score: {eval_after_finetuning_on_single_dataset(cur_train_loop_obj, gsm8k_eval_obj, params)}')
  #  logging.info(f'trial: {i}, eval: gsm8k score: {eval_score}')
  #  scores[i] = eval_score
  #  #logging.info(f'dataset: {i}, eval: mmlu score: {eval_after_finetuning_on_single_dataset(cur_train_loop_obj, mmlu_eval_obj, params)}')
  #logging.info(f'Scores mean: {np.mean(scores)}')
  #logging.info(f'Scores std: {np.std(scores)}')
  #exit()

  #exit()
  #print('params:', params['params'])
  #gsm8k_eval_obj = GSM8KEval(
  #    model=model_2b,
  #    tokenizer=tokenizer,
  #    vocab=vocab,
  #    eval_batch_size=eval_batch_size
  #    )

  #logging.info('Starting random baseline')
  #random_baseline(gsm8k_eval_obj, train_loop_obj, params, num_iterations=100)


  logging.info('Done creating eval obj')
  weights = deterministic_strategy_bandit_loop.run_deterministic_strategy_bandit_loop(
      eval_obj = gsm8k_eval_obj,
      train_obj=train_loop_obj,
      training_batch_generator_obj=dartboard_deterministic_training_batch_generator_obj,
      #training_batch_generator_obj=fixed_dataset_importance_weighting_training_batch_generator_obj,
      init_params = params,
      #step_size=0.00025,
      #step_size=1000,
      #step_size=0.05,
      #step_size=1e-5,
      step_size=0.1,
      #step_size = 1e-6,
      delta=0.01,
      #delta=1e-7,
      #delta=0.001,
      warm_start=False,
      momentum=False,
      step_size_decay=False,
      use_adagrad=False,
      #use_adam=True,
      #adam_beta1=0.8,
      #adam_beta2 = 0.999,
      #adam_beta2=0.999999999,
      #gradient_clipping=True,
      #gradient_clipping_norm=30000,
      #use_adagrad_avg=True,
  )

  #weights = bandit_loop.run_bandit_loop(
  #    eval_obj = gsm8k_eval_obj,
  #    train_obj=train_loop_obj,
  #    #training_batch_generator_obj=importance_weighting_training_batch_generator_obj,
  #    training_batch_generator_obj=dartboard_importance_weighting_training_batch_generator_obj,
  #    #training_batch_generator_obj=fixed_dataset_importance_weighting_training_batch_generator_obj,
  #    #training_batch_generator_obj=fixed_dataset_importance_weighting_training_batch_generator_obj,
  #    init_params = params,
  #    #step_size=0.00025,
  #    #step_size=1000,
  #    step_size=0.1,
  #    #step_size=1e-5,
  #    #step_size=0.1,
  #    #step_size = 1e-6,
  #    delta=0.01,
  #    #delta=1e-7,
  #    #delta=0.001,
  #    warm_start=False,
  #    momentum=False,
  #    step_size_decay=False,
  #    use_adagrad=True,
  #    adagrad_beta=0.1,
  #    #use_adagrad=True,
  #    #adagrad_beta=1e-6,
  #    #adagrad_beta=1e-6,
  #    #adagrad_beta=1e-8,
  #    #use_adam=True,
  #    #adam_beta1=0.8,
  #    #adam_beta2=0.99,
  #    #adam_beta2=0.999999999,
  #    candidate_generator_fn=bandit_loop._generate_gaussian_candidates,
  #    num_grad_evals=8,
  #    #gradient_clipping=True,
  #    #gradient_clipping_norm=30000,
  #    #use_adagrad_avg=True,
  #)