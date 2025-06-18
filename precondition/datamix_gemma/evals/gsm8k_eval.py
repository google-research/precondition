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

from contextlib import contextmanager
import pickle
import re
import signal
import time

from absl import logging
import file
from flax import jax_utils
import jax
import numpy as np
from precondition.datamix_gemma import deconstructed_sampler as sampler_lib
from precondition.datamix_gemma.dataset_builders.gsm8k_dataset_builder import GSM8KDatasetBuilder
from precondition.datamix_gemma.evals.eval import Eval
from precondition.datamix_gemma.training_loop import validation_step
import tensorflow as tf


gsm8k_pickle_file_prefix = '/home/shivguptashi/gsm8k_pickle'

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"
# @title GSM8K Prompts

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


# Extension of the default 8-shot prompt, page 35 in
# https://arxiv.org/pdf/2201.11903.pdf
# The extension is intended to improve performance on
# more complicated gsm8k examples.

EXTRA_3_SHOTS = """As an expert problem solver solve step by step the following mathematical questions.

Q: Tina makes $18.00 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?
A: Here's how to calculate Tina's earnings:

**Regular Time:**
- Hours per shift: 8 hours
- Wage per hour: $18.00
- Regular pay per shift: 8 hours * $18.00/hour = $144.00

**Overtime:**
- Overtime hours per shift: 10 hours - 8 hours = 2 hours
- Overtime pay per hour: $18.00 + ($18.00 / 2) = $27.00
- Overtime pay per shift: 2 hours * $27.00/hour = $54.00

**Total per day:**
- Regular pay + overtime pay: $144.00/shift + $54.00/shift = $198.00/day

**Total for 5 days:**
- 5 days * $198.00/day = $990.00

**Therefore, Tina will make $990.00 in 5 days.** The answer is 990.

Q: Abigail is trying a new recipe for a cold drink. It uses 1/4 of a cup of iced tea and 1 and 1/4 of a cup of lemonade to make one drink. If she fills a pitcher with 18 total cups of this drink, how many cups of lemonade are in the pitcher?
A: ## Ambiguity in the Problem Statement:

There is one main ambiguity in the problem statement:

**Total volume vs. Number of servings:** The statement "18 total cups of this drink" could be interpreted in two ways:
  * **18 cups of the combined volume:** This would mean Abigail used a total of 18 cups of liquid, including both iced tea and lemonade.
  * **18 individual servings:** This would mean Abigail made 18 individual drinks, each containing 1/4 cup of iced tea and 1 1/4 cup of lemonade.

Let us assume the interpretation "18 cups of the combined volume".

## Solution assuming 18 cups of combined volume:

**Step 1: Find the proportion of lemonade in one drink:**

* Lemonade: 1 1/4 cups
* Iced tea: 1/4 cup
* Total: 1 1/4 + 1/4 = 1 1/2 cups
* Lemonade proportion: (1 1/4) / (1 1/2) = 5/6

**Step 2: Calculate the amount of lemonade in the pitcher:**

* Total volume: 18 cups
* Lemonade proportion: 5/6
* Volume of lemonade: 18 * (5/6) = 15 cups

Therefore, there are 15 cups of lemonade in the pitcher. The answer is 15.

Q: A deep-sea monster rises from the waters once every hundred years to feast on a ship and sate its hunger. Over three hundred years, it has consumed 847 people. Ships have been built larger over time, so each new ship has twice as many people as the last ship. How many people were on the ship the monster ate in the first hundred years?
A: Let us solve it using algebra. Let x be the number of people on the ship the monster ate in the first hundred years.

The number of people on the ship eaten in the second hundred years is 2x, and in the third hundred years is 4x.

Therefore, the total number of people eaten over three hundred years is x + 2x + 4x = 847.

Combining like terms, we get 7x = 847.

Dividing both sides by 7, we find x = 121.

Therefore, there were 121 people on the ship the monster ate in the first hundred years. The answer is 121."""

TEMPLATE = """
Q: {question}
A:"""


# @title Testing library

def find_numbers(x: str) -> list[str]:
  """Finds all numbers in a string."""
  # Search for number, possibly negative (hyphen), with thousand separators
  # (comma), and with a decimal point (period inbetween digits).
  numbers = re.compile(
      r'-?[\d,]*\.?\d+',
      re.MULTILINE | re.DOTALL | re.IGNORECASE,
  ).findall(x)
  return numbers


def find_number(x: str,
                answer_delimiter: str = 'The answer is') -> str:
  """Finds the most relevant number in a string."""
  # If model uses the answer delimiter, then select the first number following
  # that format.
  if answer_delimiter in x:
    answer = x.split(answer_delimiter)[-1]
    numbers = find_numbers(answer)
    if numbers:
      return numbers[0]

  # In general, select the last number in the string.
  numbers = find_numbers(x)
  if numbers:
    return numbers[-1]
  return ''


def maybe_remove_comma(x: str) -> str:
  # Example: 5,600 -> 5600
  return x.replace(',', '')
class GSM8KEval(Eval):

  def __init__(self, model, tokenizer, vocab, eval_batch_size):
    super().__init__(model, tokenizer, vocab, eval_batch_size)
    self.gsm8k_dataset_builder = GSM8KDatasetBuilder(self.tokenizer, 1000)
    self.test_ds = None
    self.pmapped_validation_step = None
    self.sampler = sampler_lib.DeconstructedSampler(
        transformer=model,
        vocab=vocab,
        params={'params': None},
        )
    self.eval_ds = self.gsm8k_dataset_builder.get_question_answer_dataset()
    self.eval_ds = self.eval_ds.batch(eval_batch_size, drop_remainder=False, num_parallel_calls=tf.data.AUTOTUNE)
    self.eval_ds = self.eval_ds.prefetch(tf.data.AUTOTUNE)
    #if jax.process_index() == 0:
    #  #  print(f'tokenized question:', self.sampler.tokenize(str(x['question'])))
    #  #self.eval_ds = self.eval_ds.batch(jax.device_count(), drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE)
    #  self.generate_and_pickle_sampling_states()
    #  #logging.info('Done generating and pickling sampling states!')
    #self.sampling_states, self.total_sampling_steps = None, None
    #while True:
    #  time.sleep(1000000000)
    #exit()
    self.sampling_states, self.total_sampling_steps = self.unpickle_sampling_states()
    logging.info('Done unpickling sample states')


  @contextmanager
  def timeout(self, duration, formula):
      def timeout_handler(signum, frame):
          raise Exception(f"'{formula}': timed out after {duration} seconds")

      signal.signal(signal.SIGALRM, timeout_handler)
      signal.alarm(duration)
      yield
      signal.alarm(0)

  def eval_with_timeout(self, formula, max_time=3):
      try:
          with self.timeout(max_time, formula):
              return eval(formula)
      except Exception as e:
          signal.alarm(0)
          print(f"Warning: Failed to eval {formula}, exception: {e}")
          return None

  def use_calculator(self, sample):
      if "<<" not in sample:
          return None

      parts = sample.split("<<")
      remaining = parts[-1]
      if ">>" in remaining:
          return None
      if "=" not in remaining:
          return None
      lhs = remaining.split("=")[0]
      lhs = lhs.replace(",", "")
      if any([x not in "0123456789*+-/.()" for x in lhs]):
          return None
      return self.eval_with_timeout(lhs)

  def generate_and_pickle_sampling_states(self):
    all_sampling_states = []
    all_total_sampling_steps = []
    for batch_ind, batch in enumerate(self.eval_ds): #pytype: disable=attribute-error
      logging.info(f'Generating sampling states for batch {batch_ind}')
      questions_batch = batch['question'].numpy()
      processed_batch = []
      for i in range(len(questions_batch)):
        logging.info(f'Processing question {i}')
        questions_batch[i] = questions_batch[i].decode('utf-8')
        questions_batch[i] = (PREAMBLE + '\n\n' + PROMPT + '\n' + TEMPLATE.format(question=questions_batch[i]))
        processed_batch.append(questions_batch[i])
      while len(processed_batch) < self.eval_batch_size:
        processed_batch.append('')
      processed_batch = np.array(processed_batch)
      all_input_ids, forbidden_token_ids, total_sampling_steps = self.sampler.generate_tokenized_inputs(processed_batch, total_generation_steps=1024) #pytype: disable=attribute-error
      logging.info(f'Done tokenizing batch {batch_ind}')
      sampling_state = self.sampler.generate_initial_sampling_state( #pytype: disable=attribute-error
          all_input_ids=all_input_ids,
          forbidden_token_ids=forbidden_token_ids,
          total_sampling_steps=total_sampling_steps,
      )
      logging.info(f'Done generating initial sampling state for batch {batch_ind}')
      sampling_state = jax.tree_util.tree_map(
          lambda arr: jax.device_put(arr, jax.local_devices(backend='cpu')[0]),
          sampling_state)
      all_sampling_states.append(sampling_state)
      all_total_sampling_steps.append(total_sampling_steps)
      logging.info(f'Done generating sampling states for batch {batch_ind}')
    cur_pickle_file = gsm8k_pickle_file_prefix + '_' + str(self.eval_batch_size) + '_all_sampling_states' + '.pkl'
    with file.Open(cur_pickle_file, 'wb+') as f:
        pickle.dump(all_sampling_states, f)
    cur_pickle_file = gsm8k_pickle_file_prefix + '_' + str(self.eval_batch_size) + '_all_total_sampling_steps' + '.pkl'
    with file.Open(cur_pickle_file, 'wb+') as f:
        pickle.dump(all_total_sampling_steps, f)

  def unpickle_sampling_states(self):
    cur_pickle_file = (gsm8k_pickle_file_prefix + '_'
      + str(self.eval_batch_size) + '_all_sampling_states' + '.pkl')
    with file.Open(cur_pickle_file, 'rb') as f:
        all_sampling_states = pickle.load(f)
    cur_pickle_file = (gsm8k_pickle_file_prefix + '_' +
      str(self.eval_batch_size) + '_all_total_sampling_steps' + '.pkl')
    with file.Open(cur_pickle_file, 'rb') as f:
        all_total_sampling_steps = pickle.load(f)
    return all_sampling_states, all_total_sampling_steps

  def evaluate(self, params):
    #return 1 + self.evaluate_next_token_loss(params)/270
    #return self.evaluate_next_token_loss(params)
    total_count = 0
    correct_count = 0
    self.sampler.update_params(params) #pytype: disable=attribute-error
    #return self.evaluate_next_token_loss({'params':params})
    all_responses = []
    short_responses = []
    for sampling_state, sampling_steps, batch in zip(self.sampling_states, self.total_sampling_steps, self.eval_ds): #pytype: disable=attribute-error
      questions_batch = batch['question'].numpy()
      answers_batch = batch['answer'].numpy()
      print(f'questions_batch shape: {questions_batch.shape}')
      #for i in range(len(questions_batch)):
        #questions_batch[i] = questions_batch[i].decode('utf-8')
        #questions_batch[i] = (PREAMBLE + '\n\n' + PROMPT + '\n' + TEMPLATE.format(question=questions_batch[i]))
        #answers_batch[i] = answers_batch[i].decode('utf-8')
      #for _ in range(self.sample_len):
        #for i in range(len(questions_batch)):
          #questions_batch[i] += "\nOutput the final answer on the final line of your response in the following format: #### <answer>. Here <answer> is just a number.".encode('utf-8')
        #total_sampling_steps = question_tokens_batch.shape[1] + 256
      #all_input_ids, forbidden_token_ids, total_sampling_steps = self.sampler.generate_tokenized_inputs(questions_batch, total_generation_steps=1024)
      #sampling_state = self.sampler.generate_initial_sampling_state(
      #    all_input_ids=all_input_ids,
      #    forbidden_token_ids=forbidden_token_ids,
      #    total_sampling_steps=total_sampling_steps,
      #)
      logging.info(f'About to sample!')
      token_buffer, num_input_tokens = self.sampler(initial_sampling_state=sampling_state, num_devices=jax.device_count(), echo=False, return_logits=False) #pytype: disable=attribute-error
      logging.info(f'Done sampling!')
      logging.info(f'About to decode outputs!')
      decoded_responses = self.sampler.decode_outputs(token_buffer, num_input_tokens, sampling_steps, echo=False) #pytype: disable=attribute-error
      logging.info('Done decoding responses!')
      logging.info(f'About to check responses!')
      for i in range(len(questions_batch)):
        answers_batch[i] = answers_batch[i].decode('utf-8')
        all_responses.append(decoded_responses[i].split('\nQ:')[0])
        short_responses.append(maybe_remove_comma(find_number(all_responses[-1])))
        print(f"Short answer: {short_responses[-1]}")
        try:
          correct_count += float(maybe_remove_comma(
              find_number(answers_batch[i]))) == float(short_responses[-1])
          print(f'True answer: {float(maybe_remove_comma(find_number(answers_batch[i])))}, Model answer: {float(short_responses[-1])}')
        except:
          correct_count += maybe_remove_comma(
              find_number(answers_batch[i])) == maybe_remove_comma(
                  find_number(short_responses[-1]))
        total_count += 1
      logging.info(f'Done checking responses!')
      #for i in range(len(decoded_responses)):
      #  try:
      #    if decoded_responses[i][-1] == '=':
      #      answer = self.use_calculator(decoded_responses[i])
      #      if answer is not None:
      #        print('Triggered calculator, answer', answer)
      #        questions_batch[i] = decoded_responses[i] + str(answer) + ">>"
      #  except:
      #    questions_batch[i] = decoded_responses[i]
      #  #if self.is_correct(decoded_responses[i], batch['answer'][i]):
      #    #correct_count += 1
      #  #total_count += 1
      #for i in range(len(questions_batch)):
      #  if self.is_correct(questions_batch[i], answers_batch[i]):
      #    correct_count += 1
      #  total_count += 1
    return correct_count / total_count

  def extract_answer(self, completion):
    #print(f'completion: {completion}')
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS

  def is_correct(self, model_completion, gt_answer_str):
    gt_answer = self.extract_answer(gt_answer_str)
    assert gt_answer != INVALID_ANS
    model_completion_answer = self.extract_answer(model_completion)
    #print('True answer:', gt_answer, 'Model answer:', model_completion_answer)
    return model_completion_answer == gt_answer

  def evaluate_next_token_loss(self, params):
    params = {'params': params}
    if self.test_ds is None: # pytype: disable=attribute-error
      self.pmapped_validation_step = jax.pmap(
          lambda params, example_tokens, example_masks: validation_step(
              model=self.model,
              params=params,
              pad_id=self.tokenizer.pad_id,
              example_tokens=example_tokens,
              example_masks=example_masks,
          ),
          axis_name='batch',
      )
      self.test_ds = self.gsm8k_dataset_builder.get_validation_dataset( #pytype: disable=attribute-error
          batch_size=1
      )  # pylint: disable=line-too-long
      self.test_ds = self.test_ds.batch(
          1, drop_remainder=True, num_parallel_calls=tf.data.AUTOTUNE
      )
      self.test_ds = self.test_ds.batch(
          jax.local_device_count(),
          drop_remainder=True,
          num_parallel_calls=tf.data.AUTOTUNE,
      )
      self.test_ds = self.test_ds.prefetch(tf.data.AUTOTUNE)
    params = jax_utils.replicate(params)
    eval_loss = 0
    for batch in self.test_ds:
      batch_input_tokens = batch.input_tokens.numpy()
      batch_target_mask = batch.target_mask.numpy()
      cur_loss = self.pmapped_validation_step(
          #model,
          params,
          # params_vars,
          #tokenizer_pad_id,  # type: ignore
          batch_input_tokens,
          batch_target_mask,
      )
      cur_loss = jax.experimental.multihost_utils.process_allgather(
          cur_loss
      )
      eval_loss += cur_loss
    jax_utils.unreplicate(params)
    return -eval_loss[0][0]
