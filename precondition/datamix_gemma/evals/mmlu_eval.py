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

"""MMLU eval."""

import concurrent
import pickle

from absl import logging
import file
import jax
import numpy as np
import pandas as pd
from precondition.datamix_gemma import deconstructed_sampler as sampler_lib
from precondition.datamix_gemma.evals.eval import Eval


choices = ['A', 'B', 'C', 'D']
data_dir = '/home/shivguptashi/data'
pickle_file_prefix = '/home/shivguptashi/new_mmlu_pickle'
ntrain = 5

def format_subject(subject):
    l = subject.split('_')
    s = ''
    for entry in l:
        s += ' ' + entry
    return s

def format_example(df, idx, include_answer=True):
    prompt = df.iloc[idx, 0]
    k = df.shape[1] - 2
    for j in range(k):
        prompt += "\n{}. {}".format(choices[j], df.iloc[idx, j+1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, k + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(format_subject(subject))
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt

def generate_subject_prompts(subject, dev_df, test_df, batch_size):
    """Evaluates the model on the test set."""
    cors = []
    all_probs = []
    all_prompts = []
    all_labels = []
    answers = choices[:test_df.shape[1]-2]
    for i in range(0, test_df.shape[0]):
        # get prompt and make sure it fits
        k = ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        all_prompts.append(prompt)

            #while crop(prompt) != prompt:
            #    k -= 1
            #    train_prompt = gen_prompt(dev_df, subject, k)
            #    prompt = train_prompt + prompt_end

        label = test_df.iloc[i, test_df.shape[1]-1]
        all_labels.append(label)

        #preds = letter_sampler_fn(prompts=prompts)
        #prompts.clear()
        #print('predictions are:', preds)

        #for j in range(i, min(i+batch_size, test_df.shape[0])):
        #    pred = preds[j-i]
        #    label = labels[j-i]
        #    cors.append(pred == label)
        #all_probs.append(probs)
    return all_prompts, all_labels
    acc = np.mean(cors)
    cors = np.array(cors)

    all_probs = np.array(all_probs)
    print("Average accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc



def generate_and_pickle_initial_sampling_states(sampler, tokenized_prompts, max_total_sampling_steps, eval_batch_size):
    for i in range(len(tokenized_prompts)):
        print(f'tokenized prompt: {tokenized_prompts[i]}')
        all_input_ids, forbidden_token_ids = tokenized_prompts[i]
        cur_sampling_state = sampler.generate_initial_sampling_state(
            all_input_ids=all_input_ids,
            forbidden_token_ids=forbidden_token_ids,
            total_sampling_steps=max_total_sampling_steps,
        )
        cur_sampling_state = jax.tree_util.tree_map(
            lambda arr: jax.device_put(arr, jax.local_devices(backend='cpu')[0]),
            cur_sampling_state)
        cur_pickle_file = pickle_file_prefix + '_' + str(eval_batch_size) + '_' + str(i) + '.pkl'
        with file.Open(cur_pickle_file, 'wb+') as f:
            pickle.dump(cur_sampling_state, f)
            logging.info(f'Pickled batch: {i}')
        #if i >= 10:
        #    break

class MMLUEval(Eval):
    def __init__(self, model, tokenizer, vocab, eval_batch_size):
        super().__init__(model, tokenizer, vocab, eval_batch_size)
        self.sampler = sampler_lib.DeconstructedSampler(
            transformer=model,
            vocab=vocab,
            params={'params': None},
            )
        self.option_tokens = self.generate_letter_token_dict_()
        self.prompts, self.labels, self.subjects = self.generate_all_prompts_(batch_size=self.eval_batch_size)
        tokenized_states, self.max_total_sampling_steps = self.generate_mmlu_tokenized_states_()
        #generate_and_pickle_initial_sampling_states(self.sampler, tokenized_states, self.max_total_sampling_steps, eval_batch_size)
        self.initial_sampling_states = []
        for i in range(len(self.labels)):
            self.initial_sampling_states.append(self.unpickle_initial_sampling_states_(i))

    def evaluate(self, params):
        self.sampler.update_params(params)
        return self.mmlu_eval_()

    def generate_responses_(self, initial_sampling_state):
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

    def generate_single_letter_(self, responses):
        results = []
        for response in responses:
            results.append(' ')
            for token in response:
                if int(token) in self.option_tokens:
                    results[-1] = self.option_tokens[int(token)]
                    break
        return results

    def generate_letter_token_dict_(self):
        vocab_list = [(id, self.vocab.IdToPiece(id)) for id in range(self.vocab.GetPieceSize())]
        letters = ['A', 'B', 'C', 'D']
        res_dict = {}
        for id, piece in vocab_list:
            try:
                letter = piece[piece.find(next(filter(str.isalpha, piece)))]
                if letter in letters:
                    res_dict[id] = letter
            except:
                pass
        return res_dict

    def mmlu_eval_(self):
        sampling_states, total_sampling_steps= self.generate_all_responses_()
        responses = []
        for i in range(len(sampling_states)):
            logging.info(f'total_sampling_steps[i]: {total_sampling_steps[i]}')
            responses.append(self.sampler.post_process_sampling_state(
                    sampling_states[i][0],
                    sampling_states[i][1],
                    total_sampling_steps[i],
                    return_logits=False,
                    echo=False))

        logging.info('Running mmlu_eval')
        subject_accs = []
        subject_cors = {}
        #for each subject
        for i in range(len(self.labels)):
            #initial_state = unpickle_initial_sampling_states(i)
            label_batch = self.labels[i]
            preds = self.generate_single_letter_(responses[i])
            for k in range(len(label_batch)):
                if self.subjects[i][k] not in subject_cors:
                    subject_cors[self.subjects[i][k]] = []
                subject_cors[self.subjects[i][k]].append(preds[k] == label_batch[k])
        for subject in subject_cors:
            subj_acc = np.mean(subject_cors[subject])
            logging.info(f'Subject: {subject}, accuracy: {subj_acc}')
            subject_accs.append(subj_acc)
        weighted_acc =np.mean(subject_accs)
        logging.info("Weighted Average accuracy: {:.3f}".format(weighted_acc))
        return weighted_acc

    def generate_all_prompts_(self, batch_size):
        """Evaluates the model on the MMLU eval."""
        subjects = sorted([
            f.split("_test.csv")[0] for f in file.ListDir(data_dir + '/test')
            if "_test.csv" in f])

        print(subjects)

        all_prompts = []
        all_labels = []
        all_subjects = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            dev_dfs = executor.map(
                lambda subject: pd.read_csv(
                    (data_dir + "/dev/" + subject + "_dev.csv"), header=None
                ),
                subjects,
            )
            test_dfs = executor.map(
                lambda subject: pd.read_csv(
                    (data_dir + "/test/" + subject + "_test.csv"), header=None
                ),
                subjects,
            )
            for subject, dev_df, test_df in zip(subjects, dev_dfs, test_dfs):
                prompts, labels = generate_subject_prompts(subject, dev_df[:ntrain], test_df, batch_size)
                for i in range(len(prompts)):
                    all_prompts.append(prompts[i])
                    all_labels.append(labels[i])
                    all_subjects.append(subject)
                logging.info(f'Processed subject: {subject}')
        batched_prompts = []
        batched_labels = []
        batched_subjects = []
        for i in range(0, len(all_prompts), batch_size):
            end_ind = min(len(all_prompts), i + batch_size)
            batched_prompts.append(all_prompts[i:end_ind])
            batched_labels.append(all_labels[i:end_ind])
            batched_subjects.append(all_subjects[i:end_ind])
        return batched_prompts, batched_labels, batched_subjects

    def generate_all_responses_(self):
        responses = []
        total_sampling_steps = []
        for i in range(len(self.labels)):
            logging.info(f'Generating responses, batch {i}')
            #initial_sampling_states = self.unpickle_initial_sampling_states_(i)
            response = self.generate_responses_(self.initial_sampling_states[i])
            responses.append(response)
            total_sampling_steps.append(self.initial_sampling_states[i].total_sampling_steps)
        return responses, total_sampling_steps

    def unpickle_initial_sampling_states_(self, i):
        cur_pickle_file = pickle_file_prefix + '_' +str(self.eval_batch_size) + '_' + str(i) + '.pkl'
        with file.Open(cur_pickle_file, 'rb') as f:
            init_sampling_states = pickle.load(f)
        return init_sampling_states

    def generate_mmlu_tokenized_states_(self):
        tokenized_states = []
        max_total_sampling_steps = 0
        for i in range(len(self.prompts)):
            prompts_batch = self.prompts[i]
            while(len(prompts_batch) < self.eval_batch_size):
                prompts_batch.append('')
            all_input_ids, forbidden_token_ids, total_sampling_steps= self.sampler.generate_tokenized_inputs(
                prompts_batch, total_generation_steps=2
            )
            tokenized_states.append(
                (all_input_ids, forbidden_token_ids)
            )
            max_total_sampling_steps = max(max_total_sampling_steps, total_sampling_steps)
            logging.info(f'Tokenized batch: {i}')
        return tokenized_states, max_total_sampling_steps