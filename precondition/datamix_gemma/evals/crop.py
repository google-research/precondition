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

"""Byte pair encoding utilities (Adapted from the official GPT-2 GitHub repository)."""

import functools
import json
import os

import regex as re
import requests
import tqdm


data_dir = '/home/shivguptashi/data'


def _get_encoder(subdir):
  """Downloads the encoder and vocab to the subdir."""
  print('Downloading encoder and vocab to ', subdir)
  for filename in ['encoder.json', 'vocab.bpe']:
    r = requests.get(
        'https://openaipublic.blob.core.windows.net/gpt-2/'
        + subdir
        + '/'
        + filename,
        stream=True,
    )
    with open(os.path.join(subdir, filename), 'wb') as f:
      file_size = int(r.headers['content-length'])
      chunk_size = 1000
      with tqdm.tqdm(
          ncols=100,
          desc='Fetching ' + filename,
          total=file_size,
          unit_scale=True,
      ) as pbar:
        # 1k for chunk_size, since Ethernet packet size is around 1500
        # bytes
        for chunk in r.iter_content(chunk_size=chunk_size):
          f.write(chunk)
          pbar.update(chunk_size)


@functools.lru_cache()
def bytes_to_unicode():
  """Returns list of utf-8 byte and a corresponding list of unicode strings.

  The reversible bpe codes work on unicode strings.
  This means you need a large # of unicode characters in your vocab if you
  want to avoid UNKs. When you're at something like a 10B token dataset you
  end up needing around 5K for decent coverage. This is a signficant
  percentage of your normal, say, 32K bpe vocab. To avoid that, we want
  lookup tables between utf-8 bytes and unicode strings. And avoids mapping
  to whitespace/control characters the bpe code barfs on.
  """
  bs = (
      list(range(ord('!'), ord('~') + 1))
      + list(range(ord('¡'), ord('¬') + 1))
      + list(range(ord('®'), ord('ÿ') + 1))
  )
  cs = bs[:]
  n = 0
  for b in range(2**8):
    if b not in bs:
      bs.append(b)
      cs.append(2**8 + n)
      n += 1
  cs = [chr(n) for n in cs]
  return dict(zip(bs, cs))


def get_pairs(word):
  """Return set of symbol pairs in a word.

  Word is represented as tuple of symbols (symbols being variable-length
  strings).

  Args:
    word: A string.

  Returns:
    A set of symbol pairs.
  """
  pairs = set()
  prev_char = word[0]
  for char in word[1:]:
    pairs.add((prev_char, char))
    prev_char = char
  return pairs


class Encoder:
  """Encoder for byte pair encoding."""

  def __init__(self, encoder, bpe_merges, errors='replace'):
    self.encoder = encoder
    self.decoder = {v: k for k, v in self.encoder.items()}
    self.errors = errors  # how to handle errors in decoding
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
    self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    self.cache = {}

    # Should haved added re.IGNORECASE so BPE merges can happen for capitalized
    # versions of contractions
    self.pat = re.compile(
        r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )

  def bpe(self, token):
    """Performs BPE on the given token."""
    if token in self.cache:
      return self.cache[token]
    word = tuple(token)
    pairs = get_pairs(word)

    if not pairs:
      return token

    while True:
      bigram = min(
          pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf'))
      )
      if bigram not in self.bpe_ranks:
        break
      first, second = bigram
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.index(first, i)
          new_word.extend(word[i:j])
          i = j
        except: #pylint: disable=bare-except
          new_word.extend(word[i:])
          break

        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
          new_word.append(first + second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1
      new_word = tuple(new_word)
      word = new_word
      if len(word) == 1:
        break
      else:
        pairs = get_pairs(word)
    word = ' '.join(word)
    self.cache[token] = word
    return word

  def encode(self, text):
    bpe_tokens = []
    for token in re.findall(self.pat, text):
      token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
      bpe_tokens.extend(
          self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' ')
      )
    return bpe_tokens

  def decode(self, tokens):
    text = ''.join([self.decoder[token] for token in tokens])
    text = bytearray([self.byte_decoder[c] for c in text]).decode(
        'utf-8', errors=self.errors
    )
    return text


def get_encoder(model_name):
  """Returns the encoder for the given model."""
  subdir = os.path.join('models', model_name)
  if not os.path.exists(subdir):
    os.makedirs(subdir)
  if not os.path.exists(os.path.join(subdir, 'encoder.json')):
    _get_encoder(subdir)

  subdir = subdir.replace('\\', '/')  # needed for Windows

  with open(os.path.join(subdir, 'encoder.json'), 'r') as f:
    encoder = json.load(f)
  with open(os.path.join(subdir, 'vocab.bpe'), 'r', encoding='utf-8') as f:
    bpe_data = f.read()
  bpe_merges = [
      tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]
  ]
  return Encoder(
      encoder=encoder,
      bpe_merges=bpe_merges,
  )


enc = get_encoder('124M')


def crop_prompt(prompt: str):
  """Crops the prompt to 2048 tokens."""
  global enc  # pylint: disable=global-variable-not-assigned

  cropped_prompt = enc.decode(enc.encode(prompt)[:2048])
  return cropped_prompt


def crop(s):
  """Crops the prompt to 2048 tokens."""
  prompt = crop_prompt(s)
  return prompt
