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

"""Cross compile the finetune code."""

from collections.abc import Sequence

from absl import app
from absl import flags
from jax.mock_backend import mock_backend
from precondition.datamix_gemma import confusion_matrix_calc


CROSS_COMPILE = flags.DEFINE_boolean('debug', True, 'Run in debug mode')


def cross_compile():
  mock_backend.use_mock_backend(
      topology='8x8',
      chip_config='default',
  )


def main(_: Sequence[str]) -> None:
  cross_compile()
  confusion_matrix_calc.confusion_matrix_calc()
  #finetune_utils.finetune()


if __name__ == '__main__':
  app.run(main)
if __name__ == '__main__':
  app.run(main)
