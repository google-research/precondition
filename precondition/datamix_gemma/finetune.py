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

r"""Finetuning script for Gemma.

POOL=gdm
ALLOC=brain-pton

To run:
xmanager launch experimental/brain_pton/datamix_gemma/finetuning_experiment.py
--     --xm_default_build_flags= --xm_enable_build_isolation=false
--xm_resource_alloc=${POOL}/${ALLOC}

To cross compile:
"""

from collections.abc import Sequence

from absl import flags
from precondition.datamix_gemma import finetune_utils
from python import app


# os.environ['KERAS_BACKEND'] = 'jax'

CROSS_COMPILE = flags.DEFINE_boolean('debug', False, 'Run in debug mode')


def main(_: Sequence[str]) -> None:
  #confusion_matrix_calc.confusion_matrix_calc()
  #snr_calculation.snr_calculation()
  finetune_utils.finetune()
  #finetune_eval_measurement.finetune_eval_measurement()


if __name__ == '__main__':
  app.run(main)
