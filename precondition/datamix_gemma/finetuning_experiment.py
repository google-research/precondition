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

"""Experiment to finetune GEMMA."""

from typing import Sequence

from absl import flags
from absl import logging
from python import app
import tensorflow as tf
from xmanager import xm
from xmanager import xm_abc
from xmanager.contrib.internal import xm_jax


CROSS_COMPILE = flags.DEFINE_boolean('debug', False, 'Run in debug mode')


def main(_: Sequence[str]) -> None:
  """Launches finetuning experiment."""
  title = 'finetune_gemma'
  logging.set_verbosity(logging.INFO)
  tf.config.set_soft_device_placement(True)

  executor = xm_abc.Borg(
      requirements=xm.JobRequirements(
      ),
  )
  with xm_abc.create_experiment(experiment_title=title) as experiment:
    [executable] = experiment.package([
        xm.bazel_binary(
            label='//experimental/brain_pton/datamix_gemma:finetune',
            executor_spec=xm_abc.Borg.Spec(),
            args=xm_jax.JaxFlags().flags(),
            bazel_args=xm_abc.bazel_args.tpu(),
        ),
    ])
    experiment.add(
        xm.Job(
            executable,
            #args=exe_args,
            env_vars={'HF_DATASETS_CACHE': '/tmp/hf_datasets_cache',
                      'KERAS_BACKEND': 'jax'},
            executor=executor,
        )
    )


if __name__ == '__main__':
  app.run(main)
