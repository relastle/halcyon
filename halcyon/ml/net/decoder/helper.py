#  Copyright 2018/10/23 Hiroki Konishi. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from abc import ABCMeta, abstractclassmethod
from typing import Any

import tensorflow as tf
from tensorflow.contrib.seq2seq import TrainingHelper


class TrainingHelperWrapper(metaclass=ABCMeta):

    @abstractclassmethod
    def get_helper(
            self,
            inputs: tf.Tensor,
            sequence_length: tf.Tensor,
            embedding: Any,
            sampling_probability: Any,
            time_major: Any,
            seed: Any,
            scheduling_seed: Any,
            name: Any,
    ) -> TrainingHelper:
        raise NotImplementedError


class NormalHelper(TrainingHelperWrapper):

    def get_helper(
            self,
            inputs: tf.Tensor,
            sequence_length: tf.Tensor,
            embedding: Any,
            sampling_probability: Any,
            time_major: Any,
            seed: Any,
            scheduling_seed: Any,
            name: Any,
    ) -> TrainingHelper:
        helper = tf.contrib.seq2seq.TrainingHelper(
            inputs,
            sequence_length,
        )
        return helper


class ScheduledSamplingHelper(TrainingHelperWrapper):

    def get_helper(
            self,
            inputs: tf.Tensor,
            sequence_length: tf.Tensor,
            embedding: Any,
            sampling_probability: Any,
            time_major: Any,
            seed: Any,
            scheduling_seed: Any,
            name: Any,
    ) -> TrainingHelper:
        helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            inputs=inputs,
            sequence_length=sequence_length,
            embedding=embedding,
            sampling_probability=sampling_probability,
            time_major=time_major,
            seed=seed,
            scheduling_seed=scheduling_seed,
            name=name
        )
        return helper


class ScheduledOutputHelper(TrainingHelperWrapper):

    def get_helper(
            self,
            inputs: tf.Tensor,
            sequence_length: tf.Tensor,
            embedding: Any,
            sampling_probability: Any,
            time_major: Any,
            seed: Any,
            scheduling_seed: Any,
            name: Any,
    ) -> TrainingHelper:
        helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(
            inputs=inputs,
            sequence_length=sequence_length,
            sampling_probability=sampling_probability,
            time_major=time_major,
            seed=seed,
            name=name
        )
        return helper
