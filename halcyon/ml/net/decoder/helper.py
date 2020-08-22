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
