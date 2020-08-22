from abc import ABCMeta
from typing import Tuple  # noqa
from typing import Any

import tensorflow as tf
from halcyon.ml.net.sub_net import SubNet


class Decoder(SubNet, metaclass=ABCMeta):

    SIGMOID_NOISE = 0.0
    SCORE_BIAS_INIT = 0.0

    HP_NAMES = [
        'depth',
        'num_units_hidden',
        'num_units_attention',
        'keep_prob',
        'mode',
        'monotonic',
        'attention_type',
    ]

    def __init__(
            self,
            is_inference: bool,
    ) -> None:
        self.is_inference = is_inference
        return

    def get_attention_mechanism(
            self,
            num_units: int,
            memory: tf.Tensor,
            memory_sequence_length: tf.Tensor,
            mode: str,
    ) -> Any:
        if self['monotonic']:
            if self['attention_type'].lower() == 'luong':
                return tf.contrib.seq2seq.LuongMonotonicAttention(
                    num_units=num_units,
                    memory=memory,
                    memory_sequence_length=memory_sequence_length,
                    scale=True,
                    sigmoid_noise=self.SIGMOID_NOISE,
                    score_bias_init=self.SCORE_BIAS_INIT,
                    mode=mode,
                )
            elif self['attention_type'].lower() == 'bahdanau':
                return tf.contrib.seq2seq.BahdanauMonotonicAttention(
                    num_units=num_units,
                    memory=memory,
                    memory_sequence_length=memory_sequence_length,
                    sigmoid_noise=self.SIGMOID_NOISE,
                    score_bias_init=self.SCORE_BIAS_INIT,
                    mode=mode,
                )
        else:
            if self['attention_type'].lower() == 'luong':
                return tf.contrib.seq2seq.LuongAttention(
                    num_units=num_units,
                    memory=memory,
                    memory_sequence_length=memory_sequence_length,
                    scale=True,
                )
            elif self['attention_type'].lower() == 'bahdanau':
                return tf.contrib.seq2seq.BahdanauAttention(
                    num_units=num_units,
                    memory=memory,
                    memory_sequence_length=memory_sequence_length,
                )
