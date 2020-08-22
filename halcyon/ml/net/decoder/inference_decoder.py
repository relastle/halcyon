from abc import ABCMeta, abstractclassmethod
from typing import Tuple  # noqa
from typing import Any, List

import tensorflow as tf
from halcyon.ml.net.decoder.decoder import Decoder
from halcyon.models.seq import Nuc
from tensorflow.python.layers.core import Dense

COVERAGE_PENALTY_WEIGHT = 0.


class InferenceDecoder(Decoder, metaclass=ABCMeta):
    """
    Recurrent Neural Network Module
    """

    #  Override
    SIGMOID_NOISE = 0.

    @property
    def ignore_alignment_history(self) -> Any:
        return self._ignore_alignment_history

    @property
    def keep_full_alignment(self) -> Any:
        return self._keep_full_alignment

    def __init__(
            self,
            beam_width: int,
            ignore_alignment_history: bool,
            keep_full_alignment: bool,
    ) -> None:
        self._beam_width = beam_width
        self._ignore_alignment_history = ignore_alignment_history
        self._keep_full_alignment = keep_full_alignment
        return

    def get_tag(self) -> str:
        return 'decoder_rnn'

    HP_NAMES = [
        'num_units_hidden',
        'num_units_attention',
        'keep_prob',
    ]

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

    @abstractclassmethod
    def rnn(
            self,
            input_tensor: tf.Tensor,
            output_feature_num: int,
            output_max_len: int,
            ph_batch_size: tf.Tensor,
            ph_seqs_shifted: tf.Tensor,
            ph_start_tokens: tf.Tensor,
            ph_seq_max_lengths: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError
        return None


class InferenceLstmDecoder(InferenceDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.LSTMCell(
            num_units=units_num,
        )

    def rnn(
            self,
            input_tensor: tf.Tensor,
            output_feature_num: int,
            output_max_len: int,
            ph_batch_size: tf.Tensor,
            ph_seqs_shifted: tf.Tensor,
            ph_start_tokens: tf.Tensor,
            ph_seq_max_lengths: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        # attention_states: [batch_size, max_time, self.num_units]
        tiled_encoder_outputs = tf.contrib.seq2seq.tile_batch(
            input_tensor,
            multiplier=self._beam_width,
        )
        tiled_sequence_length = tf.contrib.seq2seq.tile_batch(
            sequence_length,
            multiplier=self._beam_width,
        )

        # Build RNN cell
        # Build RNN cell
        cells = [
            self.get_rnn_cell(
                self['num_units_hidden'],
                self['keep_prob'],
            ) for i in range(self['depth'])
        ]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        #  decoder_cell = self.get_rnn_cell(self['num_units_hidden'])

        # Create an attention mechanism
        attention_mechanism = self.get_attention_mechanism(
            self['num_units_hidden'],
            tiled_encoder_outputs,
            memory_sequence_length=tiled_sequence_length,
            mode=self['mode'],
        )

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            multi_rnn_cell,
            attention_mechanism,
            attention_layer_size=self['num_units_attention'],
            alignment_history=not self._ignore_alignment_history,
        )

        # Fully connected layer to infer the logits of nucleotide sequence.
        projection_layer = Dense(output_feature_num, use_bias=True)

        # Decoder
        decoder_initial_state = decoder_cell.zero_state(
            dtype=tf.float32,
            batch_size=ph_batch_size * self._beam_width,
        )
        decoder = tf.contrib.seq2seq.BeamSearchDecoder(
            cell=decoder_cell,
            embedding=lambda ids: tf.one_hot(
                ids,
                depth=output_feature_num,
            ),
            start_tokens=tf.to_int32(ph_start_tokens),
            end_token=Nuc.NUC_MAP['$'],  # TOKEN id of `$`
            initial_state=decoder_initial_state,
            beam_width=self._beam_width,
            output_layer=projection_layer,
            #  length_penalty_weight=0.0,
            #  coverage_penalty_weight=COVERAGE_PENALTY_WEIGHT,
        )

        # Dynamic decoding
        (
            self._decoder_outputs,
            self._final_context_state,
            _,
        ) = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            output_time_major=False,
            maximum_iterations=output_max_len + 10,
        )
        return self._decoder_outputs, self._final_context_state


class InferenceGruDecoder(InferenceLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.GRUCell(
            num_units=units_num,
        )


class InferenceLstmBlockDecoder(InferenceLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.LSTMBlockCell(
            num_units=units_num,
        )


class InferenceGruBlockDecoder(InferenceLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.GRUBlockCellV2(
            num_units=units_num,
        )


class InferenceSruDecoder(InferenceLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.SRUCell(
            num_units=units_num,
        )
