from abc import ABCMeta, abstractclassmethod
from typing import Tuple  # noqa
from typing import List

import tensorflow as tf
from halcyon.ml.net.decoder.decoder import Decoder
from halcyon.ml.net.decoder.helper import TrainingHelperWrapper  # noqa
from halcyon.ml.net.decoder.helper import ScheduledSamplingHelper
from halcyon.ml.net.encoder import DeviceCellWrapper  # noqa
from tensorflow.python.layers.core import Dense


class TrainDecoder(Decoder, metaclass=ABCMeta):
    '''
    Recurrent Neural Network Module
    '''

    @property
    def helper(self) -> TrainingHelperWrapper:
        return self._helper

    @helper.setter
    def helper(self, helper: TrainingHelperWrapper) -> None:
        self._helper = helper
        return

    def __init__(self, is_inference: bool) -> None:
        super(TrainDecoder, self).__init__(is_inference)
        self._helper = ScheduledSamplingHelper()
        return

    def get_tag(self) -> str:
        return 'decoder_rnn'

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
    ) -> tf.Tensor:
        raise NotImplementedError
        return None


class TrainLstmDecoder(TrainDecoder):

    HP_NAMES = (
        TrainDecoder.HP_NAMES +
        ['sampling_probability']
    )

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

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
    ) -> tf.Tensor:
        ##################
        # Decoder LSTM
        ##################
        # Helper
        #  helper = tf.contrib.seq2seq.TrainingHelper(
        #      ph_seqs_shifted,
        #      ph_seq_max_lengths,
        #  )
        helper = self._helper.get_helper(
            inputs=ph_seqs_shifted,
            sequence_length=ph_seq_max_lengths,
            embedding=lambda ids: tf.one_hot(
                ids,
                depth=output_feature_num,
            ),
            sampling_probability=self['sampling_probability'],
            time_major=False,
            seed=None,
            scheduling_seed=None,
            name=None
        )

        # Build RNN cell
        cells = [
            self.get_rnn_cell(
                self['num_units_hidden'],
                self['keep_prob'],
            ) for i in range(self['depth'])
        ]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)

        # Create an attention mechanism
        attention_mechanism = self.get_attention_mechanism(
            self['num_units_hidden'],
            input_tensor,
            memory_sequence_length=sequence_length,
            mode=self['mode'],
        )

        decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
            multi_rnn_cell,
            attention_mechanism,
            attention_layer_size=self['num_units_attention'],
            alignment_history=True,
        )

        # Fully connected layer to infer the logits of nucleotide sequence.
        projection_layer = Dense(output_feature_num, use_bias=True)

        # Decoder
        decoder = tf.contrib.seq2seq.BasicDecoder(
            decoder_cell,
            helper,
            initial_state=decoder_cell.zero_state(
                dtype=tf.float32,
                batch_size=ph_batch_size,
            ),
            output_layer=projection_layer,
        )

        # Dynamic decoding
        (
            self._decoder_outputs,
            self._final_context_state,
            _,
        ) = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=output_max_len + 10,
        )
        self._logits = self._decoder_outputs.rnn_output
        return self._logits


class TrainSruDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.SRUCell(
            num_units=units_num,
        )


class TrainLstmBlockDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        cell = tf.contrib.rnn.LSTMBlockCell(
            num_units=units_num,
        )
        return cell


class TrainGruBlockDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.GRUBlockCellV2(
            num_units=units_num,
        )


class TrainIndyLstmDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.IndyLSTMCell(
            num_units=units_num,
        )


class TrainUgrnnDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.UGRNNCell(
            num_units=units_num,
        )


class TrainNasDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.NASCell(
            num_units=units_num,
        )


class TrainFusedDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        #  cell = tf.contrib.rnn.BasicRNNCell(units_num)
        #  fw_lstm = tf.contrib.rnn.FusedRNNCellAdaptor(
        #      cell,
        #      use_dynamic_rnn=True,
        #  )
        #  return fw_lstm
        return tf.contrib.rnn.LSTMBlockFusedCell(
            num_units=units_num,
        )


class TrainCudnnDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        #  return DeviceCellWrapper(
        #      tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
        #          num_units=units_num,
        #      ), device='/gpu:0')
        return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
            num_units=units_num,
        )


class TrainGruDecoder(TrainLstmDecoder):

    def get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.GRUCell(
            num_units=units_num,
        )
