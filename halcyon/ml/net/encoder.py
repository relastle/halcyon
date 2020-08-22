from abc import ABCMeta, abstractclassmethod
from typing import Tuple  # noqa
from typing import Any, List

import tensorflow as tf
from halcyon.ml.net.cnn import GatedLinearBlocks
from halcyon.ml.net.sub_net import SubNet


class DeviceCellWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell: Any, device: Any) -> None:
        self._cell = cell
        self._device = device
        return

    @property
    def state_size(self) -> Any:
        return self._cell.state_size

    @property
    def output_size(self) -> Any:
        return self._cell.output_size

    def __call__(
            self,
            inputs: Any,
            state: Any,
            scope: Any = None,
    ) -> Any:
        with tf.device(self._device):
            return self._cell(inputs, state, scope)


class Encoder(SubNet, metaclass=ABCMeta):
    """
    Recurrent Neural Network Module
    """

    def __init__(
        self,
        is_inference: bool,
        verbose: bool = False,
    ) -> None:
        if not verbose:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            tf.get_logger().setLevel('ERROR')
        self.is_inference = is_inference
        return

    def get_tag(self) -> str:
        return 'rnn'

    @abstractclassmethod
    def rnn(
            self,
            input_tensor: tf.Tensor,
            ph_batch_size: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> tf.Tensor:
        raise NotImplementedError
        return None


class UnidirectionalLstmEncoder(Encoder):

    def get_tag(self) -> str:
        return 'encoder_rnn'

    HP_NAMES = [
        'num_units',
        'layer_depth',
        'keep_prob',
    ]

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.LSTMCell(
        )

    def rnn(
            self,
            input_tensor: tf.Tensor,
            ph_batch_size: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> tf.Tensor:
        input_tensor_transposed = tf.transpose(
            input_tensor,
            [1, 0, 2],
        )
        # create 2 LSTMCells
        rnn_layers = [
            tf.nn.rnn_cell.LSTMCell(size)
            for size in self['layer_depth'] * [self['num_units']]
        ]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        initial_state = multi_rnn_cell.zero_state(
            ph_batch_size,
            dtype=tf.float32,
        )

        # 'outputs' is a tensor of shape [batch_size, max_time, 256]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        outputs, state = tf.nn.dynamic_rnn(
            cell=multi_rnn_cell,
            inputs=input_tensor_transposed,
            dtype=tf.float32,
            sequence_length=sequence_length,
            initial_state=initial_state,
            time_major=True,
        )
        encoder_outputs = outputs
        encoder_outputs_transposed = tf.transpose(encoder_outputs, [1, 0, 2])
        return encoder_outputs_transposed


class BidirectionalLstmEncoder(Encoder):

    def get_tag(self) -> str:
        return 'encoder_rnn'

    HP_NAMES = [
        'num_units',
        'layer_depth',
        'keep_prob',
    ]

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

    def _get_rnn_cell(
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
            ph_batch_size: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> tf.Tensor:
        if self['layer_depth'] == 0:
            return input_tensor

        encoder_cells_fw = []
        encoder_cells_bw = []
        init_states_fw = []
        init_states_bw = []
        for i in range(self['layer_depth']):
            encoder_cells_fw.append(
                self._get_rnn_cell(
                    self['num_units'],
                    self['keep_prob'],
                ))
            encoder_cells_bw.append(
                self._get_rnn_cell(
                    self['num_units'],
                    self['keep_prob'],
                ))
            init_states_fw.append(encoder_cells_fw[i].zero_state(
                ph_batch_size,
                tf.float32,
            ))
            init_states_bw.append(encoder_cells_bw[i].zero_state(
                ph_batch_size,
                tf.float32,
            ))
        #  multi_cells_fw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_fw)
        #  multi_cells_bw = tf.nn.rnn_cell.MultiRNNCell(encoder_cells_bw)
        input_tensor_transposed = tf.transpose(
            input_tensor,
            [1, 0, 2],
        )
        #  outputs, _ = tf.nn.bidirectional_dynamic_rnn(
        #          cell_fw=multi_cells_fw,
        #          cell_bw=multi_cells_bw,
        #          inputs=input_tensor_transposed,
        #          sequence_length=sequence_length,
        #          dtype=tf.float32,
        #          time_major=True,
        #          )
        outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=encoder_cells_fw,
            cells_bw=encoder_cells_bw,
            inputs=input_tensor_transposed,
            sequence_length=sequence_length,
            initial_states_fw=init_states_fw,
            initial_states_bw=init_states_bw,
            dtype=tf.float32,
            time_major=True,
        )
        encoder_outputs = tf.concat(outputs, 2)
        encoder_outputs_transposed = tf.transpose(encoder_outputs, [1, 0, 2])
        return encoder_outputs_transposed


class BidirectionalSruEncoder(BidirectionalLstmEncoder):
    """
    SRU, Simple Recurrent Unit.
    (https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/SRUCell
    """

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.SRUCell(
            num_units=units_num,
        )


class BidirectionalLstmBlockEncoder(BidirectionalLstmEncoder):
    """
    http://arxiv.org/abs/1409.2329
    """

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.LSTMBlockCell(
            units_num,
        )


class BidirectionalGruBlockEncoder(BidirectionalLstmEncoder):

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.GRUBlockCellV2(
            units_num,
        )


class BidirectionalIndyLstmEncoder(BidirectionalLstmEncoder):
    """
    http://arxiv.org/abs/1409.2329
    """

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.IndyLSTMCell(
            units_num,
        )


class BidirectionalUgrnnEncoder(BidirectionalLstmEncoder):
    """
    https://arxiv.org/abs/1611.09913
    "Capacity and Trainability in Recurrent Neural Networks"
    """

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.UGRNNCell(
            units_num,
        )


class BidirectionalNasEncoder(BidirectionalLstmEncoder):
    """
    https://arxiv.org/abs/1611.01578
    """

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.NASCell(
            units_num,
        )


class BidirectionalFusedEncoder(BidirectionalLstmEncoder):
    """
    http://arxiv.org/abs/1409.2329
    """

    # Override
    def rnn(
            self,
            input_tensor: tf.Tensor,
            ph_batch_size: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> tf.Tensor:
        input_tensor_transposed = tf.transpose(
            input_tensor,
            [1, 0, 2],
        )
        lstm_fw = tf.contrib.rnn.LSTMBlockFusedCell(
            self['num_units'],
            forget_bias=1.0,
        )
        lstm_bw = tf.contrib.rnn.LSTMBlockFusedCell(
            self['num_units'],
            forget_bias=1.0,
        )
        lstm_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_bw)
        #  cell = tf.contrib.rnn.BasicRNNCell(self['num_units'])
        #  lstm_fw = tf.contrib.rnn.FusedRNNCellAdaptor(
        #          cell,
        #          use_dynamic_rnn=True,
        #          )
        #  lstm_bw = tf.contrib.rnn.TimeReversedFusedRNN(fw_lstm)
        fw_out, fw_state = lstm_fw(
            inputs=input_tensor_transposed,
            initial_state=None,
            dtype=tf.float32,
            sequence_length=sequence_length,
        )
        bw_out, bw_state = lstm_bw(
            inputs=input_tensor_transposed,
            initial_state=None,
            dtype=tf.float32,
            sequence_length=sequence_length,
        )
        encoder_outputs = tf.concat([fw_out, bw_out], 2)
        encoder_outputs_transposed = tf.transpose(encoder_outputs, [1, 0, 2])
        return encoder_outputs_transposed

    # Override
    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.LSTMBlockCell(
                units_num,
                forget_bias=1.0,
            ),
            output_keep_prob=keep_prob,
            dtype=tf.float32
        )
        return lstm_cell


class BidirectionalCudnnLSTMCell(BidirectionalLstmEncoder):

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        #  return DeviceCellWrapper(
        # tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
        #          num_units=units_num,
        #          ), device='/gpu:0')
        return tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
            num_units=units_num,
        )


class BidirectionalGruEncoder(BidirectionalLstmEncoder):

    def _get_rnn_cell(
            self,
            units_num: int,
            keep_prob: float = 1.,
    ) -> tf.Tensor:
        return tf.contrib.rnn.GRUCell(
            num_units=units_num,
        )


class BidirectionalLstmResidualEncoder(BidirectionalLstmEncoder):

    def rnn(
            self,
            input_tensor: tf.Tensor,
            ph_batch_size: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> tf.Tensor:
        x = tf.layers.conv1d(
            inputs=input_tensor,
            filters=self['num_units'] * 2,
            kernel_size=1,
            strides=1,
            padding='same',
        )
        encoder_outputs = super(BidirectionalLstmResidualEncoder, self).rnn(
            input_tensor,
            ph_batch_size,
            sequence_length,
        )
        return x + encoder_outputs


class GatedLinearBlocksEncoder(Encoder):

    def __init__(
            self,
            is_inference: bool,
    ) -> None:
        super(GatedLinearBlocksEncoder, self).__init__(is_inference)
        self._glb = GatedLinearBlocks(is_inference)
        return

    def get_tag(self) -> str:
        return 'encoder_rnn'

    HP_NAMES = [
        'num_units',
        'kernel_size',
        'layer_depth',
        'use_bn',
    ]

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

    def rnn(
            self,
            input_tensor: tf.Tensor,
            ph_batch_size: tf.Tensor,
            sequence_length: tf.Tensor,
    ) -> tf.Tensor:
        x = input_tensor
        for gbl_i in range(self['layer_depth']):
            with tf.variable_scope('GLB_encoder_no_stride_{}'.format(gbl_i + 1)):
                _x = self._glb.gated_linear_block(
                    x,
                    filters=self['num_units'],
                    kernel_size=self['kernel_size'],
                    strides=1,
                    padding='SAME',
                    use_bn=self['use_bn'],
                )
                if gbl_i == 0:
                    x = _x
                else:
                    x = x + _x
        return x
