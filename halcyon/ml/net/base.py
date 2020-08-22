import datetime
import json
import logging
import math
import os
import sys
from typing import Tuple  # noqa
from typing import Any, Dict

import logzero
import numpy as np
import tensorflow as tf
from halcyon.ml.net.cnn import CnnBase
from halcyon.ml.net.encoder import Encoder
from halcyon.ml.net.saver import Saver
from logzero import logger

MODEL_CKPT_PREFIX = 'model.ckpt'


def calc_conv_dim(in_size: int, kernel_size: int, stride: int) -> int:
    return math.ceil(float(in_size - kernel_size + 1) / float(stride))


class Net(object):
    """
    Attiributes:
        model
    """

    """
    These paremeters has no effect to the
    weight dimension of architecture.
    In other words, there parameters can be
    changed even for restored model.
    """
    RUNTIME_PARAMS = [
        'minibatch_size',
        'init_learning_rate',
        'clip_norm',
        'decay_steps',
        'decay_rate',
    ]

    @property
    def minibatch_size(self) -> int:
        return self._rt_params['minibatch_size']  # type: ignore

    @property
    def init_learning_rate(self) -> int:
        return self._rt_params['init_learning_rate']  # type: ignore

    @property
    def clip_norm(self) -> float:
        return self._rt_params['clip_norm']  # type: ignore

    @property
    def decay_steps(self) -> int:
        return self._rt_params['decay_steps']  # type: ignore

    @property
    def decay_rate(self) -> float:
        return self._rt_params['decay_rate']  # type: ignore

    BEGINNING_ALLOW_MISS_RATIO = 0.0

    def __set_logger(self) -> None:
        self.logger = logzero.logger
        level = logging.DEBUG if self._verbose else logging.ERROR
        self.logger = logzero.setup_logger(
            name='NetTrainLogger',
            logfile=os.path.join(
                self._workspace,
                'log.txt',
            ),
            fileLoglevel=level,
            level=level,
        )
        self.logger.info('set up logger')
        return

    def __init__(
            self,
            workspace_base_dir: str = '',
            name: str = '',
            ID: str = '',
            verbose: bool = False,
    ) -> None:
        self._is_restore = False
        self._restore_ckpt_dir_path = ''
        self._verbose = verbose
        # give id based on now timestamp
        if ID == '':
            now_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            self.id = '{}_{}'.format(
                now_str,
                name,
            )
        else:
            self.id = ID
        logger.info('ID = {}'.format(self.id))
        # prepare the ckpt_dir
        if workspace_base_dir:
            self._workspace = os.path.join(
                workspace_base_dir,
                self.id,
            )
        else:
            self._workspace = ''
        if self._workspace:
            os.makedirs(self._workspace, exist_ok=True)
            self.__set_logger()
        return

    @property
    def workspace(self) -> str:
        return self._workspace

    @workspace.setter
    def workspace(self, workspace: str) -> None:
        self._workspace = workspace
        return

    @property
    def restore_ckpt_dir_path(self) -> str:
        return self._restore_ckpt_dir_path

    @property
    def is_restore(self) -> bool:
        return self._is_restore

    @property
    def hps(self) -> Dict[str, Any]:
        return self._hps

    @property
    def rt_params(self) -> Dict[str, Any]:
        return self._rt_params

    @property
    def cnn(self) -> CnnBase:
        return self._cnn

    @cnn.setter
    def cnn(self, cnn: CnnBase) -> None:
        self._cnn = cnn
        return

    @property
    def encoder_rnn(self) -> Encoder:
        return self._encoder_rnn

    @encoder_rnn.setter
    def encoder_rnn(self, rnn: Encoder) -> None:
        self._encoder_rnn = rnn
        return

    def get_architecture_type(self) -> str:
        return self.__class__.__name__

    def _make_sequence_length(
            self,
            in_tensor: tf.Tensor,
    ) -> tf.Tensor:
        used = tf.sign(tf.reduce_max(tf.abs(in_tensor), 2))
        length = tf.reduce_sum(used, 1)
        length = tf.maximum(length, tf.ones(tf.shape(length)))
        length = tf.cast(length, tf.int32)
        return length

    def _make_target_weight(
            self,
            ys: np.ndarray,
            y_lengths: np.ndarray,
    ) -> np.ndarray:
        target_weight = np.ones(shape=(ys.shape[0], ys.shape[1]))
        allow_miss_of_beginning = int(
            ys.shape[1] * self.BEGINNING_ALLOW_MISS_RATIO
        )
        target_weight[:, :allow_miss_of_beginning] = 0
        for i, length in enumerate(y_lengths):
            target_weight[i, length:] = 0
        return target_weight

    def get_ready(
            self,
            sess: tf.Session,
            ckpt_model_dir_path: str = '',
            max_to_keep: int = 10,
    ) -> None:
        """ Getting ready to train network.
        This method initializes the variables, so you should call
        this under the tensorflow session context.

        Args:
            sess: configured TensorFlow Session object.
            ckpt_model_dir_path:
                check point directory path from which
                networks will be restored.  if set '',
                the networks will be initialized by
                tf.global_variables_initializer()
        """
        self._saver = Saver(max_to_keep=max_to_keep)
        if not self.is_restore:
            sess.run(tf.global_variables_initializer())
            return

        if (ckpt_model_dir_path == '' and
                self.restore_ckpt_dir_path != ''):
            self._saver.restore_auto(
                sess,
                os.path.join(self.restore_ckpt_dir_path),
            )
        else:
            self._saver.restore(
                sess,
                os.path.join(ckpt_model_dir_path, MODEL_CKPT_PREFIX),
            )
        return

    def __check_rt_keys(self) -> bool:
        for name in self.RUNTIME_PARAMS:
            if name not in self._rt_params.keys():
                sys.stderr.write(
                    'Key {} not found in json\n'.format(
                        name
                    ))
                return False
        return True

    def load_rt_params(self, d: Dict[str, Any]) -> Dict[str, Any]:
        self._rt_params = d
        if not self.__check_rt_keys():
            raise KeyError
        return d

    def read_rt_params(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r') as json_f:
            d = json.load(json_f)
        return self.load_rt_params(d)

    def get_hp_dict_scheme(self) -> Dict[str, Any]:
        d = {}  # type: Dict[str, Any]
        d[self.cnn.tag] = {}
        for name in self.cnn.get_hp_names():
            d[self.cnn.tag][name] = ''

        d[self.encoder_rnn.tag] = {}
        for name in self.encoder_rnn.get_hp_names():
            d[self.encoder_rnn.tag][name] = ''
        return d

    def read_hps(self, json_path: str) -> Dict[str, Any]:
        with open(json_path, 'r') as json_f:
            d = json.load(json_f)
        return self.load_hps(d)

    def load_hps(self, d: Dict[str, Any]) -> Dict[str, Any]:
        self.cnn.load_hps(d)
        self.encoder_rnn.load_hps(d)
        self._hps = d
        return d

    def define_io_shapes(
            self,
            input_feature_num: int,
            output_feature_num: int,
            input_max_len: int,
            output_max_len: int,
    ) -> None:
        self.input_feature_num = input_feature_num
        self.output_feature_num = output_feature_num
        self.input_max_len = input_max_len
        self.output_max_len = output_max_len
        return

    def write_config(self, d: Dict[str, Any]) -> None:
        if self._workspace:
            with open(os.path.join(self._workspace, 'config.json'), 'w') as f:
                json.dump(d, f, indent=2)
        return

    def __write_hp_config(self) -> None:
        if self._workspace:
            with open(
                os.path.join(self._workspace, 'hp_config.json'),
                'w',
            ) as f:
                json.dump(self.hps, f, indent=2)
        return

    def construct(
            self,
    ) -> None:
        """
        Constructing the network
        """
        self.__write_hp_config()

        ##################
        # Place Holder
        ##################

        # has shape of (timesteps, batch, feature) : cf) time_major
        self._ph_signals = tf.placeholder(
            tf.float32,
            [None, self.input_max_len, self.input_feature_num],
            name='input',)
        self._ph_seqs = tf.placeholder(
            tf.float32,
            [None, self.output_max_len, self.output_feature_num],
            name='answer',)
        self._ph_seqs_shifted = tf.placeholder(
            tf.float32,
            [None, self.output_max_len, self.output_feature_num],
            name='previous_input',)
        self._ph_signals_length = tf.placeholder(
            tf.int32, [None], name='input_length')
        self._ph_seq_lengths = tf.placeholder(
            tf.int32, [None], name='answer_length')
        self._ph_seq_max_lengths = tf.placeholder(
            tf.int32, [None], name='answer_length')
        self._ph_batch_size = tf.placeholder(
            tf.int32, shape=(), name='batch_size')
        # target weight is used to mask the dynamic output sequence
        self._ph_target_weight = tf.placeholder(
            tf.float32,
            [None, self.output_max_len], name='target_weight')
        self._ph_start_tokens = tf.placeholder(
            tf.int32, [None], name='start_tokens')

        ##################
        # Convolution
        ##################

        self._conv_features = self.cnn.conv(self._ph_signals)
        with tf.variable_scope('CalcLength'):
            self._sequence_length = self._make_sequence_length(
                self._conv_features
            )
        # transpose so that output dimension will be (time, batch, feature)

        ##################
        # Encoder LSTM
        ##################
        with tf.variable_scope('EncodingRNN'):
            self._encoder_outputs = self.encoder_rnn.rnn(
                input_tensor=self._conv_features,
                ph_batch_size=self._ph_batch_size,
                sequence_length=self._sequence_length,
            )
        return

    def set_restore_ckpt_dir(
        self,
        ckpt_dir: str
    ) -> None:
        self._is_restore = True
        self._restore_ckpt_dir_path = ckpt_dir
        return


def main() -> None:
    return


if __name__ == '__main__':
    main()
