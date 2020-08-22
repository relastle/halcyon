from abc import ABCMeta, abstractclassmethod
from typing import Any, List

import tensorflow as tf
from halcyon.ml.net.sub_net import SubNet


class CnnBase(SubNet, metaclass=ABCMeta):
    """
    Convolutional Neural Network Module
    """

    def __init__(
            self,
            is_inference: bool,
    ) -> None:
        self.is_inference = is_inference
        return

    def get_tag(self) -> str:
        return 'cnn'

    @property
    def activation_function(self) -> Any:
        if self['activation_function'] == 'relu':
            return tf.nn.relu
        elif self['activation_function'] == 'leaky_relu':
            return tf.nn.leaky_relu

    @abstractclassmethod
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        raise NotImplementedError
        return None


class CNN(CnnBase):
    """
    Very normal 1d-Convolution module
    """

    HP_NAMES = [
        'depth',
        'num_kernel_lst',
        'kernel_size_lst',
        'kernel_stride_lst',
        'pooling_index_lst',
        'pooling_size_lst',
        'pooling_stride_lst',
        'use_bn',
        'activation_function',
    ]

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

    # override
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        self._conv_features_lst = []  # type: List[tf.Tensor]
        self._conv_features_lst.append(input_tensor)  # pseudo append

        for conv_i in range(self['depth']):
            tmp_conv_features = tf.layers.conv1d(
                inputs=self._conv_features_lst[conv_i],
                filters=self['num_kernel_lst'][conv_i],
                kernel_size=self['kernel_size_lst'][conv_i],
                strides=self['kernel_stride_lst'][conv_i],
            )
            if self['use_bn']:
                tmp_conv_features = tf.layers.batch_normalization(
                    tmp_conv_features,
                    training=not self.is_inference,
                )
            tmp_conv_features = self.activation_function(tmp_conv_features)

            if conv_i in self['pooling_index_lst']:
                pool_index = self['pooling_index_lst'].index(conv_i)
                pool_size = self['pooling_size_lst'][pool_index]
                stride = self['pooling_stride_lst'][pool_index]
                tmp_conv_features = tf.layers.max_pooling1d(
                    tmp_conv_features,
                    pool_size=pool_size,
                    strides=stride,
                    padding='valid',
                    data_format='channels_last',
                    name=None
                )
                self._conv_features_lst.append(tmp_conv_features)
            else:
                self._conv_features_lst.append(tmp_conv_features)
        return self._conv_features_lst[-1]


class InceptionBlocks(CnnBase):
    """
    Very normal 1d-Convolution module
    """
    BRANCH_REDUCED_CHANNEL_NUM = 64

    BRANCH_1x1_CHANNEL_NUM = 64
    BRANCH_3x3_CHANNEL_NUM = 96
    BRANCH_5x5_CHANNEL_NUM = 128
    BRANCH_7x7_CHANNEL_NUM = 160
    BRANCH_POOL_CHANNEL_NUM = 32

    CHANNEL_NUMS = [
        BRANCH_1x1_CHANNEL_NUM,
        BRANCH_3x3_CHANNEL_NUM,
        BRANCH_5x5_CHANNEL_NUM,
        BRANCH_7x7_CHANNEL_NUM,
        BRANCH_POOL_CHANNEL_NUM,
    ]

    def __init__(
            self,
            is_inference: bool,
    ) -> None:
        self.is_inference = is_inference
        return

    HP_NAMES = [
        'depth',
        'pooling_index_lst',
        'pooling_size_lst',
        'pooling_stride_lst',
        'channel_num_increase_ratio',
        'use_bn',
        'activation_function',
    ]

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

    def _conv1d_bn(
            self,
            input_tensor: tf.Tensor,
            filters: int,
            width: int,
            stride: int,
            padding: str,
    ) -> tf.Tensor:
        x = tf.layers.conv1d(
            inputs=input_tensor,
            filters=filters,
            kernel_size=width,
            strides=stride,
            padding=padding,
        )
        x = self.activation_function(x)
        if self['use_bn']:
            x = tf.contrib.layers.batch_norm(
                x,
                is_training=not self.is_inference,
            )
        return x

    def _inception_block_channel_num(
            self,
            block_index: int,
    ) -> int:
        num = 0
        for channel_num in self.CHANNEL_NUMS:
            mag = self['channel_num_increase_ratio'] ** (block_index)
            num += int(
                channel_num * mag
            )
        return num

    def _inception_block(
            self,
            input_tensor: tf.Tensor,
            block_index: int,
    ) -> tf.Tensor:
        mag = self['channel_num_increase_ratio'] ** (block_index)
        with tf.variable_scope('branch_1x1'):
            branch1x1 = self._conv1d_bn(
                input_tensor,
                int(self.BRANCH_1x1_CHANNEL_NUM * mag), 1, 1,
                'same',
            )
        with tf.variable_scope('branch_3x3'):
            branch3x3 = self._conv1d_bn(
                input_tensor,
                int(self.BRANCH_REDUCED_CHANNEL_NUM), 1, 1,
                'same',
            )
            branch3x3 = self._conv1d_bn(
                branch3x3,
                int(self.BRANCH_3x3_CHANNEL_NUM * mag), 3, 1,
                'same',
            )

        with tf.variable_scope('branch_5x5'):
            branch5x5 = self._conv1d_bn(
                input_tensor,
                self.BRANCH_REDUCED_CHANNEL_NUM, 1, 1,
                'same',
            )
            branch5x5 = self._conv1d_bn(
                branch5x5,
                int(self.BRANCH_5x5_CHANNEL_NUM * mag), 5, 1,
                'same',
            )

        with tf.variable_scope('branch_7x7'):
            branch7x7 = self._conv1d_bn(
                input_tensor,
                self.BRANCH_REDUCED_CHANNEL_NUM, 1, 1,
                'same',
            )
            branch7x7 = self._conv1d_bn(
                branch7x7,
                int(self.BRANCH_7x7_CHANNEL_NUM * mag), 7, 1,
                'same',
            )

        with tf.variable_scope('branch_pool'):
            branch_pool = tf.layers.average_pooling1d(
                input_tensor,
                pool_size=3,
                strides=1,
                padding='same',
                data_format='channels_last',
                name=None
            )
            branch_pool = self._conv1d_bn(
                branch_pool,
                int(self.BRANCH_POOL_CHANNEL_NUM * mag), 1, 1,
                'same',
            )

        concatenated_conv_features = tf.concat(
            values=[
                branch1x1,
                branch3x3,
                branch5x5,
                branch7x7,
                branch_pool,
            ],
            axis=2,
        )
        return concatenated_conv_features

    # override
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        with tf.variable_scope('conv_1'):
            x = self._conv1d_bn(
                input_tensor,
                self._inception_block_channel_num(0), 3, 1,
                'same',
            )
        for conv_i in range(self['depth']):
            with tf.variable_scope('inception_block_{}'.format(conv_i)):
                # inception
                x = self._inception_block(
                    x,
                    conv_i,
                )
            if conv_i in self['pooling_index_lst']:
                pooling_index = self['pooling_index_lst'].index(conv_i)
                with tf.variable_scope('pool_{}'.format(pooling_index)):
                    x = tf.layers.max_pooling1d(
                        x,
                        pool_size=self['pooling_size_lst'][pooling_index],
                        strides=self['pooling_stride_lst'][pooling_index],
                        padding='same',
                        data_format='channels_last',
                        name=None
                    )
        return x


class InceptionResBlocks(InceptionBlocks):
    """
    It has recurrent residual connection like
    Inception v4.(https://arxiv.org/pdf/1712.09888.pdf)
    """

    HP_NAMES = [
        'depth',
        'pooling_index_lst',
        'pooling_size_lst',
        'pooling_stride_lst',
        'channel_num_increase_ratio',
        'use_bn',
    ]

    # override
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        x = input_tensor
        for conv_i in range(self['depth']):
            with tf.variable_scope('conv_for_reshape_{}'.format(conv_i)):
                reshaped = self._conv1d_bn(
                    x,
                    self._inception_block_channel_num(conv_i), 1, 1,
                    'same',
                )
            with tf.variable_scope('inception_block_{}'.format(conv_i)):
                # inception
                inception = self._inception_block(
                    reshaped,
                    conv_i,
                )
            # reshape for dimension adjustment
            x = reshaped + inception

            if conv_i in self['pooling_index_lst']:
                pooling_index = self['pooling_index_lst'].index(conv_i)
                with tf.variable_scope('pool_{}'.format(pooling_index)):
                    x = tf.layers.max_pooling1d(
                        x,
                        pool_size=self['pooling_size_lst'][pooling_index],
                        strides=self['pooling_stride_lst'][pooling_index],
                        padding='same',
                        data_format='channels_last',
                        name=None
                    )
        return x


class Inception(InceptionBlocks):
    """
    """

    HP_NAMES = [
        'use_bn',
        'depth',
        'channel_num_increase_ratio',
        'base_cnn_filter_multiplier',
        '1st_conv_kernel_size',
    ]

    # override
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        x = input_tensor

        with tf.variable_scope('conv_block_1'):
            with tf.variable_scope('conv_1'):
                x = self._conv1d_bn(
                    x,
                    32 * self['base_cnn_filter_multiplier'],
                    self['1st_conv_kernel_size'], 2,
                    'valid',
                )
            with tf.variable_scope('conv_2'):
                x = self._conv1d_bn(
                    x,
                    32 * self['base_cnn_filter_multiplier'],
                    3, 1,
                    'valid',
                )
            with tf.variable_scope('conv_3'):
                x = self._conv1d_bn(
                    x,
                    64 * self['base_cnn_filter_multiplier'],
                    3, 1,
                    'valid',
                )
            with tf.variable_scope('pool_1'):
                x = tf.layers.max_pooling1d(
                    x,
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    data_format='channels_last',
                    name=None
                )
        with tf.variable_scope('conv_block_2'):
            with tf.variable_scope('conv_4'):
                x = self._conv1d_bn(
                    x,
                    80 * self['base_cnn_filter_multiplier'],
                    1, 1,
                    'valid',
                )
            with tf.variable_scope('conv_5'):
                x = self._conv1d_bn(
                    x,
                    192 * self['base_cnn_filter_multiplier'],
                    3, 1,
                    'valid',
                )
            with tf.variable_scope('pool_2'):
                x = tf.layers.max_pooling1d(
                    x,
                    pool_size=3,
                    strides=2,
                    padding='valid',
                    data_format='channels_last',
                    name=None
                )
        with tf.variable_scope('inception_blocks_1'):
            for inception_i in range(self['depth']):
                with tf.variable_scope(
                        'inception_block_{}'.format(inception_i + 1)
                ):
                    # inception
                    x = self._inception_block(
                        x,
                        inception_i,
                    )
        return x


class InceptionNeo(Inception):
    """
    Inceptionをさらに改良する目的(2019-04-22)
    """

    HP_NAMES = [
        'use_bn',
        'depth',
        'base_cnn_filter_multiplier',
        'channel_num_increase_ratio',
    ]

    # override
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        x = input_tensor

        with tf.variable_scope('inception_upstream_1'):
            x = self._inception_block(
                x,
                1,
            )

        with tf.variable_scope('conv_block_1'):
            with tf.variable_scope('conv_1'):
                x = self._conv1d_bn(
                    x,
                    80 * self['base_cnn_filter_multiplier'],
                    1, 1,
                    'valid',
                )
            with tf.variable_scope('conv_2'):
                x = self._conv1d_bn(
                    x,
                    192 * self['base_cnn_filter_multiplier'],
                    3, 1,
                    'valid',
                )
            with tf.variable_scope('pool_1'):
                x = tf.layers.max_pooling1d(
                    x,
                    pool_size=3,
                    strides=3,
                    padding='valid',
                    data_format='channels_last',
                    name=None
                )

        with tf.variable_scope('inception_upstream_2'):
            x = self._inception_block(
                x,
                1,
            )

        with tf.variable_scope('conv_block_2'):
            with tf.variable_scope('conv_3'):
                x = self._conv1d_bn(
                    x,
                    80 * self['base_cnn_filter_multiplier'],
                    1, 1,
                    'valid',
                )
            with tf.variable_scope('conv_4'):
                x = self._conv1d_bn(
                    x,
                    192 * self['base_cnn_filter_multiplier'],
                    3, 1,
                    'valid',
                )
            with tf.variable_scope('pool_2'):
                x = tf.layers.max_pooling1d(
                    x,
                    pool_size=3,
                    strides=3,
                    padding='valid',
                    data_format='channels_last',
                    name=None
                )

        with tf.variable_scope('inception_blocks_1'):
            for inception_i in range(self['depth']):
                with tf.variable_scope(
                        'inception_block_{}'.format(inception_i + 1)
                ):
                    # inception
                    _x = self._inception_block(
                        x,
                        1,
                    )
                    if inception_i != 0:
                        x = x + _x
                    else:
                        x = _x
        return x


class GatedLinearBlocks(CnnBase):
    """
    これはCNNそうとしてのGatedLinearBlocks.
    もしencoderとしても使いたいなら、そちらを。
    """

    HP_NAMES = [
        'use_bn',
        'use_residual_connection',
    ]

    def get_hp_names(self) -> List[str]:
        return self.HP_NAMES

    def gated_linear_block(
            self,
            input_tensor: tf.Tensor,
            filters: int,
            kernel_size: int,
            strides: int,
            padding: str,
            use_bn: bool = True,
    ) -> tf.Tensor:
        with tf.variable_scope('main_stream'):
            A = tf.layers.conv1d(
                inputs=input_tensor,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
            )
            if use_bn:
                A = tf.contrib.layers.batch_norm(
                    A,
                    is_training=not self.is_inference,
                )
            A = tf.nn.relu(A)

        with tf.variable_scope('gating_stream'):
            B = tf.layers.conv1d(
                inputs=input_tensor,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
            )
            if use_bn:
                B = tf.contrib.layers.batch_norm(
                    B,
                    is_training=not self.is_inference,
                )
            B = tf.nn.sigmoid(B)

        x = tf.multiply(A, B)
        return x

    # override
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        x = input_tensor
        with tf.variable_scope('GLB_main_1'):
            x = self.gated_linear_block(
                x,
                filters=256,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bn=self['use_bn'],
            )
        with tf.variable_scope('Conv1'):
            x = self.gated_linear_block(
                x,
                filters=256,
                kernel_size=3,
                strides=3,
                padding='VALID',
                use_bn=self['use_bn'],
            )
        with tf.variable_scope('GLB_main_2'):
            x = self.gated_linear_block(
                x,
                filters=512,
                kernel_size=3,
                strides=1,
                padding='SAME',
                use_bn=self['use_bn'],
            )
        with tf.variable_scope('Conv2'):
            x = self.gated_linear_block(
                x,
                filters=512,
                kernel_size=3,
                strides=3,
                padding='VALID',
                use_bn=self['use_bn'],
            )
        return x


class HalcyonCNN(Inception):
    """
    """

    HP_NAMES = [
        'use_bn',
        'depth',
        'channel_num_increase_ratio',
    ]

    # override
    def conv(
            self,
            input_tensor: tf.Tensor,
    ) -> tf.Tensor:
        x = input_tensor

        with tf.variable_scope('conv_1'):
            x = self._conv1d_bn(
                x,
                64, 3, 1,
                'valid',
            )
        with tf.variable_scope('inception_blocks_1'):
            for inception_i in range(self['depth']):
                with tf.variable_scope(
                        'inception_block_{}'.format(inception_i)
                ):
                    # inception
                    x = self._inception_block(
                        x,
                        inception_i,
                    )
        with tf.variable_scope('pool_1'):
            x = tf.layers.max_pooling1d(
                x,
                pool_size=3,
                strides=3,
                padding='valid',
                data_format='channels_last',
                name=None
            )
        with tf.variable_scope('inception_blocks_2'):
            for inception_i in range(self['depth']):
                with tf.variable_scope(
                        'inception_block_{}'.format(inception_i)
                ):
                    # inception
                    x = self._inception_block(
                        x,
                        inception_i + self['depth'],
                    )
        with tf.variable_scope('pool_2'):
            x = tf.layers.max_pooling1d(
                x,
                pool_size=3,
                strides=3,
                padding='valid',
                data_format='channels_last',
                name=None
            )
        return x
