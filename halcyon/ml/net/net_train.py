import json
import os
from typing import Dict  # noqa
from typing import Any, Tuple

import numpy as np
import tensorflow as tf
from halcyon.ml.net.base import Net
from halcyon.ml.net.decoder.train_decoder import TrainDecoder


class NetTrain(Net):
    """ E(encoder)D(decoder)A(attention)
    Encoder-Decoder model with attention mechanisms.

    Attributes:

    """

    RUNTIME_PARAMS = Net.RUNTIME_PARAMS + [
        'optimizer',
        'label_smoothing_rate',
    ]

    OPTIMIZER_DICT = {
        'adam': tf.train.AdamOptimizer,
        'rms_prop': tf.train.RMSPropOptimizer,
        'sgd': tf.train.GradientDescentOptimizer,
        'ada_grad': tf.train.AdagradOptimizer,
        'ada_delta': tf.train.AdadeltaOptimizer,
    }

    @property
    def optimizer(self) -> Any:
        return self.OPTIMIZER_DICT[self._rt_params['optimizer']]

    @property
    def decoder_rnn(self) -> TrainDecoder:
        return self._decoder_rnn

    @decoder_rnn.setter
    def decoder_rnn(self, rnn: TrainDecoder) -> None:
        self._decoder_rnn = rnn
        return

    def get_hp_dict_scheme(self) -> Dict[str, Any]:
        d = super(NetTrain, self).get_hp_dict_scheme()
        d[self.decoder_rnn.tag] = {}
        for name in self.decoder_rnn.get_hp_names():
            d[self.decoder_rnn.tag][name] = ''
        return d

    def load_hps(self, d: Dict[str, Any]) -> Dict[str, Any]:
        d = super(NetTrain, self).load_hps(d)
        self.decoder_rnn.load_hps(d)
        self._hps = d
        return d

    def construct(
            self,
    ) -> None:
        super(NetTrain, self).construct()
        """
        This step is different from global step
        in that this is not restore from checkpoint.
        This is used for learning rate decay.
        """
        ##################
        # Decoder LSTM
        ##################
        with tf.variable_scope('DecodingRNN'):
            self._logits = self.decoder_rnn.rnn(
                input_tensor=self._encoder_outputs,
                output_feature_num=self.output_feature_num,
                output_max_len=self.output_max_len,
                ph_batch_size=self._ph_batch_size,
                ph_seqs_shifted=self._ph_seqs_shifted,
                ph_start_tokens=self._ph_start_tokens,
                ph_seq_max_lengths=self._ph_seq_max_lengths,
                sequence_length=self._sequence_length,
            )

        with tf.variable_scope('Optimization'):
            self._crossent = tf.losses.softmax_cross_entropy(
                onehot_labels=self._ph_seqs,
                logits=self._logits,
                label_smoothing=self.rt_params['label_smoothing_rate'],
            )
            self._loss = tf.reduce_sum(
                self._crossent * self._ph_target_weight
            ) / tf.cast(
                self._ph_batch_size,
                tf.float32
            )  # type: tf.Variale
            ys = tf.argmax(self._ph_seqs, 2)
            ys_hat = tf.argmax(self._logits, 2)
            correct_prediction = tf.equal(ys, ys_hat)
            equals_sum = tf.reduce_sum(
                tf.cast(
                    correct_prediction,
                    tf.float32,
                ) * self._ph_target_weight
            )
            non_mask_weight_sum = tf.reduce_sum(
                tf.cast(self._ph_target_weight, tf.float32))
            self._acc_per_base = (
                equals_sum / non_mask_weight_sum
            )  # type: tf.Variale
            self._loss_per_base = (
                self._loss / non_mask_weight_sum *
                tf.cast(
                    self._ph_batch_size,
                    tf.float32
                )
            )  # type: tf.Variale

            # Calculate and clip gradients

            params = tf.trainable_variables()
            self._gradients = tf.gradients(self._loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(
                self._gradients,
                self.clip_norm,
            )
            # Global step (restored)
            self._global_step = tf.Variable(
                0,
                trainable=False,
                name='global_step',
            )
            self._ph_step = tf.placeholder(
                tf.int32,
                (),
                name='step',
            )
            self._learning_rate = tf.train.exponential_decay(
                learning_rate=self.init_learning_rate,
                global_step=self._ph_step,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=True,
            )  # type: tf.Tensor
            optimizer = self.optimizer(self._learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self._update_step = optimizer.apply_gradients(
                    zip(clipped_gradients, params),
                    global_step=self._global_step,
                )
        return

    def prepare_tensor_board(self, sess: tf.Session) -> None:
        """ Set writer
        """
        loss_summary = tf.summary.scalar(
            'summary/loss',
            self._loss,
        )
        acc_summary = tf.summary.scalar(
            'summary/accuracy',
            self._acc_per_base,
        )
        self._summary = tf.summary.merge(
            [loss_summary, acc_summary],
            name='summary',
        )

        learning_rate_summary = tf.summary.scalar(
            'params/learning_rate',
            self._learning_rate,
        )
        self._params_summary = tf.summary.merge(
            [learning_rate_summary],
            name='summary',
        )
        self._writer = tf.summary.FileWriter(
            os.path.join(self._workspace, 'tb_log', 'general'),
            sess.graph,
        )
        self._train_writer = tf.summary.FileWriter(
            os.path.join(self._workspace, 'tb_log', 'train'),
            sess.graph,
        )
        self._validation_writer = tf.summary.FileWriter(
            os.path.join(self._workspace, 'tb_log', 'validation'),
            sess.graph,
        )

        params_dict = {}  # type: Dict[str, Any]
        params_dict.update(self.hps)
        params_dict.update({
            'runtime': self.rt_params,
        })
        markdown_str = '|key|value|\n\t|---|---|\n\t'
        for k, v in params_dict.items():
            markdown_str += '|{}|{}|\n\t'.format(
                k,
                v,
            )
        text_summary = tf.summary.text(
            "net_config/all",
            tf.convert_to_tensor(markdown_str),
        )
        self._text_summary = tf.summary.merge(
            [text_summary],
            name='summary',
        )
        summary = sess.run(self._text_summary)
        self._writer.add_summary(summary)
        self._writer.flush()
        return

    def train_minibatch(
            self,
            sess: tf.Session,
            xs: np.ndarray,
            ys: np.ndarray,
            ys_shifted: np.ndarray,
            x_lengths: np.ndarray,
            y_lengths: np.ndarray,
            minibatch_size: int,
            step: int,
            write_summary: bool = True,
            verbose: bool = True,
    ) -> Tuple[float, float]:
        """ Train the network with one minibatch.
        """
        if verbose:
            self.logger.info(
                'train minibatch with minibatch_size = {}'.format(
                    minibatch_size,
                )
            )
        target_weight = self._make_target_weight(ys, y_lengths)
        global_step = tf.train.global_step(sess, self._global_step)
        res_tuple = sess.run([
            self._summary,
            self._params_summary,
            self._update_step,
            self._loss,
            self._loss_per_base,
            self._acc_per_base,
        ],
            feed_dict={
                self._ph_signals: xs,
                self._ph_seqs: ys,
                self._ph_seqs_shifted: ys_shifted,
                self._ph_signals_length: x_lengths,
                self._ph_seq_lengths: y_lengths,
                self._ph_seq_max_lengths: ys.shape[1] * np.ones((ys.shape[0])),
                self._ph_batch_size: minibatch_size,
                self._ph_step: step,
                #  self._ph_learning_rate: learning_rate,
                self._ph_target_weight: target_weight,
        },
        )
        (
            summary,
            params_summary,
            _,
            loss_value,
            loss_per_base_value,
            acc_per_base,
        ) = res_tuple
        if verbose:
            fmt_str = '[{} Training] Step: {}, Loss: {:6.3f}, Loss per bsae: {:6.3f}, Acc: {:6.3f}'  # noqa
            self.logger.info(fmt_str.format(
                self.id,
                global_step,
                loss_value,
                loss_per_base_value,
                acc_per_base,
            ))
        self._writer.add_summary(params_summary, global_step)
        if write_summary:
            self._train_writer.add_summary(summary, global_step)
        return float(acc_per_base), float(loss_value)

    def validate(
            self,
            sess: tf.Session,
            xs: np.ndarray,
            ys: np.ndarray,
            ys_shifted: np.ndarray,
            x_lengths: np.ndarray,
            y_lengths: np.ndarray,
            minibatch_size: int,
    ) -> Tuple[float, float]:
        """ Validate the model against the input data, and make checkpoint-snapshot.
        """
        global_step = tf.train.global_step(sess, self._global_step)
        step_ckpt_dir = os.path.join(self._workspace, str(global_step))
        os.makedirs(step_ckpt_dir, exist_ok=True)
        target_weight = self._make_target_weight(ys, y_lengths)
        res_tuple = sess.run([
            self._summary,
            self._logits,
            self._loss,
            self._loss_per_base,
            self._acc_per_base,
        ],
            feed_dict={
                self._ph_signals: xs,
                self._ph_seqs: ys,
                self._ph_seqs_shifted: ys_shifted,
                self._ph_signals_length: x_lengths,
                self._ph_seq_lengths: y_lengths,
                self._ph_seq_max_lengths: ys.shape[1] * np.ones((ys.shape[0])),
                self._ph_batch_size: minibatch_size,
                self._ph_target_weight: target_weight,
        },
        )
        (
            summary,
            logits_value,
            loss_value,
            loss_per_base_value,
            acc_per_base,
        ) = res_tuple

        with open(os.path.join(step_ckpt_dir, 'validation.txt'), 'w') as f:
            json.dump({
                'loss': float(loss_value),
            },
                f,
                indent=2,
            )
        np.save(os.path.join(step_ckpt_dir, 'logits.npy'), logits_value)
        fmt_str = '[{} Validation] Step: {}, Loss: {:6.3f}, Loss per bsae: {:6.3f}, Acc: {:6.3f}'  # noqa
        self.logger.info(fmt_str.format(
            self.id,
            global_step,
            loss_value,
            loss_per_base_value,
            acc_per_base,
        ))
        # checkpoint save
        self._saver.save(
            sess,
            loss_value,
            step_ckpt_dir,
        )
        self._validation_writer.add_summary(summary, global_step)
        return float(acc_per_base), float(loss_value)
