import math
from typing import Dict  # noqa
from typing import Any, List

import numpy as np
import tensorflow as tf
from halcyon.ml.net.base import Net
from halcyon.ml.net.decoder.inference_decoder import InferenceDecoder
from halcyon.models.seq import Nuc
from logzero import logger


class InferredSeqOneBeam(object):

    HOMOPOLYMER_LENGTH_THRESHOLD = 20
    ALIGNMENT_BREAK_THRESHOLD = 0.3
    ALIGNMENT_CORRECT_THRESHOLD = 0.9
    ALIGNMENT_NON_ATTENTION_THRESHOLD = 1e-5

    @property
    def seq(self) -> str:
        return self._seq

    def update_seq(self, seq: str) -> None:
        self._seq = seq
        return

    @property
    def scores(self) -> List[float]:
        return self._scores

    @property
    def alignment_matrix(self) -> List[List[float]]:
        return self._alignment_matrix

    @property
    def attention_sums(self) -> List[float]:
        return self._attention_sums

    @property
    def logits(self) -> List[float]:
        return self._logits

    def __init__(
            self,
            seq: str,
            scores: List[float],
            alignment_matrix: List[List[float]],
            attention_sums: List[float],
    ) -> None:
        self._seq = seq
        self._scores = scores  # type: List[float]
        self._alignment_matrix = alignment_matrix
        self._attention_sums = attention_sums
        self.__calc_logits()
        return

    def __calc_logits(self) -> None:
        self._logits = []  # type: List[float]
        for i, cumulative_score in enumerate(self._scores):
            cumulative_score_privious = 0. if i == 0 else self._scores[i - 1]
            score_now = cumulative_score - cumulative_score_privious
            try:
                logit = math.pow(math.e, score_now)
            except Exception:
                logit = 0.
            self._logits.append(logit)
        return

    def trim_end_token(self) -> int:
        """
        seqから$ を削除し、scoresやlogitsの長さも合わせる。
        """
        end_token_index = self._seq.find('$')
        if end_token_index < 0:
            return 0
        self._seq = self._seq[:end_token_index]
        self._scores = self._scores[:end_token_index]
        self._logits = self._logits[:end_token_index]
        return 1

    def __get_homopolytail_len(self) -> int:
        last_nuc = self.seq[-1]
        count = 0
        for i in range(len(self.seq) - 1, -1, -1):
            if self.seq[i] == last_nuc:
                count += 1
            else:
                break
        return count

    def remove_homopolytail(self) -> int:
        """
        末尾についているあからさまなhomopolymerを削除する
        """
        if len(self.seq) == 0:
            return 0
        homopoly_len = self.__get_homopolytail_len()
        if homopoly_len < self.HOMOPOLYMER_LENGTH_THRESHOLD:
            return 0
        logger.debug('homopoly len = {}'.format(homopoly_len))
        self._seq = self._seq[:-homopoly_len]
        self._scores = self._scores[:-homopoly_len]
        self._logits = self._logits[:-homopoly_len]
        return 1

    def _get_alignment_break_index(
            self,
            attention_normal_threshold: float,
    ) -> int:
        alignment_break_threshold = attention_normal_threshold / 10
        correct_alignment_flag = False
        for i, attention_sum in enumerate(self.attention_sums):
            if correct_alignment_flag:
                if attention_sum < alignment_break_threshold:
                    return i
                else:
                    continue
            elif attention_sum > attention_normal_threshold:
                correct_alignment_flag = True
                continue
        return -1

    def _get_attention_sum_quantile(self) -> float:
        _, q25 = np.percentile(
            [
                attention_sum for attention_sum in self.attention_sums
                if attention_sum > self.ALIGNMENT_NON_ATTENTION_THRESHOLD
            ],
            [75, 25],
        )
        return q25  # type: ignore

    def trim_by_alignment(self) -> int:
        """
        attention alignmentの結果から、末尾のattentionを払っていない
        塩基配列を削除する。
        塩基あたりのsignalが少ないときにこうなりがち。
        """
        if len(self.attention_sums) == 0:
            logger.debug('alignment is ignore')
            return 0
        quantile = self._get_attention_sum_quantile()
        logger.debug('alignment quantile = {}'.format(quantile))  # noqa
        alignment_break_index = self._get_alignment_break_index(quantile)
        logger.debug('alignment break index = {}'.format(alignment_break_index))  # noqa
        if alignment_break_index < 0:
            return 0
        self._seq = self._seq[:alignment_break_index]
        self._scores = self._scores[:alignment_break_index]
        self._logits = self._logits[:alignment_break_index]
        return 1


class NetInference(Net):
    """ E(encoder)D(decoder)A(attention)
    Encoder-Decoder model with attention mechanisms.

    Attributes:

    """

    START_TOKEN_ID = 0

    @property
    def decoder_rnn(self) -> InferenceDecoder:
        return self._decoder_rnn

    @decoder_rnn.setter
    def decoder_rnn(self, rnn: InferenceDecoder) -> None:
        self._decoder_rnn = rnn
        return

    def get_hp_dict_scheme(self) -> Dict[str, Any]:
        d = super(NetInference, self).get_hp_dict_scheme()
        d[self.decoder_rnn.tag] = {}
        for name in self.decoder_rnn.get_hp_names():
            d[self.decoder_rnn.tag][name] = ''
        return d

    def load_hps(self, d: Dict[str, Any]) -> Dict[str, Any]:
        d = super(NetInference, self).load_hps(d)
        self.decoder_rnn.load_hps(d)
        self._hps = d
        return d

    def construct(
        self,
    ) -> None:
        super(NetInference, self).construct()
        with tf.variable_scope('DecodingRNN'):
            (
                self.decoder_outputs,
                self.final_context_state,
            ) = self.decoder_rnn.rnn(
                input_tensor=self._encoder_outputs,
                output_feature_num=self.output_feature_num,
                output_max_len=self.output_max_len,
                ph_batch_size=self._ph_batch_size,
                ph_seqs_shifted=self._ph_seqs_shifted,
                ph_start_tokens=self._ph_start_tokens,
                ph_seq_max_lengths=self._ph_seq_max_lengths,
                sequence_length=self._sequence_length,
            )
        return

    def infer(
        self,
        sess: tf.Session,
        xs: np.ndarray,
        x_lengths: np.ndarray,
        minibatch_size: int,
    ) -> List[List[InferredSeqOneBeam]]:
        """
        Infer new data
        """
        final_decoder_output, alignment = sess.run(
            [
                self.decoder_outputs,
                self.final_context_state.cell_state.alignment_history,
            ],
            feed_dict={
                self._ph_signals: xs,
                self._ph_signals_length: x_lengths,
                self._ph_seq_max_lengths: [
                    x_length for x_length in x_lengths
                ],
                self._ph_batch_size: minibatch_size,
                self._ph_start_tokens:
                [self.START_TOKEN_ID] * minibatch_size,
            },
        )
        predicted_ids = final_decoder_output.predicted_ids
        decoder_output = final_decoder_output.beam_search_decoder_output
        scores = decoder_output.scores
        logger.debug('scores.shape: {}'.format(scores.shape))
        if not self.decoder_rnn.ignore_alignment_history:
            logger.debug('alignment.shape: {}'.format(alignment.shape))
            alignment_transposed = alignment.transpose([1, 0, 2])
        nuc_arr = np.array(Nuc.NUC_ARR)
        beam_width = predicted_ids.shape[2]
        res = []
        for minibatch_index, batch_beam_outputs in enumerate(predicted_ids):
            inferred_seqs_beam = []  # type: List[InferredSeqOneBeam]
            for beam_index in range(beam_width):
                _seq = ''.join(nuc_arr[predicted_ids[
                    minibatch_index,
                    :,
                    beam_index,
                ]])
                _scores = scores[
                    minibatch_index,
                    :,
                    beam_index,
                ]

                if self.decoder_rnn.keep_full_alignment:
                    _alignment_matrix = alignment_transposed[
                        minibatch_index * beam_width + beam_index,
                        :,
                        :,
                    ].tolist()
                else:
                    _alignment_matrix = []
                if not self.decoder_rnn.ignore_alignment_history:
                    _attention_sums = np.sum(
                        alignment_transposed[
                            minibatch_index * beam_width + beam_index,
                            :,
                            :,
                        ],
                        axis=1,
                    ).tolist()
                else:
                    _attention_sums = []

                inferred_seqs_beam.append(
                    InferredSeqOneBeam(
                        seq=_seq,
                        scores=_scores.tolist(),
                        alignment_matrix=_alignment_matrix,
                        attention_sums=_attention_sums,
                    ))
            res.append(inferred_seqs_beam)
        return res
