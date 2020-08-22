import json
import logging
import time
from abc import ABCMeta, abstractmethod
from itertools import chain
from os.path import dirname
from typing import Any, Dict, List, Type

import logzero
import numpy as np
import tensorflow as tf
from Bio import pairwise2
from halcyon.config import Const
from halcyon.ml.net.cnn import (CNN, CnnBase, GatedLinearBlocks, HalcyonCNN,
                                Inception, InceptionBlocks, InceptionNeo,
                                InceptionResBlocks)
from halcyon.ml.net.decoder.inference_decoder import (
    InferenceDecoder, InferenceGruBlockDecoder, InferenceGruDecoder,
    InferenceLstmBlockDecoder, InferenceLstmDecoder, InferenceSruDecoder)
from halcyon.ml.net.encoder import (BidirectionalCudnnLSTMCell,
                                    BidirectionalFusedEncoder,
                                    BidirectionalGruBlockEncoder,
                                    BidirectionalGruEncoder,
                                    BidirectionalIndyLstmEncoder,
                                    BidirectionalLstmBlockEncoder,
                                    BidirectionalLstmEncoder,
                                    BidirectionalLstmResidualEncoder,
                                    BidirectionalNasEncoder,
                                    BidirectionalSruEncoder,
                                    BidirectionalUgrnnEncoder, Encoder,
                                    GatedLinearBlocksEncoder,
                                    UnidirectionalLstmEncoder)
from halcyon.ml.net.net_inference import InferredSeqOneBeam, NetInference
from halcyon.models.fast5 import Fast5
from halcyon.models.seq import Nuc
from logzero import logger
from more_itertools import chunked

CNN_DICT = {
    'cnn': CNN,
    'glb': GatedLinearBlocks,
    'inception_blocks': InceptionBlocks,
    'inception_res_blocks': InceptionResBlocks,
    'inception': Inception,
    'inception_neo': InceptionNeo,
    'halcyon_cnn': HalcyonCNN,
}  # type: Dict[str, Type[CnnBase]]

ENCODER_DICT = {
    'uni_lstm': UnidirectionalLstmEncoder,
    'bi_lstm': BidirectionalLstmEncoder,
    'bi_lstm_block': BidirectionalLstmBlockEncoder,
    'bi_gru_block': BidirectionalGruBlockEncoder,
    'bi_indy_lstm': BidirectionalIndyLstmEncoder,
    'bi_nas': BidirectionalNasEncoder,
    'bi_ugrnn': BidirectionalUgrnnEncoder,
    'bi_fused': BidirectionalFusedEncoder,
    'bi_cudnn': BidirectionalCudnnLSTMCell,
    'bi_lstm_res': BidirectionalLstmResidualEncoder,
    'bi_gru': BidirectionalGruEncoder,
    'bi_sru': BidirectionalSruEncoder,
    'glb': GatedLinearBlocksEncoder,
}  # type: Dict[str, Type[Encoder]]


INFERENCE_DECODER_DICT = {
    'lstm': InferenceLstmDecoder,
    'lstm_block': InferenceLstmBlockDecoder,
    'gru': InferenceGruDecoder,
    'gru_block': InferenceGruBlockDecoder,
    'sru': InferenceSruDecoder,
}  # type: Dict[str, Type[InferenceDecoder]]

EOS_LEN = Const.END_OF_SIGNALS_LENGTH
EOS_VALUE = Const.END_OF_SIGNALS_VALUE
SIGNALS_ENDS_MARGIN = 100
SIGNALS_PER_BASE_OVERLAP = 8
MAX_SIGNALS_PER_BASE = 7


class SeqLogitsPair(object):

    @classmethod
    def align_logits(
            cls,
            seq_gapped: str,
            logits_non_gapped: List[float],
    ) -> List[float]:
        logits_gapped = []  # type: List[float]
        index = 0
        for c in seq_gapped:
            if c == '-':
                logits_gapped.append(-1.)
            else:
                logits_gapped.append(logits_non_gapped[index])
                index += 1
        return logits_gapped

    @property
    def seq(self) -> str:
        return self._seq

    @property
    def logits(self) -> List[float]:
        return self._logits

    def __init__(self, seq: str, logits: List[float]) -> None:
        assert(len(seq) == len(logits))
        self._seq = seq
        self._logits = logits
        return


class InfererOutput(object):

    @property
    def res_d(self) -> Dict[str, SeqLogitsPair]:
        return self._res_d

    @property
    def meta_d(self) -> Dict[str, Any]:
        return self._meta_d

    def __init__(
            self,
            res_d: Dict[str, SeqLogitsPair],
            meta_d: Dict[str, Any],
    ) -> None:
        self._res_d = res_d
        self._meta_d = meta_d
        self.meta_any = None  # type: Any
        return


class Merger(metaclass=ABCMeta):

    def __init__(self) -> None:
        return

    @abstractmethod
    def merge(
            self,
            seq_logits_pair1: SeqLogitsPair,
            seq_logits_pair2: SeqLogitsPair,
    ) -> SeqLogitsPair:
        raise NotImplementedError


class MergerLeftPriority(Merger):

    def __get_end_index(
            self,
            seq: str,
    ) -> int:
        for index in range(len(seq) - 1, -1, -1):
            if seq[index] != '-':
                return index
        raise ValueError

    def merge(
            self,
            seq_logits_pair1: SeqLogitsPair,
            seq_logits_pair2: SeqLogitsPair,
    ) -> SeqLogitsPair:
        seq1 = seq_logits_pair1.seq
        seq2 = seq_logits_pair2.seq
        logits1 = seq_logits_pair1.logits
        logits2 = seq_logits_pair2.logits
        assert(len(seq1) == len(seq2))
        end_index_seq1 = self.__get_end_index(seq1)
        seq_merged_gapped = (
            seq1[:end_index_seq1 + 1] +
            seq2[end_index_seq1 + 1:]
        )
        logits_merged_gapped = (
            logits1[:end_index_seq1 + 1] +
            logits2[end_index_seq1 + 1:]
        )
        assert(len(seq_merged_gapped) == len(logits_merged_gapped))
        seq_merged = seq_merged_gapped.replace('-', '')
        logits_merged = [
            score for score in logits_merged_gapped
            if score > 0
        ]
        return SeqLogitsPair(
            seq=seq_merged,
            logits=logits_merged,
        )


class MergerUseBoth(Merger):

    def merge(
            self,
            seq_logits_pair1: SeqLogitsPair,
            seq_logits_pair2: SeqLogitsPair,
    ) -> SeqLogitsPair:
        seq1 = seq_logits_pair1.seq
        seq2 = seq_logits_pair2.seq
        logits1 = seq_logits_pair1.logits
        logits2 = seq_logits_pair2.logits
        assert(len(seq1) == len(seq2))
        seq_merged = ''
        logits_merged = []
        for c1, c2, score1, score2 in zip(seq1, seq2, logits1, logits2):
            if c1 == '-':
                seq_merged += c2
                logits_merged.append(score2)
            else:
                seq_merged += c1
                logits_merged.append(score1)
        assert(len(seq_merged) == len(seq1))
        return SeqLogitsPair(
            seq=seq_merged,
            logits=logits_merged,
        )


class SignalsSnippet(object):
    """
    Snippet of signals.
    """

    @property
    def signals(self) -> List[float]:
        return self._signals

    @property
    def ID(self) -> str:
        return self._ID

    @property
    def index(self) -> int:
        return self._index

    @property
    def seq(self) -> str:
        return self._inferred_seq.seq

    @property
    def scores(self) -> List[float]:
        return self._inferred_seq.scores

    @property
    def logits(self) -> List[float]:
        return self._inferred_seq.logits

    @property
    def inferred_seq(self) -> InferredSeqOneBeam:
        return self._inferred_seq

    @inferred_seq.setter
    def inferred_seq(self, inferred_seq: InferredSeqOneBeam) -> None:
        self._inferred_seq = inferred_seq
        return

    @property
    def left_merge_len(self) -> int:
        return self._left_merge_len

    @left_merge_len.setter
    def left_merge_len(self, left_merge_len: int) -> None:
        self._left_merge_len = left_merge_len
        return

    @property
    def right_merge_len(self) -> int:
        return self._right_merge_len

    @right_merge_len.setter
    def right_merge_len(self, right_merge_len: int) -> None:
        self._right_merge_len = right_merge_len
        return

    def __init__(
            self,
            signals: List[float],
            ID: str,
            index: int,
    ) -> None:
        self._signals = signals
        self._ID = ID
        self._index = index
        self._inferred_seq = InferredSeqOneBeam(
            seq='',
            scores=[],
            alignment_matrix=[],
            attention_sums=[],
        )
        self._left_merge_len = 0
        self._right_merge_len = 0
        return

    def to_seriarizable(self) -> Dict[str, Any]:
        return {
            'ID': self._ID,
            'index': self._index,
            'signals': self._signals,
            'seq': self._inferred_seq.seq,
            'logits': self._inferred_seq.logits,
            'right_merge_len': self._right_merge_len,
            'left_merge_len': self._left_merge_len,
            'alignment_matrix': self._inferred_seq.alignment_matrix,
            'attention_sums': self._inferred_seq.attention_sums,
        }


class SignalsSnippets(object):

    @classmethod
    def organize(
            cls,
            signals_snippets: List[SignalsSnippet],
    ) -> Dict[str, List[SignalsSnippet]]:
        res_dict = {}  # type: Dict[str, List[SignalsSnippet]]
        for signals_snippet in signals_snippets:
            res_dict.setdefault(signals_snippet.ID, [])
            res_dict[signals_snippet.ID].append(signals_snippet)
        for k, v in res_dict.items():
            res_dict[k] = sorted(v, key=lambda x: x.index)
        return res_dict

    @classmethod
    def merge(
            cls,
            signals_snippets: List[SignalsSnippet],
            overlap_len: int,
            signals_per_base: int,
    ) -> SeqLogitsPair:
        """
        Merge the overlapped sequences using
        pairwise alignemnt.
        """
        MATCH_SCORE = 4.
        MISMATCH_SCORE = -4.5
        GAP_OPEN_SCORE = -5.
        GAP_EXTEND_SCORE = -3.
        MERGER = MergerLeftPriority()
        overlap_seq_len = (
            (overlap_len - SIGNALS_ENDS_MARGIN) // signals_per_base
        )
        seq_merged = signals_snippets[0].seq
        logits_merged = signals_snippets[0].logits
        merge_flag = False  # 一回でもmergeが成功したかどうかを判定する
        for i in range(1, len(signals_snippets)):
            logger.debug(
                'Merger between {}th and {}th snippets'.format(
                    i - 1,
                    i,
                ))
            seq_appended = signals_snippets[i].seq
            logits_appended = signals_snippets[i].logits
            seq1_overlap = seq_merged[-overlap_seq_len:]
            seq2_overlap = seq_appended[:overlap_seq_len]
            logits1_overlap = logits_merged[-overlap_seq_len:]
            logits2_overlap = logits_appended[:overlap_seq_len]
            algns = pairwise2.align.localms(
                seq1_overlap,
                seq2_overlap,
                MATCH_SCORE,
                MISMATCH_SCORE,
                GAP_OPEN_SCORE,
                GAP_EXTEND_SCORE,
            )
            if len(algns) == 0:
                logger.warning(
                    'no alignment was found between {}th and {}th snippets'.format(
                        i - 1,
                        i,
                    ))
                if not merge_flag:
                    seq_merged = signals_snippets[i].seq
                    logits_merged = signals_snippets[i].logits
                    logger.debug('alignment does not still exist')
                    continue
                else:
                    logger.debug('merged seq already found, so that is returned')
                    return SeqLogitsPair(
                        seq=seq_merged,
                        logits=logits_merged,
                    )
            else:
                merge_flag = True
                a = algns[0]
                seq1_gapped = a[0]
                seq2_gapped = a[1]
                alignment_score = a[2]
                merge_len = (
                    len(seq1_gapped) -
                    seq1_gapped.count('-') -
                    seq2_gapped.count('-')
                )
                alignment_score_per_base = alignment_score / merge_len
                #  Logging
                if alignment_score_per_base > MATCH_SCORE - 1.5:
                    color = 'white'
                elif alignment_score_per_base > MATCH_SCORE - 2.5:
                    color = 'yellow'
                elif alignment_score_per_base > MATCH_SCORE - 3.5:
                    color = 'magenta'
                else:
                    color = 'red'
                _ = color

                logger.debug('align 1: {}'.format(seq1_gapped))
                logger.debug('align 2: {}'.format(seq2_gapped))

                signals_snippets[i - 1].right_merge_len = merge_len
                signals_snippets[i].left_merge_len = merge_len
                logits1_gapped = SeqLogitsPair.align_logits(
                    seq_gapped=seq1_gapped,
                    logits_non_gapped=logits1_overlap,
                )
                logits2_gapped = SeqLogitsPair.align_logits(
                    seq_gapped=seq2_gapped,
                    logits_non_gapped=logits2_overlap,
                )

                seq_logits_pair1 = SeqLogitsPair(
                    seq=seq1_gapped,
                    logits=logits1_gapped,
                )
                seq_logits_pair2 = SeqLogitsPair(
                    seq=seq2_gapped,
                    logits=logits2_gapped,
                )
                seq_logits_pair_merged = MERGER.merge(
                    seq_logits_pair1,
                    seq_logits_pair2,
                )

                seq_merged = (
                    seq_merged[:-overlap_seq_len] +
                    seq_logits_pair_merged.seq +
                    seq_appended[overlap_seq_len:]
                )
                logits_merged = (
                    logits_merged[:-overlap_seq_len] +
                    seq_logits_pair_merged.logits +
                    logits_appended[overlap_seq_len:]
                )
        return SeqLogitsPair(
            seq=seq_merged,
            logits=logits_merged,
        )


class Inferer(object):

    def _select_seq(
            self,
            inferred_seqs: List[InferredSeqOneBeam],
    ) -> InferredSeqOneBeam:
        for inferred_seq in inferred_seqs:
            if '$' in inferred_seq.seq:
                return inferred_seq
        return inferred_seqs[0]

    def _split_signals(
            self,
            signals: List[float],
            signals_ID: str,
    ) -> List[SignalsSnippet]:
        start_index = 0
        snippet_index = 0
        signals_snippets = []  # type: List[SignalsSnippet]
        while(True):
            if start_index >= len(signals):
                break
            end_index = min(len(signals), start_index + self._signals_len)
            signals_part = signals[start_index:end_index]
            if len(signals_part) < self._signals_len // 2:
                break
            signals_snippet = SignalsSnippet(
                signals=signals_part,
                ID=signals_ID,
                index=snippet_index,
            )
            signals_snippets.append(signals_snippet)
            if end_index == len(signals):
                break
            start_index = end_index - self._overlap_len
            snippet_index += 1
        return signals_snippets

    def _make_snippets(
            self,
            fast5_paths: List[str],
    ) -> List[SignalsSnippet]:
        signals_snippets = []  # type: List[SignalsSnippet]
        for i, fast5_path in enumerate(fast5_paths):
            fast5 = Fast5()
            fast5.load_file(fast5_path)
            signals = fast5.get_signals()
            signals_snippets += self._split_signals(
                signals=signals,
                signals_ID=fast5_path,
            )
        return signals_snippets

    def _make_net(self) -> None:
        net = NetInference()
        cnn_cls = CNN_DICT[self._net_cfg['cnn']]
        encoder_cls = ENCODER_DICT[self._net_cfg['encoder']]
        decoder_cls = INFERENCE_DECODER_DICT[self._net_cfg['decoder']]
        cnn = cnn_cls(is_inference=True)
        encoder_rnn = encoder_cls(is_inference=True)
        decoder_rnn = decoder_cls(
            beam_width=self._beam_width,
            ignore_alignment_history=self._ignore_alignment_history,
            keep_full_alignment=self._keep_full_alignment,
        )
        net.cnn = cnn
        net.encoder_rnn = encoder_rnn
        net.decoder_rnn = decoder_rnn
        net.load_hps(self._hp_cfg)
        net.define_io_shapes(
            input_feature_num=1,
            output_feature_num=Nuc.TOKEN_NUM,
            input_max_len=self._signals_len + EOS_LEN,
            output_max_len=int(self._signals_len / MAX_SIGNALS_PER_BASE),
        )
        net.set_restore_ckpt_dir(dirname(self._config_path))
        net.construct()
        self._net = net
        return

    def __init__(
            self,
            config_path: str,
            signals_len: int,
            overlap_len: int,
            name: str,
            minibatch_size: int,
            beam_width: int,
            num_threads: int,
            gpus: List[int],
            ignore_alignment_history: bool,
            keep_full_alignment: bool,
            verbose: bool = False,
    ) -> None:
        if not verbose:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            tf.get_logger().setLevel('ERROR')

        # set attributes
        self._config_path = config_path
        self._signals_len = signals_len
        self._overlap_len = overlap_len
        #  logger.debug('signals_len: {}'.format(self._signals_len))
        #  logger.debug('overlap_len: {}'.format(self._overlap_len))
        self._name = name
        self._minibatch_size = minibatch_size
        self._beam_width = beam_width
        self._num_threads = num_threads
        self._ignore_alignment_history = ignore_alignment_history
        self._keep_full_alignment = keep_full_alignment
        self._verbose = verbose
        if self._verbose:
            logzero.loglevel(logging.DEBUG)
        else:
            logzero.loglevel(logging.ERROR)

        # Load config
        with open(config_path) as json_f:
            cfg = json.load(json_f)
        self._net_cfg = cfg['net']
        self._hp_cfg = cfg['hp']
        self._rt_cfg = cfg['rt']

        self._tf_config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=self._num_threads,
            intra_op_parallelism_threads=self._num_threads,
            gpu_options=tf.GPUOptions(
                per_process_gpu_memory_fraction=0.8,
                allow_growth=True,
                visible_device_list=','.join(list(map(str, gpus))),
            )
        )
        self._make_net()
        self._sess = tf.Session(config=self._tf_config)
        self._net.get_ready(
            sess=self._sess,
            ckpt_model_dir_path='',
            max_to_keep=1,
        )
        return

    def _infer_signals_snippets(
            self,
            signals_snippets: List[SignalsSnippet],
    ) -> List[List[InferredSeqOneBeam]]:
        start = time.time()
        minibatch_size = len(signals_snippets)
        #  max_signals_len = max([len(s.signals) for s in signals_snippets])
        # make input
        xs = np.zeros((
            minibatch_size,
            self._signals_len + EOS_LEN,
            1,
        ))
        x_lengths = []
        for i, signals_snippet in enumerate(signals_snippets):
            length = len(signals_snippet.signals)
            xs[i, :length, 0] = signals_snippet.signals
            xs[i, length:, :] = EOS_VALUE
            x_lengths.append(length + EOS_LEN)
        beam_seqs_lst = self._net.infer(
            sess=self._sess,
            xs=xs,
            x_lengths=x_lengths,
            minibatch_size=minibatch_size,
        )
        logger.debug('Basecalling speed: {:5.3f} signals/s'.format(
            sum([
                len(signals_snippet.signals)
                for signals_snippet in signals_snippets]
                ) / (time.time() - start),
        ))
        return beam_seqs_lst

    def _select_from_beams(
            self,
            signals_snippets: List[SignalsSnippet],
            beam_seqs_lst: List[List[InferredSeqOneBeam]],
    ) -> List[InferredSeqOneBeam]:
        inferred_seqs = [
            self._select_seq(beam_seqs)
            for beam_seqs in beam_seqs_lst
        ]
        for ss, inferred_seq in zip(signals_snippets, inferred_seqs):
            res = inferred_seq.trim_end_token()
            ss.inferred_seq = inferred_seq
            if res:
                continue
            logger.debug('end token not found\nID:{}\nindex:{}\n{}'.format(
                ss.ID,
                ss.index,
                ss.seq,
            ))
            res = ss.inferred_seq.trim_by_alignment()
            if res:
                logger.debug('Trimmed by alignment\nID:{}\nindex:{}\n{}'.format(
                    ss.ID,
                    ss.index,
                    ss.seq,
                ))
                continue
            res = ss.inferred_seq.remove_homopolytail()
            if res:
                logger.debug('Homopolymer was removed\nID:{}\nindex:{}\n{}'.format(
                    ss.ID,
                    ss.index,
                    ss.seq,
                ))
        return inferred_seqs

    def infer_fast5s(
            self,
            fast5_paths: List[str],
    ) -> InfererOutput:
        """
        Process one sub chunk(Fast5 files),
        and return the fasta format string.
        """
        # Make snippets of signals
        signals_snippets = self._make_snippets(fast5_paths)

        # Divide into minibatch
        ss_minibatches = list(chunked(signals_snippets, self._minibatch_size))

        # Infer against each minibatch
        beam_seqs_lsts = [
            self._infer_signals_snippets(minibatch)
            for minibatch in ss_minibatches
        ]  # type: List[List[List[InferredSeqOneBeam]]]

        # chain
        signals_snippets = list(chain(*ss_minibatches))
        beam_seqs_lst = list(chain(*beam_seqs_lsts))

        # Select best beam
        self._select_from_beams(
            signals_snippets=signals_snippets,
            beam_seqs_lst=beam_seqs_lst,
        )

        # Organize signal snippets
        organized_d = SignalsSnippets.organize(signals_snippets)

        # Merge signal snippets
        seq_d = {}  # type: Dict[str, SeqLogitsPair]
        for signals_ID in sorted(organized_d.keys()):
            logger.debug('Merger of {}'.format(
                signals_ID,
            ))
            signals_snippets = organized_d[signals_ID]
            start = time.time()
            seq_logits_pair = SignalsSnippets.merge(
                signals_snippets=signals_snippets,
                overlap_len=self._overlap_len,
                signals_per_base=SIGNALS_PER_BASE_OVERLAP,
            )
            logger.debug('Merge elapsed time: {:5.3f} s'.format(
                time.time() - start,
            ))
            seq_d[signals_ID] = seq_logits_pair

        # dumps meta json
        seriarizable_d = {
            k: [ss.to_seriarizable() for ss in v]
            for k, v in organized_d.items()
        }

        #  fasta_str = ''
        #  for signals_ID in sorted(organized_d.keys()):
        #      fast5_path = signals_ID
        #      fasta_str += '>{}\n{}\n'.format(
        #              basename(fast5_path),
        #              seq_d[signals_ID].seq,
        #              )
        inferer_output = InfererOutput(
            res_d=seq_d,
            meta_d=seriarizable_d,
        )
        #  inferer_output.meta_any = beam_seqs_lst
        return inferer_output

    def close(self) -> None:
        self._sess.close()
        return
