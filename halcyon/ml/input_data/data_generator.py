#  Copyright 2018/10/23 Hiroki Konishi. All Rights Reserved.
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import random
from abc import ABCMeta, abstractclassmethod
from typing import Generator, List, Tuple

import numpy as np
from halcyon.config import Const
from halcyon.ml.input_data.sample import Sample, Samples
from halcyon.models.seq import Nuc, NucSeq


class DataGenerator(metaclass=ABCMeta):

    @property
    def input_feature_num(self) -> int:
        if self._input_feature_num < 1:
            raise ValueError
        return self._input_feature_num

    @property
    def output_feature_num(self) -> int:
        if self._output_feature_num < 1:
            raise ValueError
        return self._output_feature_num

    @property
    def input_max_len(self) -> int:
        if self._input_max_len < 1:
            raise ValueError
        return self._input_max_len

    @property
    def output_max_len(self) -> int:
        if self._output_max_len < 1:
            raise ValueError
        return self._output_max_len

    def __init__(self) -> None:
        self._input_feature_num = 0
        self._output_feature_num = 0
        self._input_max_len = 0
        self._output_max_len = 0

    @abstractclassmethod
    def get_minibatch(
            self,
            minibatch_size: int,
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray, int,
    ]:
        ''' This function returns the set of data(signals) and labels(sequences).
        The dimention of ndarray in a minibatch should be consistent throught
        out training.  So you should specify the dimention first with the
        method `determine_sample_dimension`.

        Returns:
            xs : np.ndarray (minibatch_size, signals_max_len, 1)
            ys : np.ndarray (minibatch_size, seq_max_len, 5)
            ys_shifted           : np.ndarray (minibatch_size, seq_max_len, 5)
            Actual lengths of xs : np.ndarray (minibatch_size)
            Actual lengths of ys : np.ndarray (minibatch_size)
            minibatch_size       : int
        '''
        raise NotImplementedError

    @abstractclassmethod
    def close(self) -> None:
        raise NotImplementedError


class DataGeneratorNanopore(DataGenerator):
    ''' Generating the data fed to the LSTM.

    Attributes:
        _samples: Samples
        _cursor: int # indicating the index of fast5 to yield
    '''

    def __init__(self, samples: Samples) -> None:
        self._input_feature_num = Const.INPUT_FEATURE_NUM
        self._output_feature_num = Nuc.TOKEN_NUM
        self._samples = samples  # type: Samples
        self._input_max_len = self._samples.get_in_max_len()
        self._output_max_len = self._samples.get_out_max_len()
        self._pointer_lst = list(range(len(self._samples)))
        self._cursor = 0
        return

    def shuffle(self) -> None:
        random.shuffle(self._pointer_lst)
        return

    def _increment_cursor(self) -> None:
        if self._cursor == len(self._samples) - 1:
            self._cursor = 0
            self.shuffle()
        else:
            self._cursor += 1
        return

    @classmethod
    def find_consensus(cls, dgs: List['DataGeneratorNanopore']) -> None:
        ''' Have DataGenerators reach consensus of max length of
        input and output.  You should call this function when you want to
        have consensus among several DataGenerator such as DataGenerator
        for training and DataGenerator for validation
        '''
        input_max_len = max([dg.input_max_len for dg in dgs])
        output_max_len = max([dg.output_max_len for dg in dgs])
        for dg in dgs:
            dg._input_max_len = input_max_len
            dg._output_max_len = output_max_len
        return

    def get_sample(self) -> Tuple[np.ndarray, np.ndarray]:
        '''Get pair of encoded signals and nucseq
        '''
        pointer = self._pointer_lst[self._cursor]
        sample = self._samples[pointer]  # type: Sample
        seq_str = sample.seq
        nucseq = NucSeq(seq_str)
        signals = np.array(sample.signals)

        # Regularize
        if np.average(signals) > 1:
            LOWER_BOUND = 0
            UPPER_BOUND = 1000
            signals_lower_bounded = np.maximum(LOWER_BOUND, signals)
            signals_upper_bounded = np.minimum(UPPER_BOUND, signals_lower_bounded)
            signals = signals_upper_bounded / (UPPER_BOUND - LOWER_BOUND)
        # validation of sample
        #  if not (len(seq) > 0 and len(signals) > 0):
        #      raise ValueError
        self._increment_cursor()
        return (
            np.reshape(signals, (len(signals), 1)),
            nucseq.encode(),
        )

    def get_minibatch(self, minibatch_size: int) -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
    ]:
        xs = np.zeros(shape=(
            minibatch_size,
            self._input_max_len,
            self._input_feature_num,
        ))
        ys = np.zeros(shape=(
            minibatch_size,
            self._output_max_len,
            self._output_feature_num,
        ))
        x_lengths = []  # type : List[int]
        y_lengths = []  # type : List[int]
        for i in range(minibatch_size):
            x, y = self.get_sample()
            xs[i, 0:x.shape[0], :] = x  # insert signals
            xs[i, x.shape[0]:, :] = [Const.END_OF_SIGNALS_VALUE]  # insert EOS
            ys[i, 0:y.shape[0], :] = y  # insert seq
            end_token_vector = [0] * Nuc.TOKEN_NUM
            end_token_vector[Nuc.NUC_MAP['$']] = 1
            ys[i, y.shape[0]:, :] = end_token_vector  # create end of seq
            x_lengths.append(x.shape[0] + Const.END_OF_SIGNALS_LENGTH)
            y_lengths.append(y.shape[0] + Const.END_TOKEN_LENGTH)
        ys_shifted = np.zeros(shape=ys.shape)
        start_token_vector = [0] * Nuc.TOKEN_NUM
        start_token_vector[Nuc.NUC_MAP['^']] = 1
        ys_shifted[:, 1:, :] = ys[:, :-1, :]
        ys_shifted[:, 0, :] = start_token_vector  # initial nucleotide
        return (
            xs,
            ys,
            ys_shifted,
            np.array(x_lengths),
            np.array(y_lengths),
            minibatch_size,
        )

    def generate(
            self,
            minibatch_size: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        while(True):
            xs, ys, _, _, _, _ = self.get_minibatch(minibatch_size)
            yield xs, ys

    def close(self) -> None:
        self._samples.close()
        return
