# -*- coding: utf-8 -*-
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

'''
Providing dataset that can be easily solved by the network.
You can validatet that network construction
is successful by using this data generator
'''

from typing import List  # noqa
from typing import Tuple

import numpy as np
from halcyon.ml.input_data.data_generator import DataGenerator
from halcyon.models.seq import Nuc, NucSeq, NucSeqSignals


class DataGeneratorShifted(DataGenerator):
    '''
    this data is nucleotide sequence to k-shifted nucleotide sequence
    '''

    LENGTH = 60

    def __init__(self) -> None:
        self._input_feature_num = Nuc.TOKEN_NUM
        self._output_feature_num = Nuc.TOKEN_NUM

    def get_minibatch(self, minibatch_size: int) -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
    ]:
        '''
        Returns:
            xs : np.ndarray (minibatch_size, signals_max_len, 1)
            ys : np.ndarray (minibatch_size, seq_max_len, Nuc.TOKEN_NUM)
            ys_shifted           : np.ndarray (same as ys)
            Actual lengths of xs : np.ndarray (minibatch_size)
            Actual lengths of ys : np.ndarray (minibatch_size)
            minibatch_size       : int
        '''
        k = 10  # the number of end token
        xs = np.zeros(shape=(
            minibatch_size,
            DataGeneratorShifted.LENGTH,
            Nuc.TOKEN_NUM,
        ))
        ys = np.zeros(shape=(
            minibatch_size,
            DataGeneratorShifted.LENGTH,
            Nuc.TOKEN_NUM,
        ))
        x_lengths = []  # type : List[int]
        # TODO: 今はmaxのlengthsを入れていく
        y_lengths = []  # type : List[int]
        for i in range(minibatch_size):
            seq_str = NucSeq.create_random_seq(DataGeneratorShifted.LENGTH)
            seq = NucSeq(seq_str)
            x = seq.encode()
            xs[i, :, :] = x
            if k == 0:
                ys[i, :, :] = x
            else:
                ys[i, k:, :] = x[:-k]
                ys[i, :k, :] = np.array(
                    [1] + [0] * (self.output_feature_num - 1))
            x_lengths.append(x.shape[0])
            y_lengths.append(x.shape[0])
        ys_shifted = np.zeros(shape=ys.shape)
        ys_shifted[:, 1:, :] = ys[:, :-1, :]
        start_token_vector = [0] * Nuc.TOKEN_NUM
        start_token_vector[Nuc.NUC_MAP['^']] = 1
        ys_shifted[:, 0, :] = start_token_vector  # initial nucleotide
        return (
            xs,
            ys,
            ys_shifted,
            np.array(x_lengths),
            np.array(y_lengths),
            minibatch_size,
        )

    def close(self) -> None:
        return


class DataGeneratorSignals(DataGenerator):

    def __init__(
            self,
            length: int = 100,
            end_token_length: int = 5,
    ) -> None:
        self._input_feature_num = 1
        self._output_feature_num = Nuc.TOKEN_NUM
        self._end_token_length = end_token_length
        self._output_max_len = length
        self._input_max_len = self._output_max_len * NucSeqSignals.LEN_MAX
        self._nuc_seq_length = self._output_max_len - self._end_token_length
        return

    def get_minibatch(self, minibatch_size: int) -> Tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            int,
    ]:
        '''
        Returns:
            xs : np.ndarray (minibatch_size, signals_max_len, 1)
            ys : np.ndarray (minibatch_size, seq_max_len, 5)
            ys_shifted           : np.ndarray (minibatch_size, seq_max_len, 5)
            Actual lengths of xs : np.ndarray (minibatch_size)
            Actual lengths of ys : np.ndarray (minibatch_size)
            minibatch_size       : int
        '''
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
            seq_str = NucSeq.create_random_seq(self._nuc_seq_length)
            seq = NucSeqSignals(seq_str)
            x = seq.get_answer()
            xs[i, :x.shape[0], :] = x
            x_lengths.append(x.shape[0])
            y = seq.encode()
            ys[i, :self._nuc_seq_length, :] = y
            ys[
                i,
                self._nuc_seq_length:self._output_max_len,
                Nuc.NUC_MAP['$'],
            ] = 1
            y_lengths.append(self._output_max_len)
        ys_shifted = np.zeros(shape=ys.shape)
        ys_shifted[:, 1:, :] = ys[:, :-1, :]
        ys_shifted[:, 0, 1] = 1
        return (
            xs,
            ys,
            ys_shifted,
            np.array(x_lengths),
            np.array(y_lengths),
            minibatch_size,
        )

    def close(self) -> None:
        return
