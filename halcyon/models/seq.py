"""
model of Genome sequence
"""

import random
from pprint import pprint as pp
from typing import Generator, List, Tuple

import numpy as np


class Nuc(object):
    """
    """

    NUC_NUM = 4
    TOKEN_NUM = 6
    NUC_MAP = {
        '^': 0,  # start character for base-calling
        '$': 1,  # end character for base-calling
        'A': 2,
        'T': 3,
        'G': 4,
        'C': 5,
        'a': 2,
        't': 3,
        'g': 4,
        'c': 5,
    }

    NUC_ARR = [
        '^',  # start character for base-calling
        '$',  # end character for base-calling
        'A',
        'T',
        'G',
        'C',
    ]

    @classmethod
    def randnuc(cls) -> str:
        randi = random.randint(0, Nuc.NUC_NUM - 1)
        if randi == 0:
            return 'A'
        elif randi == 1:
            return 'T'
        elif randi == 2:
            return 'G'
        elif randi == 3:
            return 'C'
        return ''

    @classmethod
    def encode(cls, nuc: str) -> List[int]:
        vec = [0] * cls.TOKEN_NUM
        vec[cls.NUC_MAP[nuc]] = 1
        return vec

    @classmethod
    def int2nuc(cls, n: int) -> str:
        if n == 0:
            return '^'
        if n == 1:
            return '$'
        if n == 2:
            return 'A'
        elif n == 3:
            return 'T'
        elif n == 4:
            return 'G'
        elif n == 5:
            return 'C'
        else:
            raise ValueError


class NucSeq(object):
    """
    Attribute:
        seq : str
    """

    @classmethod
    def create_random_seq(cls, n: int) -> str:
        return ''.join([Nuc.randnuc() for i in range(n)])

    @classmethod
    def decode(cls, encoded_nucseq: np.ndarray) -> str:
        return ''.join([Nuc.int2nuc(np.argmax(nuc)) for nuc in encoded_nucseq])

    @classmethod
    def get_pairs(
            cls,
            n_sample: int,
            length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for i in range(n_sample):
            x, y = cls.get_pair(length)
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    @classmethod
    def get_pair(cls, n: int) -> Tuple[np.ndarray, np.ndarray]:
        seq_in = NucSeq(cls.create_random_seq(n))
        ans = seq_in.get_answer()
        x = seq_in.encode()
        return x, ans

    @classmethod
    def generate(
            cls,
            length: int,
            minibatch_size: int,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        while(True):
            xs, ys = cls.get_pairs(minibatch_size, length)
            yield xs, ys
        return

    @classmethod
    def is_valid_nuc(cls, c: str) -> bool:
        if c in Nuc.NUC_MAP.keys():
            return True
        else:
            return False

    @classmethod
    def is_valid_seq(cls, seq: str) -> bool:
        for char in seq:
            if cls.is_valid_nuc(char):
                continue
            else:
                return False
        return True

    @classmethod
    def force_to_valid_seq(cls, seq: str) -> str:
        seq_lst = list(seq)
        for i, char in enumerate(seq):
            if cls.is_valid_nuc(char):
                continue
            else:
                seq_lst[i] = 'N'
        return ''.join(seq_lst)

    @classmethod
    def encode_seq(cls, seq: str) -> np.ndarray:
        arr = np.array([Nuc.encode(nuc) for nuc in seq])
        return arr

    def get_answer(self) -> np.ndarray:
        pass

    def __init__(self, seq: str) -> None:
        """ Initialize from Sequence.
        Check if invalid characters are included.
        """
        self.seq = NucSeq.force_to_valid_seq(seq)
        return

    def __repr__(self) -> str:
        return self.seq

    def encode(self) -> np.ndarray:
        """
        Encode nucleotide seuqnece(str) to the 2d-array in one-hot encoding.
        """
        return NucSeq.encode_seq(self.seq)


class NucSeqShifted(NucSeq):
    def get_answer(self) -> np.ndarray:
        """
        Get the answer sequence. Answer sequence has the same sequence as the query
        until the first 'C' Occurence and the rest is set to 'N'.
        i.g.) 'ATGGCTTA' -> 'ATGGCNNN'
        """
        ans_seq = ''
        for i, char in enumerate(self.seq):
            ans_seq += char
            if char == 'C':
                ans_seq += 'N' * (len(self.seq) - i - 1)
                break
        return NucSeq.encode_seq(ans_seq)


class NucSeqSignals(NucSeq):
    """ Test nucleotide modules that make signals based on nucs.
    The number of signals per one nucleotide is randomly defined.
    """
    LEN_MIN = 5
    LEN_MAX = 10

    def _get_signals(self, nuc: str) -> List[float]:
        length = random.randint(NucSeqSignals.LEN_MIN, NucSeqSignals.LEN_MAX)
        if nuc == 'A':
            return [0.2] * length
        elif nuc == 'T':
            return [0.4] * length
        elif nuc == 'G':
            return [0.6] * length
        elif nuc == 'C':
            return [0.8] * length
        elif nuc == 'N':
            return [0.0] * length
        raise ValueError

    def get_answer(self) -> np.ndarray:
        signals = []  # type: List[float]
        for nuc in self.seq:
            signals += self._get_signals(nuc)
        return np.array(signals).reshape(len(signals), 1)


def main() -> None:
    pp(NucSeq.force_to_valid_seq('AAGGNNMM'))
    return


if __name__ == '__main__':
    main()
