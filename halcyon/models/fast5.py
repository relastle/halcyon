import os
import random
from typing import Any, List  # noqa

import h5py
import numpy as np
from halcyon.models.seq import NucSeq


class Fast5(object):
    """class for fast5(HDF5)

    Attribute:
        fastq   : List[str] (length = 4)
        signals : List[int]
    """

    @classmethod
    def get_a_single_key(cls, group: h5py._hl.group.Group) -> str:
        keys = list(group.keys())
        if len(keys) == 1:
            return str(keys[0])
        else:
            return ''

    @classmethod
    def copy_group(
        cls,
        h5_from: h5py.File,
        h5_to: h5py.File,
        group_path: str,
    ) -> None:
        h5_to.create_group(group_path)
        for k, v in h5_from[group_path].attrs.items():
            h5_to[group_path].attrs.create(k, v)
        return

    @classmethod
    def split_fast5(
            cls,
            src_path: str,
            dest_dir: str,
            length: int,
            export_residue: bool = True,
            overlap_length: int = 0,
    ) -> None:
        """splitting the fast5 file in order to reduce the number of signals
        per one file.
        Args:
            src_path       : Path of fast5 file to be splited.
            dest_dir       : directory to which splited files will be exported
            length         : splited-length of signals
            export_residue :
        """
        h5 = h5py.File(src_path, 'r')
        raw_group = h5['Raw']
        key1 = Fast5.get_a_single_key(raw_group)

        reads_group = raw_group[key1]
        key2 = Fast5.get_a_single_key(reads_group)

        read_group = reads_group[key2]
        key3 = Fast5.get_a_single_key(read_group)

        signal_dataset = read_group[key3]

        fast5 = Fast5()
        fast5.load_file(src_path)
        whole_length = len(fast5.signals)

        os.makedirs(dest_dir, exist_ok=True)
        split_index = 0
        can_return = False
        while(True):
            signal_start_index = (
                split_index * length -
                split_index * overlap_length
            )
            signal_end_index = signal_start_index + length
            if signal_end_index < whole_length:
                signal_dataset_partial = signal_dataset[
                    signal_start_index:signal_end_index
                ]
                length_partial = length
                new_path = os.path.join(
                    dest_dir,
                    'split_length-{}_overlap-{}_{}.fast5'.format(
                        length,
                        overlap_length,
                        split_index,
                    ))
            else:
                if not export_residue:
                    return
                signal_dataset_partial = signal_dataset[signal_start_index:]
                length_partial = len(signal_dataset) - signal_start_index
                new_path = os.path.join(
                    dest_dir,
                    'split_length-{}_overlap-{}_residue.fast5'.format(
                        length,
                        overlap_length,
                    ))
                can_return = True
            if os.path.exists(new_path):
                if can_return:
                    return
                else:
                    split_index += 1
                    continue
            with h5py.File(new_path, 'w') as new_h5:
                # Copy of HDF5 data
                read_group_path = 'Raw/{}/{}/'.format(key1, key2)
                new_h5.create_group(read_group_path)
                new_h5[read_group_path].create_dataset(
                    key3,
                    shape=(length_partial,),
                    dtype=np.int16,
                    data=signal_dataset_partial,
                )
                cls.copy_group(h5, new_h5, 'UniqueGlobalKey/channel_id')
                cls.copy_group(h5, new_h5, 'UniqueGlobalKey/context_tags')
                cls.copy_group(h5, new_h5, 'UniqueGlobalKey/tracking_id')
            if can_return:
                return
            split_index += 1
        return

    def __init__(self) -> None:
        return

    def _set_fastq(self, h5: h5py._hl.files.File) -> None:
        k1 = 'Analyses'
        k2 = 'Basecall_1D_000'
        k3 = 'BaseCalled_template'
        k4 = 'Fastq'
        try:
            fastq_dataset = h5[k1][k2][k3][k4]
        except KeyError:
            return
        fastq_str = str(np.array(fastq_dataset))[2:-1].strip('\\n')
        fastq_elms = fastq_str.split('\\n')
        self.fastq = fastq_elms
        return

    def _set_signals(self, h5: h5py._hl.files.File) -> None:
        raw_group = h5['Raw']
        key = Fast5.get_a_single_key(raw_group)

        reads_group = raw_group[key]
        key = Fast5.get_a_single_key(reads_group)

        read_group = reads_group[key]
        key = Fast5.get_a_single_key(read_group)

        signal_dataset = read_group[key]
        signals = np.array(signal_dataset)
        self.signals = list(signals)
        return

    def is_valid(self) -> bool:
        return len(self.get_seq()) > 0 and len(self.signals) > 0

    def _set_params(self, h5: h5py._hl.files.File) -> None:
        self._set_fastq(h5)
        self._set_signals(h5)

    def load_file(self, path: str) -> None:
        """ Initializer from Fast5 file path
        """
        h5 = h5py.File(path, 'r')
        self._set_params(h5)
        return

    def get_seq(self) -> str:
        return self.fastq[1]

    def get_signals(self) -> List[float]:
        """ Get the regularized signal data
        """
        LOWER_BOUND = 0
        UPPER_BOUND = 1000
        signals = np.array(self.signals)
        signals_lower_bounded = np.maximum(LOWER_BOUND, signals)
        signals_upper_bounded = np.minimum(UPPER_BOUND, signals_lower_bounded)
        result = signals_upper_bounded / (UPPER_BOUND - LOWER_BOUND)
        return list(result)


class Fast5Resquiggled(Fast5):

    def _set_resquiggled_seq(self, h5: h5py._hl.files.File) -> None:
        k1 = 'Analyses'
        k2 = 'RawGenomeCorrected_000'
        k3 = 'BaseCalled_template'
        k4 = 'Events'
        event_dataset = h5[k1][k2][k3][k4]
        self.resquiggled_seq = ''.join(
            [row[4].decode('utf-8') for row in event_dataset[0:]]
        )
        return

    # Override
    def load_file(self, path: str) -> None:
        h5 = h5py.File(path, 'r')
        super(Fast5Resquiggled, self)._set_params(h5)
        self._set_resquiggled_seq(h5)
        return

    @classmethod
    def get_partial_event(
            cls,
            event_dataset: Any,
            signal_start_index: int,
            signal_end_index: int,
    ) -> Any:
        seq_start_index = -1
        seq_end_index = -1
        # NOTE: ここの[0:]っていうスライシングがないと何故かクソ遅い
        for i, row in enumerate(event_dataset[0:]):
            if seq_start_index == -1 and int(row[2]) >= signal_start_index:
                seq_start_index = i
            if seq_end_index == -1 and int(row[2]) >= signal_end_index:
                seq_end_index = i
                break
        return event_dataset[seq_start_index:seq_end_index]

    @classmethod
    def split_fast5(
            cls,
            src_path: str,
            dest_dir: str,
            length: int,
            export_residue: bool = True,
            overlap_length: int = 0,
    ) -> None:
        """splitting the fast5 file in order to reduce the number of signals
        per one file.
        Args:
            src_path       : Path of fast5 file to be splited.
            dest_dir       : directory to which splited files will be exported
            length         : splited-length of signals
            export_residue :
        TODO: This method is almost the same logic as super method
        """
        try:
            h5 = h5py.File(src_path, 'r')
        except Exception:  # TODO
            return
        raw_group = h5['Raw']
        key1 = Fast5.get_a_single_key(raw_group)

        reads_group = raw_group[key1]
        key2 = Fast5.get_a_single_key(reads_group)

        read_group = reads_group[key2]
        key3 = Fast5.get_a_single_key(read_group)

        signal_dataset = read_group[key3]

        k1 = 'Analyses'
        k2 = 'RawGenomeCorrected_000'
        k3 = 'BaseCalled_template'
        k4 = 'Events'
        try:
            event_dataset = h5[k1][k2][k3][k4]
        except Exception:  # TODO
            return

        fast5 = Fast5()
        fast5.load_file(src_path)
        whole_length = len(fast5.signals)
        os.makedirs(dest_dir, exist_ok=True)
        split_index = 0
        can_return = False
        while(True):
            signal_start_index = (
                split_index * length -
                split_index * overlap_length
            )
            signal_end_index = signal_start_index + length
            event_dataset_partial = cls.get_partial_event(
                event_dataset,
                signal_start_index,
                signal_end_index,
            )
            if signal_end_index < whole_length:
                signal_dataset_partial = signal_dataset[
                    signal_start_index:signal_end_index
                ]
                length_partial = length
                new_path = os.path.join(
                    dest_dir,
                    'split_length-{}_overlap-{}_{}.fast5'.format(
                        length,
                        overlap_length,
                        split_index,
                    ))
            else:
                if not export_residue:
                    return
                signal_dataset_partial = signal_dataset[signal_start_index:]
                length_partial = len(signal_dataset) - signal_start_index
                new_path = os.path.join(
                    dest_dir,
                    'split_length-{}_overlap-{}_residue.fast5'.format(
                        length,
                        overlap_length,
                    ))
                can_return = True
            if os.path.exists(new_path):
                if can_return:
                    return
                else:
                    split_index += 1
                    continue
            with h5py.File(new_path, 'w') as new_h5:
                # Copy of HDF5 data
                read_group_path = 'Raw/{}/{}/'.format(key1, key2)
                event_group_path = '{}/{}/{}/'.format(k1, k2, k3)
                new_h5.create_group(read_group_path)
                new_h5.create_group(event_group_path)
                new_h5[read_group_path].create_dataset(
                    key3,
                    shape=(length_partial,),
                    dtype=np.int16,
                    data=signal_dataset_partial,
                )
                new_h5[event_group_path].create_dataset(
                    k4,
                    data=event_dataset_partial,
                )
                cls.copy_group(h5, new_h5, 'UniqueGlobalKey/channel_id')
                cls.copy_group(h5, new_h5, 'UniqueGlobalKey/context_tags')
                cls.copy_group(h5, new_h5, 'UniqueGlobalKey/tracking_id')
            if can_return:
                return
            split_index += 1
        return


class Fast5Chiron(Fast5):
    """ Fast5 object using called nucleotide sequences by Chiron.
    """

    def __init__(self) -> None:
        return

    def _set_fastq(self, fastq_path: str) -> None:
        with open(fastq_path) as fastq:
            lines = fastq.readlines()
        self.fastq = [line.strip() for line in lines]
        return

    def load_fast5_fastq(self, fast5_path: str, fastq_path: str) -> None:
        """ Initializer from Fast5 file path
        """
        h5 = h5py.File(fast5_path, 'r')
        self._set_signals(h5)
        self._set_fastq(fastq_path)
        return


class Fast5Mock(Fast5):

    def __init__(self) -> None:
        signals_len = random.randint(4, 10)
        seq_len = random.randint(4, 10)
        self.signals = list(range(0, signals_len))
        self.fastq = [
            '@aaa',
            NucSeq.create_random_seq(seq_len),
            '+',
            '+' * seq_len,
        ]


def main() -> None:
    return


if __name__ == '__main__':
    main()
