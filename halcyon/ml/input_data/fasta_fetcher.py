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

from abc import ABCMeta, abstractclassmethod
from typing import Dict  # noqa
from typing import Tuple


class FastaFetcher(metaclass=ABCMeta):

    @abstractclassmethod
    def fetch(self, fast5_name: str) -> str:
        raise NotImplementedError

    @abstractclassmethod
    def get_item_dict(self) -> Dict[str, str]:
        raise NotImplementedError


class FastaFetcherGroundTruth(FastaFetcher):

    def __init__(self, fasta_path: str) -> None:
        with open(fasta_path) as f:
            lines = f.readlines()
        self.record_dict = {}  # type: Dict[str, str]
        fast5_name = ''
        seq = ''
        for line in lines:
            if line[0] == '>':
                if fast5_name != '':
                    self.record_dict[fast5_name] = seq
                fast5_name = line[1:].strip()
                seq = ''
            else:
                seq += line.strip()
        if fast5_name != '':
            self.record_dict[fast5_name] = seq
        return

    def fetch(self, fast5_name: str) -> str:
        seq = self.record_dict.get(fast5_name, '')
        return seq

    def get_item_dict(self) -> Dict[str, str]:
        res = {}
        for k, v in self.record_dict.items():
            res[k] = v
        return res


class FastaFetcherScrappie(FastaFetcher):
    ''' Play a role of parsing fasta file, output file of Scrappie
    ID line is as below.
    split_length_1000_15.fast5 { json }

    Atrributes:
        record_dict: dict {
            (fast_name: str) :
                json_info : str
                seq       : str
        }
    '''

    def __init__(self, fasta_path: str) -> None:
        with open(fasta_path) as f:
            lines = f.readlines()
        self.record_dict = {}  # type: Dict[str, Tuple[str, str]]
        fast5_name = ''
        json_info = ''
        seq = ''
        for line in lines:
            if line[0] == '>':
                if fast5_name != '':
                    self.record_dict[fast5_name] = (json_info, seq)
                json_start_index = line.index('{')
                fast5_name = line[1:json_start_index].strip()
                json_info = line[json_start_index:].strip()
                seq = ''
            else:
                seq += line.strip()
        if fast5_name != '':
            self.record_dict[fast5_name] = (json_info, seq)
        return

    def fetch(self, fast5_name: str) -> str:
        _, seq = self.record_dict.get(fast5_name, ('', ''))
        return seq

    def get_item_dict(self) -> Dict[str, str]:
        res = {}
        for k, (_, v) in self.record_dict.items():
            res[k] = v
        return res
