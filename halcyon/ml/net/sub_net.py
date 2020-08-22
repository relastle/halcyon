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

import sys
from abc import ABCMeta, abstractclassmethod
from typing import Tuple  # noqa
from typing import Any, Dict, List


class HP_KEY_NOT_FOUND(Exception):
    """
    """
    pass


class SubNet(metaclass=ABCMeta):
    """
    Convolutional Neural Network Module
    """

    @property
    def hps(self) -> Dict[str, Any]:
        return self._hps  # type: ignore

    @property
    def tag(self) -> str:
        return self.get_tag()

    def __init__(self) -> None:
        return

    @abstractclassmethod
    def get_tag(self) -> str:
        raise NotImplementedError

    @abstractclassmethod
    def get_hp_names(self) -> List[str]:
        raise NotImplementedError

    def __check_hp_keys(self) -> bool:
        for name in self.get_hp_names():
            if name not in self.hps.keys():
                sys.stderr.write(
                    'Key {} not found in json\n'.format(
                        name
                    ))
                return False
        return True

    def __getitem__(self, key: str) -> Any:
        return self.hps[key]

    def load_hps(
            self,
            hp_dict: Dict[str, Any],
    ) -> None:
        self._hps = hp_dict[self.tag]
        if not self.__check_hp_keys():
            raise HP_KEY_NOT_FOUND
        return
