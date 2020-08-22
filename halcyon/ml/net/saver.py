import heapq
import json
import os
from glob import glob
from typing import List  # noqa

import tensorflow as tf
from logzero import logger

MODEL_CKPT_PREFIX = 'model.ckpt'


class Saver(object):
    """ You can use this class as light-weighted wrapper
    for TensorFlow.Saver
    """

    # file name, which is used by Saver to look up best model,
    # or to watch the learning condition.
    VALUE_STORE_FILE_NAME = 'saver_value.json'
    VALUE_STORE_JSON_KEY = 'value'

    @property
    def buffer(self) -> List:
        return self.__buffer

    @property
    def tf_saver(self) -> tf.train.Saver:
        return self.__tf_saver

    def __init__(
        self,
        max_to_keep: int = 1,
        verbose: bool = False,
    ) -> None:
        if not verbose:
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            tf.get_logger().setLevel('ERROR')
        assert(max_to_keep > 0)
        self.__tf_saver = tf.train.Saver(max_to_keep=0)
        self.__max_to_keep = max_to_keep
        self.__buffer = []  # type: List
        return

    def __get_ckpt_file(self, dir_path: str) -> List[str]:
        return glob(os.path.join(dir_path, MODEL_CKPT_PREFIX + '*'))

    def __save_value(
            self,
            value: float,
            dir_path: str,
    ) -> None:
        with open(
                os.path.join(dir_path, self.VALUE_STORE_FILE_NAME,),
                'w') as f:
            json.dump({
                self.VALUE_STORE_JSON_KEY: float(value),
            },
                f,
                indent=2,
            )
        return

    def save(
        self,
        sess: tf.Session,
        value: float,
        dir_path: str,
        global_step: tf.Tensor = None,
    ) -> None:
        self.__save_value(value, dir_path)
        if len(self.__buffer) < self.__max_to_keep:
            heapq.heappush(self.__buffer, (-value, dir_path))
        elif value > -self.buffer[0][0]:
            return
        else:
            _, dir_path_old = heapq.heapreplace(
                self.__buffer,
                (-value, dir_path),
            )
            for path in self.__get_ckpt_file(dir_path_old):
                os.remove(path)
        os.path.join(dir_path, MODEL_CKPT_PREFIX),
        self.__tf_saver.save(
            sess,
            os.path.join(dir_path, MODEL_CKPT_PREFIX),
            global_step=global_step,
        )
        return

    def restore(
            self,
            sess: tf.Session,
            ckpt_model_dir_path: str,
    ) -> None:
        self.__tf_saver.restore(
            sess,
            os.path.join(ckpt_model_dir_path, MODEL_CKPT_PREFIX),
        )
        return

    def __get_value(
            self,
            ckpt_model_dir_path: str,
    ) -> float:
        with open(
                os.path.join(
                    ckpt_model_dir_path,
                    self.VALUE_STORE_FILE_NAME
                )
        ) as json_f:
            json_dict = json.load(json_f)
        return json_dict[self.VALUE_STORE_JSON_KEY]  # type: ignore

    def __get_best_ckpt_model_dir_path(
            self,
            ckpt_dir_path: str,
    ) -> str:
        ckpt_files = glob(os.path.join(
            ckpt_dir_path,
            '*',
            MODEL_CKPT_PREFIX + '*',
        ))
        dir_paths = list(set([os.path.dirname(f) for f in ckpt_files]))
        if len(dir_paths) == 0:
            logger.error('no ckpt models')
            raise ValueError
        dir_value_pairs = [
            (dir_path, self.__get_value(dir_path))
            for dir_path in dir_paths
        ]
        return min(dir_value_pairs, key=lambda x: x[1])[0]

    def restore_auto(
            self,
            sess: tf.Session,
            ckpt_dir_path: str,
    ) -> None:
        ckpt_model_dir_path = self.__get_best_ckpt_model_dir_path(
            ckpt_dir_path,
        )
        self.__tf_saver.restore(
            sess,
            os.path.join(ckpt_model_dir_path, MODEL_CKPT_PREFIX),
        )
        return
