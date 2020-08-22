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


class HParams(object):
    """ Class of a set of hyperparameters of a deep learning model.
    Its datatype is a wrapper of dictionary.
    Hyperparameters are the parameters that are dependent on values
    restored from heckpoint.
    In other words, parameters such as minibatch_size and input feature num are
    nor hyperparameters because they can be changed after a model is restored
    from checkpoint.
    """

    def __init__(self) -> None:
        pass
