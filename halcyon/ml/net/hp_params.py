
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
