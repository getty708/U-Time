import utime
from .hparams import YAMLHParams as _YAMLHParams


class YAMLHParams(_YAMLHParams):
    def __init__(self, *args, **kwargs):
        kwargs["package"] = utime.__name__
        super(YAMLHParams, self).__init__(
            *args, **kwargs
        )
