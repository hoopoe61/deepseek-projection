from enum import Enum


class DataType(Enum):
    FP32 = 0
    TF32 = 1
    BF16 = 2
    FP8 = 3

    def bytes_of_dtype(self):
        if self in (DataType.FP32, DataType.TF32):
            return 4
        elif self == DataType.BF16:
            return 2
        elif self == DataType.FP8:
            return 1
        else:
            raise NotImplementedError()


class NormType(Enum):
    LAYER_NORM = 0
    RMS_NORM = 1
