from numpy import ndarray, array
from bolt.spark.array import BoltArraySpark
from thunder.series import fromarray, fromrdd

def toseries(y):

    if type(y) is ndarray:
        y = fromarray(y)
    elif type(y) is BoltArraySpark:
        y = fromrdd(y.tordd())

    return y
