import numpy as np


def slice_length(s, max_length):
    """ Given a slice, will return the length of the slice

        Parameters
        ----------
        s : slice
        max_length : int
            The maximum length of the array to be indexed by the slice
    """
    if isinstance(s, (int, np.int64)):
        return 1

    start = 0 if s.start is None else s.start
    stop = max_length if s.stop is None else s.stop
    step = 1 if s.step is None else s.step

    return (stop - start) // step
