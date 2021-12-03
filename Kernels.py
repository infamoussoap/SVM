import tables


class LazyKernel:
    """ Kernel elements are never stored and are always computed when needed """
    def __init__(self, x_train, kernel_function):
        self.x_train = x_train
        self.kernel_function = kernel_function

    def __getitem__(self, slice_):
        if isinstance(slice_, slice):
            slice1 = slice_
            slice2 = slice(None, None, None)
        else:  # slice is instance of tuple
            assert len(slice_) == 2, "Indexing only valid for 2D indexing"
            slice1, slice2 = slice_

        return self.kernel_function(self.x_train[slice1], self.x_train[slice2])


class DiskKernel:
    """ Once initialized, kernel elements are computed and stored on the disk

        Accessing elements in the kernel will therefore require to read from disk
    """
    def __init__(self, x_train, kernel_function, table_filename="temp_kernel.h5"):
        self.x_train = x_train
        self.kernel_function = kernel_function

        self.table_filename = table_filename

        self.n = len(x_train)

        with tables.open_file(self.table_filename, mode="a", title="Root") as h5file:
            h5file.create_carray(h5file.root, "kernel", tables.Float64Atom(),
                                 shape=(self.n, self.n), chunkshape=(1, self.n))

        self.cache_kernel_to_disk()

    def cache_kernel_to_disk(self):
        """ Computes the kernel and sends results to disk (not RAM) """
        with tables.open_file(self.table_filename, mode="a", title="Root") as h5file:
            for i in range(self.n):
                h5file.root.kernel[i] = self.kernel_function(self.x_train[i], self.x_train)

    def __getitem__(self, item):
        if isinstance(item, (slice, int)):
            return self._getitem(item, slice(None, None, None))
        elif isinstance(item, tuple):
            assert len(item) == 2, "Indexing only valid for 2D indexing"

            s1, s2 = item
            if slice_length(s1, self.n) < slice_length(s2, self.n):
                return self._getitem(s1, s2)
            else:
                return self._getitem(s2, s1).T

        raise ValueError(f"Indexing not supported for type {type(item)}")

    def _getitem(self, s1, s2):
        with tables.open_file(self.table_filename, mode="r", title="Root") as h5file:
            return h5file.root.kernel[s1, s2]


def slice_length(s, max_length):
    if isinstance(s, int):
        return 1

    start = 0 if s.start is None else s.start
    stop = max_length if s.stop is None else s.stop
    step = 1 if s.step is None else s.step

    return (stop - start) // step
