import os

import numpy as np
import tables

from .helper_file import slice_length


class DiskKernel:
    """ Once initialized, kernel elements are computed and stored on the disk

        Accessing elements in the kernel will therefore require to read from disk
    """
    def __init__(self, x_train, kernel_function, table_filename="temp_kernel.h5"):
        self.x_train = x_train
        self.kernel_function = kernel_function

        self.table_filename = table_filename

        self.n = len(x_train)

        with tables.open_file(self.table_filename, mode="w", title="Root") as h5file:
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

            # Indexing is slower row-wise, so better to perform as little row access as you can
            s1, s2 = item
            if slice_length(s1, self.n) <= slice_length(s2, self.n):
                return self._getitem(s1, s2)
            else:
                return self._getitem(s2, s1).T

        raise ValueError(f"Indexing not supported for type {type(item)}")

    def _getitem(self, s1, s2):
        # If you are getting a single item from the kernel it is usually faster to just compute it, rather
        # than to retrieve it from the disk
        if isinstance(s1, slice) or isinstance(s2, slice):
            with tables.open_file(self.table_filename, mode="r", title="Root") as h5file:
                return h5file.root.kernel[s1, s2]
        else:
            return self.kernel_function(self.x_train[s1], self.x_train[s2])

    def remove_file(self):
        os.remove(self.table_filename)
