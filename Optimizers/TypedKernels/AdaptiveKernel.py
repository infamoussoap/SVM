import numpy as np
import tables

import os
import threading
import time
import warnings

from Optimizers.TypedKernels import DiskKernelV2
from Optimizers.TypedKernels import LazyKernelV2


class AdaptiveKernel:
    def __init__(self, x_train, kernel_function, sleep_time=0.01, table_filename="temp_kernel.h5"):
        self.n = x_train.shape[0]
        self.x_train = x_train

        self.kernel_function = kernel_function

        self.table_filename = DiskKernelV2.get_unique_h5_filename(table_filename)

        self.read_from_disk = self.is_read_disk_faster(i=0)

        if self.read_from_disk:
            print("Adaptive Kernel is reading from disk.")
            self.kernel = DiskKernelV2(x_train, kernel_function, sleep_time=sleep_time,
                                      table_filename=table_filename)
        else:
            print("Adaptive Kernel is computing on the fly.")
            self.kernel = LazyKernelV2(x_train.copy(), kernel_function)

    def create_table(self):
        with tables.open_file(self.table_filename, mode="w", title="Root") as h5file:
            h5file.create_carray(h5file.root, "kernel", tables.Float64Atom(),
                                 shape=(self.n, self.n), chunkshape=(1, self.n))

    def is_read_disk_faster(self, i=0):
        self.create_table()

        disk_time = self.time_to_read_row_from_disk(i)
        compute_time = self.time_to_compute_row(i)

        os.remove(self.table_filename)

        time_tol = 1e-3
        return (disk_time - time_tol) < compute_time

    def time_to_read_row_from_disk(self, i=0):
        start = time.time()
        with tables.open_file(self.table_filename, mode="r", title="Root") as h5file:
            val = h5file.root.kernel[i]
        end = time.time()

        return end - start

    def time_to_compute_row(self, i):
        start = time.time()
        val = self.kernel_function(self.x_train[i], self.x_train)
        end = time.time()

        return end - start

    def get_row(self, i):
        return self.kernel.get_row(i)

    def get_element(self, i, j):
        return self.kernel.get_element(i, j)

    def remove_file(self):
        self.kernel.remove_file()
