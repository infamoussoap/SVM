import numpy as np
import tables

import os
import threading
import time
import warnings


class DiskKernel:
    def __init__(self, x_train, kernel_function, sleep_time=0.01, table_filename="temp_kernel.h5"):
        self.n = x_train.shape[0]
        self.x_train = x_train

        self.sleep_time = sleep_time

        self.kernel_function = kernel_function

        self.table_filename = table_filename
        self.create_h5_file()

        self.is_row_set = np.zeros(self.n).astype(bool)

        self.kernel_locked = False
        self.queue = []
        self.initialize_kernel()

    def initialize_kernel(self):
        for i in reversed(range(self.n)):
            self.queue.append(("set", i, np.array(False), None))

        self.kernel_locked = True

        thread = threading.Thread(target=self._simplify_queue)
        thread.start()

    def get_element(self, i, j):
        return self.kernel_function(self.x_train[i], self.x_train[j])

    def get_row(self, i):
        if self.kernel_locked:
            is_action_complete = np.array(False)
            val = np.zeros(self.n)

            self.queue.append(("get", i, is_action_complete, val))

            while not is_action_complete:
                time.sleep(self.sleep_time)
            return val

        return self._get_row(i)

    def _set_row(self, i):
        if self.is_row_set[i]:
            return

        val = self.kernel_function(self.x_train[i], self.x_train)
        with tables.open_file(self.table_filename, mode="a", title="Root") as h5file:
            h5file.root.kernel[i] = val

        self.is_row_set[i] = True

    def _get_row(self, i):
        if not self.is_row_set[i]:
            warnings.warn(f"Row {i} is being retrived without being set first.")

        with tables.open_file(self.table_filename, mode="r", title="Root") as h5file:
            return h5file.root.kernel[i]

    def remove_file(self):
        os.remove(self.table_filename)

    def _simplify_queue(self):
        while self.kernel_locked:
            action, i, is_action_complete, val = self.queue.pop()

            if action == "set":
                self._set_row(i)
            elif action == "get":
                self._set_row(i)
                memoryview(val)[:] = self._get_row(i)
            else:
                raise ValueError(f"'{action}' is not a valid action.")

            memoryview(is_action_complete)[...] = np.array(True)

            if len(self.queue) == 0:
                self.kernel_locked = False
                print("Kernel Is Fully Computed")

    def create_h5_file(self):
        with tables.open_file(self.table_filename, mode="w", title="Root") as h5file:
            h5file.create_carray(h5file.root, "kernel", tables.Float64Atom(),
                                 shape=(self.n, self.n), chunkshape=(1, self.n))
