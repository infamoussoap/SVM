import numpy as np


class LazyKernel:
    def __init__(self, x_train, kernel_function, *args, **kwargs):
        self.n = x_train.shape[0]
        self.x_train = x_train

        self.kernel_function = kernel_function

    def get_element(self, i, j):
        return self.kernel_function(self.x_train[i], self.x_train[j])

    def get_row(self, i):
        return self.kernel_function(self.x_train[i], self.x_train)

    def remove_file(self):
        pass
