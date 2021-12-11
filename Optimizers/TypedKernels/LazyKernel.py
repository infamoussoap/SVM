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
