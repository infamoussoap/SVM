class StockKernel:
    """ Kernel elements are just stored in RAM """
    def __init__(self, x_train, kernel_function):
        self.x_train = x_train
        self.kernel_function = kernel_function

        self.kernel = kernel_function(x_train, x_train)

    def __getitem__(self, item):
        return self.kernel.__getitem__(item)
