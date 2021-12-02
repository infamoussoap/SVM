from SMO import SMO
from StochasticSMO import StochasticSMO


def get_optimizer(optimizer):
    if isinstance(optimizer, (SMO, StochasticSMO)):
        return optimizer
    elif optimizer.lower() == 'stochasticsmo':
        return StochasticSMO()
    elif optimizer.lower() == 'smo':
        return SMO()
    else:
        raise ValueError(f"{optimizer} is not a valid optimizer.")


class SVM:
    def __init__(self, kernel_function, C=1, optimizer='StochasticSMO'):
        self.optimizer = get_optimizer(optimizer)

        self.kernel_function = kernel_function
        self.C = C

        self.alphas = None
        self.b = None

        self.x_train = None
        self.y_train = None

        self.fitted = False

        self.history = None

    def fit(self, x_train, y_train, max_iter=100):
        self.x_train = x_train
        self.y_train = y_train

        self.history = self.optimizer.optimize(x_train, y_train, self.kernel_function, max_iter=max_iter)

        self.alphas = self.optimizer.alphas
        self.b = self.optimizer.b

    def predict(self, x_new):
        kernel = self.kernel_function(self.x_train, x_new)
        return (self.y_train * self.alphas) @ kernel - self.b
