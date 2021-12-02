import numpy as np

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

    def predict(self, x_new, predict_as_batch=False, batch_size=128):
        """ Return the prediction on the given dataset

            Parameters
            ----------
            x_new : np.array
            predict_as_batch : bool, optional
                If true then prediction will be computed on batches of the dataset. The full prediction will be
                compile the predictions on the batches of dataset

                Note - This should really only be used if the data size is too big, which will result in a huge
                       kernel matrix
            batch_size : int, optional
                If predict_as_batch is False, this parameter is ignored
                The batch_size the prediction will be computed on
        """
        if predict_as_batch:
            return self.predict_as_batches(x_new, batch_size=batch_size)

        kernel = self.kernel_function(self.x_train, x_new)
        return (self.y_train * self.alphas) @ kernel - self.b

    def predict_as_batches(self, x_new, batch_size=128):
        num_batch = np.ceil(len(x_new) / batch_size).astype(int)

        batch_predictions = []
        for i in range(num_batch):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size

            predictions = self.predict(x_new[start_index: end_index], predict_as_batch=False)
            batch_predictions.append(predictions)

        return np.concatenate(batch_predictions)
