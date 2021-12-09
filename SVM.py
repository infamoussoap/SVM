import numpy as np
from Optimizers import get_new_instance_of_optimizer


class SVM:
    """ Support Vector Machine for Multiclass Classification using One-vs-All rule """
    def __init__(self, kernel_function, C=1, optimizer='StochasticSMO'):
        self.optimizer = optimizer

        self.kernel_function = kernel_function
        self.C = C

        self.x_train = None
        self.y_train = None

        self.fitted = False

        self.history = None

        self.models = []

        self.unique_labels = None

    def fit(self, x_train, y_train, max_iter=100):
        self.x_train = x_train
        self.y_train = y_train

        self.unique_labels = np.unique(y_train)
        if len(self.unique_labels) == 2:
            self._fit_for_binary_classification(x_train, y_train, max_iter=max_iter)
        else:
            self._fit_for_multiclass_classification(x_train, y_train, max_iter=max_iter)

        self.fitted = True

    def _fit_for_binary_classification(self, x_train, y_train, max_iter=100):
        """ This performs classifications when it is only a binary classification problem.
            Unlike the multiclass classification, no sampling is performed here
        """
        positive_label = self.unique_labels[1]

        self.models = [SVMBinaryClassification(self.kernel_function, self.C, self.optimizer)]

        positive_class_indices = np.argwhere(y_train == positive_label).flatten()
        negative_class_indices = np.argwhere(y_train != positive_label).flatten()

        positive_class_x_train = x_train[positive_class_indices]
        negative_class_x_train = x_train[negative_class_indices]

        combined_class_x_train = np.concatenate([positive_class_x_train, negative_class_x_train])
        combined_class_y_train = np.concatenate([np.ones(len(positive_class_indices)),
                                                 -np.ones(len(negative_class_indices))])

        self.models[0].fit(combined_class_x_train, combined_class_y_train, max_iter=max_iter)
        self.history = self.models[0].history

    def _fit_for_multiclass_classification(self, x_train, y_train, max_iter=100):
        """ Multiclass Classifications samples the negative class to be the same length
            as the positive class, so there is no class imbalance
        """
        for _ in range(len(self.unique_labels)):
            self.models.append(SVMBinaryClassification(self.kernel_function, self.C, self.optimizer))

        self.history = []
        for label, model in zip(self.unique_labels, self.models):
            n = np.sum(y_train == label)

            positive_class_indices = np.random.choice(np.argwhere(y_train == label).flatten(), n, replace=False)
            negative_class_indices = np.random.choice(np.argwhere(y_train != label).flatten(), n, replace=False)

            positive_class_x_train = x_train[positive_class_indices]
            negative_class_x_train = x_train[negative_class_indices]

            combined_class_x_train = np.concatenate([positive_class_x_train, negative_class_x_train])
            combined_class_y_train = np.concatenate([np.ones(n), -np.ones(n)])

            self.history.append(model.fit(combined_class_x_train, combined_class_y_train, max_iter=max_iter))

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
        if len(self.unique_labels) == 2:
            return self._predict_for_binary_classification(x_new, predict_as_batch, batch_size)
        else:
            return self._predict_for_multiclass_classification(x_new, predict_as_batch, batch_size)

    def _predict_for_binary_classification(self, x_new, predict_as_batch=False, batch_size=128):
        model_predictions = self.models[0].predict(x_new, predict_as_batch, batch_size)
        predicted_labels = (model_predictions > 0).astype(int)

        return self.unique_labels[predicted_labels]

    def _predict_for_multiclass_classification(self, x_new, predict_as_batch=False, batch_size=128):
        model_predictions = np.array([model.predict(x_new, predict_as_batch, batch_size)
                                      for model in self.models])
        predicted_labels = np.argmax(model_predictions, axis=0)

        return self.unique_labels[predicted_labels]

    @property
    def alphas(self):
        if len(self.unique_labels) == 2:
            return self.models[0].alphas
        return [model.alphas for model in self.models]

    @property
    def b(self):
        if len(self.unique_labels) == 2:
            return self.models[0].b
        return [model.b for model in self.models]


class SVMBinaryClassification:
    """ Support Vector Machine for Binary Classification """
    def __init__(self, kernel_function, C=1, optimizer='StochasticSMO'):
        self.optimizer = get_new_instance_of_optimizer(optimizer)

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

        # No need to evaluate the kernel when alpha_i = 0
        kernel = np.zeros((len(self.alphas), len(x_new)))

        non_zero_alphas = self.alphas > 0
        if np.sum(non_zero_alphas) > 0:
            kernel[non_zero_alphas, :] = self.kernel_function(self.x_train[non_zero_alphas], x_new)
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
