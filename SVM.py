from SMO import SMO


class SVM:
    def __init__(self, kernel_function, C=1):
        self.kernel_function = kernel_function
        self.C = C

        self.alphas = None
        self.b = None

        self.x_train = None
        self.y_train = None

        self.fitted = False

    def fit(self, x_train, y_train, max_iter=100):
        self.x_train = x_train
        self.y_train = y_train

        optimizer = SMO(x_train, y_train, self.kernel_function, C=self.C)
        alphas, b = optimizer.optimize(max_iter=max_iter)

        self.alphas = alphas
        self.b = b

    def predict(self, x_new):
        kernel = self.kernel_function(self.x_train, x_new)
        return (self.y_train * self.alphas) @ kernel - self.b
