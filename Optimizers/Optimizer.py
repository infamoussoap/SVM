import abc


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def __init__(self, C, alpha_tol, error_tol, *args, **kwargs):
        """ Initialize Optimizer

            Parameters
            ----------
            kernel : np.ndarray (or array like that implements __getitem__)
            y_trian : np.ndarray
            C : float
            alpha_tol : float
                Error tolerance for lagrange multipliers
            error_tol : float
                Error tolerance for model residuals
        """
        pass

    @abc.abstractmethod
    def get_config(self):
        """ Returns the variables (as a dictionary) required to create a new
            instance of this class

            Returns
            -------
            dict
        """
        pass

    @abc.abstractmethod
    def optimize(self, x_train, y_train, kernel_function, *args, **kwargs):
        """ Find the optimal lagrange multipliers

            Parameters
            ----------
            x_train : (n, ...) np.ndarray
            y_train : (n,) np.ndarray
            kernel_function : function or callable
        """
