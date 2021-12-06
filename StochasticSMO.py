import numpy as np
from functools import lru_cache
import warnings

from Kernels import get_kernel


class StochasticSMO:
    def __init__(self, C=1.0, alpha_tol=1e-2, error_tol=1e-2, kernel_type="stock", batch_size=128):
        """ Initialize sequential minimal optimization

            Parameters
            ----------
            C : float, optional
                The regularization strength
            alpha_tol : float, optional
                Tolerance for lagrange multipliers
            error_tol : float, optional
                Tolerance for objective function
            kernel_type : str, optional
                Type of kernel, either ["stock", "lazy", "disk"]
                    stock - The kernel is saved to RAM
                    lazy - The kernel is never saved, and is always computed when required
                    disk - The kernel is saved to disk
            batch_size : int, optional
                This parameter is ignored when kernel_type="stock"
        """
        self.C = C

        self.alpha_tol = alpha_tol
        self.error_tol = error_tol

        self.kernel_type = kernel_type
        self._kernel = None

        self.batch_size = batch_size

        # To be initialized later
        self.x_train = None
        self.y_train = None
        self.kernel_function = None

        self.alphas = None
        self.b = None

        self.cached_errors = None

    def get_config(self):
        """ Returns the variables required to create a new instance of this class """
        return {'C': self.C,
                'alpha_tol': self.alpha_tol,
                'error_tol': self.error_tol,
                'kernel_type': self.kernel_type,
                'batch_size': self.batch_size}

    def initialize_attributes(self, x_train, y_train, kernel_function):
        self.y_train = y_train
        self.x_train = x_train

        self.kernel_function = kernel_function

        self.alphas = np.zeros(len(x_train))
        self.b = 0.0

        # Coefficients are all 0 in the beginning, so error is just -y_train
        self.cached_errors = -y_train

    @property
    def kernel(self):
        if self._kernel is None:
            kernel_class = get_kernel(self.kernel_type)
            self._kernel = kernel_class(self.x_train, self.kernel_function)

        return self._kernel

    @staticmethod
    def objective_function(y_train, alphas, kernel, batch_indices, eps=1e-7):
        non_zero_alphas = np.argwhere(alphas > eps)
        non_zero_batch_alphas = np.argwhere(alphas[batch_indices] > eps)

        # No need to compute the kernel where alpha is 0
        batched_kernel = np.zeros((len(y_train), len(batch_indices)))
        if len(non_zero_alphas) > 0 and len(non_zero_batch_alphas) > 0:
            batched_kernel[non_zero_alphas, non_zero_batch_alphas] = kernel[non_zero_alphas, non_zero_batch_alphas]

        batched_y = y_train[:, None] * y_train[None, batch_indices]
        batched_alpha = alphas[:, None] * alphas[None, batch_indices]

        return 0.5 * np.sum(batched_kernel * batched_y * batched_alpha) - np.sum(alphas)

    def optimize(self, x_train, y_train, kernel_function, max_iter=1000):
        self.initialize_attributes(x_train, y_train, kernel_function)

        num_changed = 0
        examine_all = True
        count = 0

        num_batches = np.ceil(len(self.y_train) / self.batch_size).astype(int)
        indices = np.arange(len(self.y_train))

        history = [np.sum(self.cached_errors ** 2)]

        while (num_changed > 0 or examine_all) and (count < max_iter):
            np.random.shuffle(indices)
            num_changed = 0

            for batch_num in range(num_batches):
                start_index = batch_num * self.batch_size
                end_index = (batch_num + 1) * self.batch_size
                batch_indices = indices[start_index: end_index]

                num_changed += self.examine_batch(batch_indices, examine_all)

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            history.append(np.sum(self.cached_errors ** 2))
            count += 1

        if count == max_iter:
            warnings.warn("Max iterations reached and convergence is not guaranteed.")

        return history

    def examine_batch(self, batch_indices, examine_all):
        if examine_all:
            num_changed = self.examine_all_lagrange_multipliers(batch_indices)
        else:
            num_changed = self.examine_all_non_zero_and_non_c_lagrange_multipliers(batch_indices)
        return num_changed

    def examine_all_lagrange_multipliers(self, batch_indices):
        """ Examines each lagrange multiplier sequentially """
        num_changed = 0
        for j in range(len(self.alphas)):
            num_changed += self.examine_example(j, batch_indices)
        return num_changed

    def examine_all_non_zero_and_non_c_lagrange_multipliers(self, batch_indices):
        """ Only examines the lagrange multipliers alpha_i such that 0 < alpha_i < C """
        num_changed = 0
        for j, alpha in enumerate(self.alphas):
            is_non_zero_and_non_c = 0 < alpha < self.C
            if is_non_zero_and_non_c:
                num_changed += self.examine_example(j, batch_indices)
        return num_changed

    def examine_example(self, j, batch_indices):
        y2 = self.y_train[j]
        alpha2 = self.alphas[j]
        E2 = self.cached_errors[j]
        r2 = E2 * y2

        if (r2 < -self.error_tol and alpha2 < self.C) or (r2 > self.error_tol and alpha2 > 0):
            batch_alphas = self.alphas[batch_indices]
            non_zero_and_non_c_alpha = (batch_alphas > 0) & (batch_alphas < self.C)

            # Take step based on heuristic
            if np.sum(non_zero_and_non_c_alpha) > 1:
                i = self.arg_with_maximum_distance(j, batch_indices)
                if self.take_step(i, j, batch_indices):
                    return True

            # Take step on non-zero and non-c values
            # batch_indices is already shuffled, so no need to re-shuffle
            non_zero_and_non_c_indices = batch_indices[non_zero_and_non_c_alpha]
            step_taken = self.take_step_over_indices(j, non_zero_and_non_c_indices, batch_indices)
            if step_taken:
                return True

            # Take step on everything else
            other_indices = batch_indices[~non_zero_and_non_c_alpha]
            step_taken = self.take_step_over_indices(j, other_indices, batch_indices)
            if step_taken:
                return True

        return False

    def arg_with_maximum_distance(self, j, batch_indices):
        """ Returns index i where it solves the problem
                max_i |svm_errors[j] - svm_errors[i]|
        """
        E2 = self.cached_errors[j]
        batched_errors = self.cached_errors[batch_indices]

        if E2 > 0:
            return batch_indices[np.argmin(batched_errors)]
        return batch_indices[np.argmax(batched_errors)]

    def take_step_over_indices(self, j, shuffled_working_indices, batch_indices):
        for i in shuffled_working_indices:
            if self.take_step(i, j, batch_indices):
                return True
        return False

    def take_step(self, i, j, batch_indices):
        if i == j:
            return False

        alpha1, alpha2 = self.alphas[i], self.alphas[j]
        y1, y2 = self.y_train[i], self.y_train[j]
        E1, E2 = self.cached_errors[i], self.cached_errors[j]

        s = y1 * y2
        L, H = self.get_bounds_for_lagrange_multipliers(alpha1, alpha2, y1, y2, self.C)
        if L == H:
            return False

        k11 = self.kernel[i, i]
        k12 = self.kernel[i, j]
        k22 = self.kernel[j, j]
        eta = k11 + k22 - 2 * k12

        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
        else:
            a2 = self.get_new_alpha2_with_negative_eta(j, L, H, alpha2, batch_indices)

        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        if abs(a2 - alpha2) < self.alpha_tol * (a2 + alpha2 + self.alpha_tol):
            return False

        a1 = alpha1 + s * (alpha2 - a2)

        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b
        b_new = self.get_new_threshold(a1, a2, b1, b2, self.C)

        # Updated Cached Errors
        self.cached_errors = self.cached_errors + y1 * (a1 - alpha1) * self.kernel[i, :] \
                             + y2 * (a2 - alpha2) * self.kernel[j, :] - (b_new - self.b)

        for index, alpha in zip([i, j], [a1, a2]):
            if 0 < alpha < self.C:
                self.cached_errors[index] = 0.0

        # Update Parameters
        self.alphas[i] = a1
        self.alphas[j] = a2
        self.b = b_new

        return True

    def get_new_alpha2_with_negative_eta(self, j, L, H, alpha2, batch_indices):
        alphas_adj = self.alphas.copy()

        alphas_adj[j] = L
        L_obj = self.objective_function(self.y_train, alphas_adj, self.kernel, batch_indices)

        alphas_adj[j] = H
        H_obj = self.objective_function(self.y_train, alphas_adj, self.kernel, batch_indices)

        if L_obj < H_obj - self.alpha_tol:
            return L
        elif L_obj > H_obj + self.alpha_tol:
            return H
        else:
            return alpha2

    @staticmethod
    def get_new_threshold(a1, a2, b1, b2, C):
        if 0 < a1 < C:
            return b1
        elif 0 < a2 < C:
            return b2

        return (b1 + b2) / 2

    @staticmethod
    def get_bounds_for_lagrange_multipliers(a1, a2, y1, y2, C):
        if y1 != y2:
            L = max(0, a2 - a1)
            H = min(C, C + a2 - a1)
        else:
            L = max(0, a2 + a1 - C)
            H = min(C, a2 + a1)

        return L, H
