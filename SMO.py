import numpy as np
import warnings


class SMO:
    def __init__(self, C=1.0, alpha_tol=1e-2, error_tol=1e-2):
        """ Initialize sequential minimal optimization """

        self.C = C

        self.alpha_tol = alpha_tol
        self.error_tol = error_tol

        # To be initialized later
        self.y_train = None
        self.kernel = None

        self.alphas = None
        self.b = None

        self.cached_errors = None

    def get_config(self):
        """ Returns the variables required to create a new instance of this class """
        return {'C': self.C,
                'alpha_tol': self.alpha_tol,
                'error_tol': self.error_tol}

    @staticmethod
    def objective_function(y_train, alphas, kernel):
        return 0.5 * np.sum(kernel * y_train[:, None] * y_train[None, :] * alphas[:, None] * alphas[None, :]) \
               - np.sum(alphas)

    def initialize_attributes(self, x_train, y_train, kernel_function):
        self.y_train = y_train
        self.kernel = kernel_function(x_train, x_train)

        self.alphas = np.zeros(len(x_train))
        self.b = 0.0

        # Coefficients are all 0 in the beginning, so error is just -y_train
        self.cached_errors = -y_train

    def optimize(self, x_train, y_train, kernel_function, max_iter=1000):
        self.initialize_attributes(x_train, y_train, kernel_function)

        num_changed = 0
        examine_all = True
        count = 0

        history = [np.sum(self.cached_errors ** 2)]

        while (num_changed > 0 or examine_all) and (count < max_iter):
            if examine_all:
                num_changed = self.examine_all_lagrange_multipliers()
            else:
                num_changed = self.examine_all_non_zero_and_non_c_lagrange_multipliers()

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            history.append(np.sum(self.cached_errors ** 2))
            count += 1

        if count == max_iter:
            warnings.warn("Max iterations reached and convergence is not guaranteed.")

        return history

    def examine_all_lagrange_multipliers(self):
        """ Examines each lagrange multiplier sequentially """
        num_changed = 0
        for j in range(len(self.alphas)):
            num_changed += self.examine_example(j)
        return num_changed

    def examine_all_non_zero_and_non_c_lagrange_multipliers(self):
        """ Only examines the lagrange multipliers alpha_i such that 0 < alpha_i < C """
        num_changed = 0
        for j, alpha in enumerate(self.alphas):
            is_non_zero_and_non_c = 0 < alpha < self.C
            if is_non_zero_and_non_c:
                num_changed += self.examine_example(j)
        return num_changed

    def examine_example(self, j):
        y2 = self.y_train[j]
        alpha2 = self.alphas[j]
        E2 = self.cached_errors[j]
        r2 = E2 * y2

        if (r2 < -self.error_tol and alpha2 < self.C) or (r2 > self.error_tol and alpha2 > 0):
            N = len(self.alphas)
            non_zero_and_non_c_alpha = (self.alphas > 0) & (self.alphas < self.C)

            # Take step based on heuristic
            if np.sum(non_zero_and_non_c_alpha) > 1:
                i = self.arg_with_maximum_distance(j)
                if self.take_step(i, j):
                    return True

            # Take step on non-zero and non-c values
            non_zero_and_non_c_indices = np.argwhere(non_zero_and_non_c_alpha).flatten()
            step_taken = self.take_step_over_indices(j, non_zero_and_non_c_indices)
            if step_taken:
                return True

            # Take step on everything else
            other_indices = np.argwhere(non_zero_and_non_c_alpha).flatten()
            step_taken = self.take_step_over_indices(j, other_indices)
            if step_taken:
                return True

        return False

    def arg_with_maximum_distance(self, j):
        """ Returns index i where it solves the problem
                max_i |svm_errors[j] - svm_errors[i]|
        """
        E2 = self.cached_errors[j]
        if E2 > 0:
            return np.argmin(self.cached_errors)
        return np.argmax(self.cached_errors)

    def take_step_over_indices(self, j, indices):
        shuffled_indices = indices.copy()
        np.random.shuffle(shuffled_indices)

        for i in shuffled_indices:
            if self.take_step(i, j):
                return True
        return False

    def take_step(self, i, j):
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
            a2 = self.get_new_alpha2_with_negative_eta(j, L, H, alpha2)

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

        self.cached_errors = self.cached_errors + y1 * (a1 - alpha1) * self.kernel[i, :] \
                             + y2 * (a2 - alpha2) * self.kernel[j, :] - (b_new - self.b)

        self.alphas[i] = a1
        self.alphas[j] = a2
        self.b = b_new

        for index, alpha in zip([i, j], [a1, a2]):
            if 0 < alpha < self.C:
                self.cached_errors[index] = 0.0

        return True

    def get_new_alpha2_with_negative_eta(self, j, L, H, alpha2):
        alphas_adj = self.alphas.copy()

        alphas_adj[j] = L
        L_obj = self.objective_function(self.y_train, alphas_adj, self.kernel)

        alphas_adj[j] = H
        H_obj = self.objective_function(self.y_train, alphas_adj, self.kernel)

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
