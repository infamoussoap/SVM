import numpy as np
import warnings
cimport cython


cdef class CythonSMO:
    cdef readonly double[:] y_train, alphas, cached_errors
    cdef readonly double b
    cdef double[:, :] kernel
    cdef double C, alpha_tol, error_tol
    cdef int random_seed

    def __init__(self, double C=1.0, double alpha_tol=1e-2, double error_tol=1e-2,
                 int random_seed=-1):
        self.C = C

        self.alpha_tol = alpha_tol
        self.error_tol = error_tol

        self.random_seed = random_seed
        if random_seed >= 0:
            np.random.seed(random_seed)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    def cython_objective_function(double[:] y_train, double[:] alphas, double[:, :] kernel):
        cdef double total = 0.0
        cdef double alpha_sum = 0.0
        cdef int N, i, j, k

        N = y_train.shape[0]

        for i in range(N):
            for j in range(i + 1):
                if i == j:
                    # Diagonal Elements
                    total = total + y_train[i] * y_train[j] * alphas[i] * alphas[j] * kernel[i, j]
                else:
                    # Since the kernel is symmetric: bottom half = top half. So just multiply by 2
                    total = total + 2 * (y_train[i] * y_train[j] * alphas[i] * alphas[j] * kernel[i, j])

        total = total * 0.5

        for i in range(N):
            total = total - alphas[i]

        return total

    def initialize_attributes(self, x_train, double[:] y_train, kernel_function):
        self.y_train = np.zeros(y_train.shape[0], dtype=np.float64)
        self.y_train[:] = y_train

        self.kernel = kernel_function(x_train, x_train).astype(np.float64)

        self.alphas = np.zeros(y_train.shape[0], dtype=np.float64)
        self.b = 0.0

        self.cached_errors = np.zeros(y_train.shape[0], dtype=np.float64) - y_train

    def optimize(self, x_train, double[:] y_train, kernel_function, max_iter=1000):
        self.initialize_attributes(x_train, y_train, kernel_function)

        num_changed = 0
        examine_all = True
        count = 0

        history = [np.sum(np.asarray(self.cached_errors) ** 2)]

        while (num_changed > 0 or examine_all) and (count < max_iter):
            if examine_all:
                num_changed = self.examine_all_lagrange_multipliers()
            else:
                num_changed = self.examine_all_non_zero_and_non_c_lagrange_multipliers()

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True

            history.append(np.sum(np.asarray(self.cached_errors) ** 2))
            count += 1

        if count == max_iter:
            warnings.warn("Max iterations reached and convergence is not guaranteed.")

        return history

    cdef examine_all_lagrange_multipliers(self):
        """ Examines each lagrange multiplier sequentially """
        cdef int num_changed = 0
        cdef int j

        for j in range(len(self.alphas)):
            num_changed += self.examine_example(j)
        return num_changed

    cdef examine_all_non_zero_and_non_c_lagrange_multipliers(self):
        """ Only examines the lagrange multipliers alpha_i such that 0 < alpha_i < C """
        cdef int num_changed = 0
        for j, alpha in enumerate(self.alphas):
            is_non_zero_and_non_c = 0 < alpha < self.C
            if is_non_zero_and_non_c:
                num_changed += self.examine_example(j)
        return num_changed

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef examine_example(self, int j):
        cdef double y2 = self.y_train[j]
        cdef double alpha2 = self.alphas[j]
        cdef double E2 = self.cached_errors[j]
        cdef double r2 = E2 * y2

        if (r2 < -self.error_tol and alpha2 < self.C) or (r2 > self.error_tol and alpha2 > 0):
            N = len(self.alphas)
            non_zero_and_non_c_alpha = (np.asarray(self.alphas) > 0.0) & (np.asarray(self.alphas) < self.C)

            # Take step based on heuristic
            if np.sum(non_zero_and_non_c_alpha) > 1:
                i = self.arg_with_maximum_distance(j)
                if self.take_step(i, j):
                    return True

            # Take step on non-zero and non-c values
            non_zero_and_non_c_indices = np.argwhere(non_zero_and_non_c_alpha).flatten().astype(np.int32)
            step_taken = self.take_step_over_indices(j, non_zero_and_non_c_indices)
            if step_taken:
                return True

            # Take step on everything else
            other_indices = np.argwhere(~non_zero_and_non_c_alpha).flatten().astype(np.int32)
            step_taken = self.take_step_over_indices(j, other_indices)
            if step_taken:
                return True

        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arg_with_maximum_distance(self, int j):
        """ Returns index i where it solves the problem
                argmax_i |svm_errors[j] - svm_errors[i]|
        """
        cdef double E2 = self.cached_errors[j]
        if E2 > 0:
            return np.argmin(self.cached_errors)
        return np.argmax(self.cached_errors)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef take_step_over_indices(self, int j, int[:] indices):
        shuffled_indices = np.asarray(indices).copy()
        np.random.shuffle(shuffled_indices)

        cdef int N = shuffled_indices.shape[0]
        cdef int k

        for k in range(N):
            if self.take_step(shuffled_indices[k], j):
                return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef take_step(self, int i, int j):
        if i == j:
            return False

        cdef double alpha1 = self.alphas[i], alpha2 = self.alphas[j]
        cdef double y1 = self.y_train[i], y2 = self.y_train[j]
        cdef double E1 = self.cached_errors[i], E2 = self.cached_errors[j]

        cdef double s = y1 * y2
        cdef double L, H
        L, H = CythonSMO.get_bounds_for_lagrange_multipliers(alpha1, alpha2, y1, y2, self.C)
        if L == H:
            return False

        cdef double k11 = self.kernel[i, i]
        cdef double k12 = self.kernel[i, j]
        cdef double k22 = self.kernel[j, j]
        cdef double eta = k11 + k22 - 2 * k12

        cdef double a1, a2
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            a2 = np.clip(a2, L, H)
        else:
            a2 = self.get_new_alpha2_with_negative_eta(j, L, H, alpha2)
            print("Get new alpha2")

        if a2 < 1e-8:
            a2 = 0.0
        elif a2 > (self.C - 1e-8):
            a2 = self.C

        if abs(a2 - alpha2) < self.alpha_tol * (a2 + alpha2 + self.alpha_tol):
            return False

        a1 = alpha1 + s * (alpha2 - a2)

        cdef double b1, b2
        b1 = E1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12 + self.b
        b2 = E2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22 + self.b
        b_new = CythonSMO.get_new_threshold(a1, a2, b1, b2, self.C)

        self.cached_errors = np.asarray(self.cached_errors) \
                             + y1 * (a1 - alpha1) * np.asarray(self.kernel[i, :]) \
                             + y2 * (a2 - alpha2) * np.asarray(self.kernel[j, :]) \
                             - (b_new - self.b)

        self.alphas[i] = a1
        self.alphas[j] = a2
        self.b = b_new

        for index, alpha in zip([i, j], [a1, a2]):
            if 0 < alpha < self.C:
                self.cached_errors[index] = 0.0

        return True

    @staticmethod
    cdef get_bounds_for_lagrange_multipliers(double a1, double a2, double y1, double y2, double C):
        if y1 != y2:
            L = max(0, a2 - a1)
            H = min(C, C + a2 - a1)
        else:
            L = max(0, a2 + a1 - C)
            H = min(C, a2 + a1)

        return L, H

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef get_new_alpha2_with_negative_eta(self, int j, double L, double H, double alpha2):
        alphas_adj = np.asarray(self.alphas).copy()

        alphas_adj[j] = L
        L_obj = CythonSMO.cython_objective_function(self.y_train, alphas_adj, self.kernel)

        alphas_adj[j] = H
        H_obj = CythonSMO.cython_objective_function(self.y_train, alphas_adj, self.kernel)

        if L_obj < H_obj - self.alpha_tol:
            return L
        elif L_obj > H_obj + self.alpha_tol:
            return H
        else:
            return alpha2

    @staticmethod
    cdef get_new_threshold(double a1, double a2, double b1, double b2, double C):
        if 0 < a1 < C:
            return b1
        elif 0 < a2 < C:
            return b2

        return (b1 + b2) / 2
