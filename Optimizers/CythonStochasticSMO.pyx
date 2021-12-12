# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
# cython: profile=True

import numpy as np
import warnings
cimport cython
from cython.parallel import prange

from Optimizers.TypedKernels import DiskKernelV2


cdef clip(double a, double min_val, double max_val):
    if a <= min_val:
        return min_val
    elif a >= max_val:
        return max_val
    return a


cdef class CythonStochasticSMO:
    cdef:
        readonly double[:] y_train, alphas, cached_errors
        readonly double b
        double C, alpha_tol, error_tol
        double sleep_time
        int random_seed, batch_size
        object kernel

    def __init__(self, double C=1.0, double alpha_tol=1e-2, double error_tol=1e-2,
                 int random_seed=-1, int batch_size=128, double sleep_time=0.01):
        self.C = C

        self.alpha_tol = alpha_tol
        self.error_tol = error_tol

        self.batch_size = batch_size

        self.sleep_time = sleep_time

        self.random_seed = random_seed
        if random_seed >= 0:
            np.random.seed(random_seed)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    def cython_objective_function(double[:] y_train, double[:] alphas, kernel, int[:] batch_indices):
        cdef double total = 0.0
        cdef double alpha_sum = 0.0
        cdef int N, batch_length, i, j, k

        batch_length = batch_indices.shape[0]
        N = y_train.shape[0]

        cdef double[:] kernel_i, kernel_j

        for i in range(batch_length):
            kernel_i = kernel.get_row(i)
            for j in range(N):
                total = total + y_train[i] * y_train[j] * alphas[i] * alphas[j] * kernel_i[j]

        total = total * 0.5

        for i in range(N):
            total = total - alphas[i]

        return total

    def initialize_attributes(self, x_train, double[:] y_train, kernel_function):
        self.y_train = np.zeros(y_train.shape[0], dtype=np.float64)
        self.y_train[:] = y_train

        self.kernel = DiskKernelV2(x_train, kernel_function)

        self.alphas = np.zeros(y_train.shape[0], dtype=np.float64)
        self.b = 0.0

        self.cached_errors = np.zeros(y_train.shape[0], dtype=np.float64) - y_train

    def optimize(self, x_train, double[:] y_train, kernel_function, max_iter=1000):
        self.initialize_attributes(x_train, y_train, kernel_function)

        cdef int num_changed = 0, count = 0, batch_num
        cdef bint examine_all = True

        cdef int num_batches = np.ceil(len(self.y_train) / self.batch_size).astype(np.int32)
        indices = np.arange(len(self.y_train)).astype(np.int32)

        history = [np.sum(np.asarray(self.cached_errors) ** 2)]

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

            history.append(np.sum(np.asarray(self.cached_errors) ** 2))
            count += 1

        if count == max_iter:
            warnings.warn("Max iterations reached and convergence is not guaranteed.")

        self.kernel.remove_file()

        return history

    def examine_batch(self, batch_indices, examine_all):
        if examine_all:
            num_changed = self.examine_all_lagrange_multipliers(batch_indices)
        else:
            num_changed = self.examine_all_non_zero_and_non_c_lagrange_multipliers(batch_indices)
        return num_changed

    cdef examine_all_lagrange_multipliers(self, int[:] batch_indices):
        """ Examines each lagrange multiplier sequentially """
        cdef int num_changed = 0
        cdef int j

        for j in range(len(self.alphas)):
            num_changed += self.examine_example(j, batch_indices)
        return num_changed

    cdef examine_all_non_zero_and_non_c_lagrange_multipliers(self, int[:] batch_indices):
        """ Only examines the lagrange multipliers alpha_i such that 0 < alpha_i < C """
        cdef int num_changed = 0
        for j, alpha in enumerate(self.alphas):
            is_non_zero_and_non_c = 0 < alpha < self.C
            if is_non_zero_and_non_c:
                num_changed += self.examine_example(j, batch_indices)
        return num_changed

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef examine_example(self, int j, int[:] batch_indices):
        cdef double y2 = self.y_train[j]
        cdef double alpha2 = self.alphas[j]
        cdef double E2 = self.cached_errors[j]
        cdef double r2 = E2 * y2

        if (r2 < -self.error_tol and alpha2 < self.C) or (r2 > self.error_tol and alpha2 > 0):
            non_zero_and_non_c_alpha = self.number_of_support_vectors()

            # Take step based on heuristic
            if non_zero_and_non_c_alpha > 1:
                i = self.arg_with_maximum_distance(j, batch_indices)
                if self.take_step(i, j, batch_indices):
                    return True

            # Take step on non-zero and non-c values
            step_taken = self.take_step_over_non_zero_and_non_c_indices(j, batch_indices)
            if step_taken:
                return True

            # Take step on everything else
            step_taken = self.take_step_over_other_indices(j, batch_indices)
            if step_taken:
                return True

        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef number_of_support_vectors(self):
        cdef int N = self.alphas.shape[0]
        cdef int count = 0, i

        for i in range(N):
            alpha = self.alphas[i]
            count += (alpha > 0.0 and alpha < self.C)

        return count

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef arg_with_maximum_distance(self, int j, int[:] batch_indices):
        """ Returns index i where it solves the problem
                argmax_i |svm_errors[j] - svm_errors[i]| | for i in batch_indices
        """
        cdef double E2 = self.cached_errors[j]

        if E2 > 0:
            return self.get_argmin_of_cached_errors_at_batch_indices(batch_indices)
        return self.get_argmax_of_cached_errors_at_batch_indices(batch_indices)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef get_argmin_of_cached_errors_at_batch_indices(self, int[:] batch_indices):
        cdef int N = batch_indices.shape[0], n, i
        cdef int min_arg = 0
        cdef double min_val = batch_indices[0]

        for n in range(N):
            i = batch_indices[n]
            val = self.cached_errors[i]
            if val < min_val:
                min_val = val
                min_arg = i

        return min_arg

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef get_argmax_of_cached_errors_at_batch_indices(self, int[:] batch_indices):
        cdef int N = batch_indices.shape[0], n, i
        cdef int max_arg = 0
        cdef double max_val = batch_indices[0]

        for n in range(N):
            i = batch_indices[n]
            val = self.cached_errors[i]
            if val > max_val:
                max_val = val
                max_arg = i

        return max_arg

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef take_step_over_non_zero_and_non_c_indices(self, int j, int[:] batch_indices):
        """ A step is taken only if 0 < alpha_i < C """
        cdef int N = batch_indices.shape[0]
        cdef int k, i

        for k in range(N):
            i = batch_indices[k]
            alpha = self.alphas[i]

            if (alpha > 0.0) and (alpha < self.C):
                if self.take_step(i, j, batch_indices):
                    return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef take_step_over_other_indices(self, int j, int[:] batch_indices):
        """ A step is only taken if alpha_i = 0 or alpha_i = C """
        cdef int N = batch_indices.shape[0]
        cdef int k, i

        for k in range(N):
            i = batch_indices[k]
            alpha = self.alphas[i]

            if (alpha < 1e-7) or (alpha > self.C - 1e-7):
                if self.take_step(i, j, batch_indices):
                    return True
        return False

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef take_step(self, int i, int j, int[:] batch_indices):
        if i == j:
            return False

        cdef double alpha1 = self.alphas[i], alpha2 = self.alphas[j]
        cdef double y1 = self.y_train[i], y2 = self.y_train[j]
        cdef double E1 = self.cached_errors[i], E2 = self.cached_errors[j]

        cdef double s = y1 * y2
        cdef double L, H
        L, H = CythonStochasticSMO.get_bounds_for_lagrange_multipliers(alpha1, alpha2, y1, y2, self.C)
        if L == H:
            return False

        cdef double k11 = self.kernel.get_element(i, i)
        cdef double k12 = self.kernel.get_element(i, j)
        cdef double k22 = self.kernel.get_element(j, j)
        cdef double eta = k11 + k22 - 2 * k12

        cdef double a1, a2
        if eta > 0:
            a2 = alpha2 + y2 * (E1 - E2) / eta
            a2 = clip(a2, L, H)
        else:
            a2 = self.get_new_alpha2_with_negative_eta(j, L, H, alpha2, batch_indices)
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
        b_new = CythonStochasticSMO.get_new_threshold(a1, a2, b1, b2, self.C)

        cdef double c1 = y1 * (a1 - alpha1)
        cdef double c2 = y2 * (a2 - alpha2)
        self.update_cached_errors(c1, c2, b_new, i, j)

        self.alphas[i] = a1
        self.alphas[j] = a2
        self.b = b_new

        cdef int index
        cdef double alpha
        for index, alpha in zip([i, j], [a1, a2]):
            if 0 < alpha < self.C:
                self.cached_errors[index] = 0.0

        return True

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def update_cached_errors(self, double c1, double c2, double b_new, int i, int j):
        cdef int N = self.cached_errors.shape[0]
        cdef int k

        cdef double[:] kernel_i = self.kernel.get_row(i)
        cdef double[:] kernel_j = self.kernel.get_row(j)
        # cdef double[:] cached_errors = self.cached_errors

        for k in prange(N, nogil=True):
            self.cached_errors[k] = self.cached_errors[k] + c1 * kernel_i[k]\
                                    + c2 * kernel_j[k] - (b_new - self.b)


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
    cdef get_new_alpha2_with_negative_eta(self, int j, double L, double H, double alpha2, int[:] batch_indices):
        cdef double original_val = self.alphas[j]

        self.alphas[j] = L
        L_obj = CythonStochasticSMO.cython_objective_function(self.y_train, self.alphas, self.kernel, batch_indices)

        self.alphas[j] = H
        H_obj = CythonStochasticSMO.cython_objective_function(self.y_train, self.alphas, self.kernel, batch_indices)

        self.alphas[j] = original_val
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
