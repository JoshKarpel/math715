__author__ = 'Josh Karpel'

from utils import *

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.getcwd(), 'out')

GRID_KWARGS = {
    'linestyle': '-',
    'color': 'black',
    'linewidth': .25,
    'alpha': 0.4
}


class BSplineBasis:
    def __init__(self, knot_vector, polynomial_order, parameter_space_points = 1000):
        self.knot_vector = np.array(knot_vector)
        self.basis_function_indices = np.array(range(len(knot_vector) - polynomial_order - 1))
        self.polynomial_order = polynomial_order

        self.parameter_space_points = parameter_space_points

    @property
    def name(self):
        return f'p={self.polynomial_order}_k=[{",".join(str(s) for s in self.knot_vector)}]'

    @property
    def xi_min(self):
        return self.knot_vector[0]

    @property
    def xi_max(self):
        return self.knot_vector[-1]

    @property
    def xi(self):
        return np.linspace(self.xi_min, self.xi_max, self.parameter_space_points)

    @memoize
    def basis_null(self, basis_function_index):
        """Return the zero-order basis function evaluated along the parameter space for a given basis function index."""
        return np.where(np.greater_equal(self.xi, self.knot_vector[basis_function_index]) * np.less(self.xi, self.knot_vector[basis_function_index + 1]),
                        1.0, 0.0)

    @memoize
    def basis_function(self, basis_function_index, polynomial_order):
        """Return the p = polynomial_order basis function evaluated along the parameter space for a given basis function index."""
        if polynomial_order == 0:  # base case
            return self.basis_null(basis_function_index)
        else:
            # recursion formula from Hughes et. al. 2004, p. 4140
            first_num = self.xi - self.knot_vector[basis_function_index]
            first_den = self.knot_vector[basis_function_index + polynomial_order] - self.knot_vector[basis_function_index]
            first_basis = self.basis_function(basis_function_index, polynomial_order - 1)

            second_num = self.knot_vector[basis_function_index + polynomial_order + 1] - self.xi
            second_den = self.knot_vector[basis_function_index + polynomial_order + 1] - self.knot_vector[basis_function_index + 1]
            second_basis = self.basis_function(basis_function_index + 1, polynomial_order - 1)

            with np.errstate(divide = 'ignore', invalid = 'ignore'):  # ignore divide by zero errors, the np.where calls bypass them
                first_term = np.where(np.not_equal(first_den, 0), first_num * first_basis / first_den, 0)
                second_term = np.where(np.not_equal(second_den, 0), (second_num * second_basis / second_den), 0)

            return first_term + second_term

    def plot_basis_functions(self, **kwargs):
        fig = get_figure('full')
        ax = fig.add_subplot(111)

        for basis_index in self.basis_function_indices:
            y = self.basis_function(basis_index, self.polynomial_order)  # get basis function
            y = ma.masked_equal(y, 0)  # mask zeros so they won't be displayed in the plot

            ax.plot(self.xi, y, label = rf'$N_{{ {basis_index}, {self.polynomial_order} }} $')

        ax.set_xlim(self.xi_min, self.xi_max)
        ax.set_ylim(0, 1.01)

        ax.set_xlabel(r'$\xi$', fontsize = 12)
        ax.set_ylabel(r'$N_{i,p}(\xi)$', fontsize = 12)
        ax.set_title(fr'Basis Functions for $\Xi = \left[ {",".join(str(s) for s in self.knot_vector)} \right]$, $p = {self.polynomial_order}$')

        ax.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left', borderaxespad = 0., fontsize = 12, ncol = 1 + (len(self.basis_function_indices) // 15))

        ax.grid(True, **GRID_KWARGS)

        save_current_figure(name = self.name + '__basis', target_dir = OUT_DIR, **kwargs)

        plt.close()


if __name__ == '__main__':
    bsplines = (
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 0),
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 1),
        BSplineBasis(knot_vector = [0, 0, 1, 2, 3, 4, 5, 5], polynomial_order = 1),
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 2),
        BSplineBasis(knot_vector = [0, 0, 1, 2, 3, 4, 5], polynomial_order = 2),
        BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2),
    )

    for bspline in bsplines:
        print(bspline.name)

        print('knot vector:', bspline.knot_vector)
        print('number of basis functions:', len(bspline.basis_function_indices))
        print('polynomial order:', bspline.polynomial_order)

        print('number of knots:', len(bspline.knot_vector))
        print('# of basis functions + polynomial order + 1:', len(bspline.basis_function_indices) + bspline.polynomial_order + 1)

        bspline.plot_basis_functions()

        print('-' * 80)
