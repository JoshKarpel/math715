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
        return np.where(np.greater_equal(self.xi, self.knot_vector[basis_function_index]) * np.less_equal(self.xi, self.knot_vector[basis_function_index + 1]),
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

    def __iter__(self):
        yield from (self.basis_function(basis_index, self.polynomial_order) for basis_index in self.basis_function_indices)

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

        save_current_figure(name = self.name + '__basis', **kwargs)

        plt.close()


class BSplineCurve:
    def __init__(self, basis, control_points):
        self.basis = basis
        self.control_points = [np.array(control_point) for control_point in control_points]

    def curve(self):
        return sum(np.outer(basis_function, control_point) for basis_function, control_point in zip(self.basis, self.control_points)).T

    def plot_curve(self, **kwargs):
        """Only works in 2D..."""
        fig = get_figure('full')
        ax = fig.add_subplot(111)

        curve_x, curve_y = self.curve()

        control_points_x = np.array([control_point[0] for control_point in self.control_points])
        control_points_y = np.array([control_point[1] for control_point in self.control_points])

        x_min = min(np.min(curve_x), np.min(control_points_x))
        x_max = max(np.max(curve_x), np.max(control_points_x))
        x_range = np.abs(x_max - x_min)

        y_min = min(np.min(curve_y), np.min(control_points_y))
        y_max = max(np.max(curve_y), np.max(control_points_y))
        y_range = np.abs(y_max - y_min)

        ax.set_xlim(x_min - .1 * x_range, x_max + .1 * x_range)
        ax.set_ylim(y_min - .1 * y_range, y_max + .1 * y_range)

        ax.plot(control_points_x, control_points_y, color = 'C3', linestyle = '--', marker = 'o', linewidth = 1)
        ax.plot(curve_x, curve_y, color = 'C0', linewidth = 2)

        ax.axis('off')

        save_current_figure(name = 'curve', **kwargs)

        plt.close()


if __name__ == '__main__':
    plt_kwargs = dict(
        target_dir = OUT_DIR,
    )

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

        bspline.plot_basis_functions(**plt_kwargs)

        print('-' * 80)

    curve = BSplineCurve(BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2), [
        (0, 0),
        (0, 1),
        (1, -1),
        (1, .5),
        (2, .5),
        (2.5, -1),
        (3, 0),
        (3.5, 0),
    ])

    curve.plot_curve(**plt_kwargs)
