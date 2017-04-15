__author__ = 'Josh Karpel'

import os
import collections

import utils

import numpy as np
import numpy.ma as ma
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # looks unused, but necessary for MPL to plot projection = '3d' axes
import matplotlib.animation as anim

OUT_DIR = os.path.join(os.getcwd(), 'out')

GRID_KWARGS = {
    'linestyle': '-',
    'color': 'black',
    'linewidth': .25,
    'alpha': 0.4
}


class BSplineBasis:
    def __init__(self, knot_vector, polynomial_order, parameter_space_points = 10000):
        self.knot_vector = np.array(knot_vector)
        self.basis_function_indices = np.array(range(len(knot_vector) - polynomial_order - 1))
        self.polynomial_order = polynomial_order

        self.parameter_space_points = parameter_space_points

    @property
    def name(self):
        return f'p={self.polynomial_order}_k=[{",".join(str(s) for s in self.knot_vector)}]'

    def info(self):
        out = [
            self.name,
            f'  Knot Vector: {self.knot_vector}',
            f'  Number of Basis Functions: {self.number_of_basis_functions}',
            f'  Polynomial Order: {self.polynomial_order}',
            f'  Number of Knots: {self.number_of_knots}',
            f'  Number of Unique Knots: {self.number_of_unique_knots}',
            f'  # of Basis Functions + Polynomial Order + 1: {len(self.basis_function_indices) + bspline.polynomial_order + 1}',
        ]

        return '\n'.join(out)

    @property
    def number_of_basis_functions(self):
        return len(self.basis_function_indices)

    @property
    def number_of_knots(self):
        return len(self.knot_vector)

    @property
    def number_of_unique_knots(self):
        return len(set(self.knot_vector))

    @property
    def xi_min(self):
        return self.knot_vector[0]

    @property
    def xi_max(self):
        return self.knot_vector[-1]

    @property
    def xi(self):
        return np.linspace(self.xi_min, self.xi_max, self.parameter_space_points)

    def basis_null(self, basis_function_index):
        """Return the zero-order basis function evaluated along the parameter space for a given basis function index."""
        if basis_function_index == self.basis_function_indices[-1]:
            comp = np.less_equal
        else:
            comp = np.less

        return np.where(np.greater_equal(self.xi, self.knot_vector[basis_function_index]) * comp(self.xi, self.knot_vector[basis_function_index + 1]),
                        1.0, 0.0)

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

    def plot_basis_functions(self, fig_scale = 'full', **kwargs):
        fig = utils.get_figure(fig_scale)
        ax = fig.add_subplot(111)

        for basis_index in self.basis_function_indices:
            y = self.basis_function(basis_index, self.polynomial_order)  # get basis function
            y = ma.masked_equal(y, 0)  # mask zeros so they won't be displayed in the plot

            ax.plot(self.xi, y, label = rf'$N_{{ {basis_index}, {self.polynomial_order} }} $')

        ax.set_xlim(self.xi_min, self.xi_max)
        ax.set_ylim(0, 1.01)

        ax.set_xlabel(r'$\xi$', fontsize = 12)
        ax.set_ylabel(r'$N_{i,p}(\xi)$', fontsize = 12)

        if kwargs.get('img_format') != 'pgf':
            ax.set_title(fr'Basis Functions for $\Xi = \left[ {",".join(str(s) for s in self.knot_vector)} \right]$, $p = {self.polynomial_order}$')
            ax.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left', borderaxespad = 0., fontsize = 12, ncol = 1 + (len(self.basis_function_indices) // 15))
        else:
            ax.legend(loc = 'best', handlelength = 1)

        ax.grid(True, **GRID_KWARGS)

        utils.save_current_figure(name = self.name + '__basis', **kwargs)

        plt.close()


CONTROL_POLYGON_KWARGS = dict(
    color = 'C3',
    linestyle = '-',
    marker = 'o',
    linewidth = .75,
)

CURVE_KWARGS = dict(
    color = 'C0',
    linewidth = 2,
)


class BSplineCurve:
    def __init__(self, basis, control_points):
        self.basis = basis
        self.control_points = list(np.array(control_point) for control_point in control_points)

    def info(self):
        out = self.basis.info().split('\n')
        out += [f'  Control Points: {self.control_points}']

        return '\n'.join(out)

    def curve(self):
        """Return the d-dimensional curve given by the control points and basis functions."""
        return sum(np.outer(basis_function, control_point) for basis_function, control_point in zip(self.basis, self.control_points)).T

    def plot_curve_2D(self, fig_scale = 'full', **kwargs):
        """Only works in 2D..."""
        fig = utils.get_figure(fig_scale)
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

        ax.set_xlim(x_min - .02 * x_range, x_max + .02 * x_range)
        ax.set_ylim(y_min - .02 * y_range, y_max + .02 * y_range)

        ax.plot(control_points_x, control_points_y, **CONTROL_POLYGON_KWARGS)
        ax.plot(curve_x, curve_y, **CURVE_KWARGS)

        ax.axis('off')

        utils.save_current_figure(**kwargs)

        plt.close()

    def plot_curve_3D(self, length = 30, fps = 30, **kwargs):
        """Only works in 3D..."""
        fig = utils.get_figure(scale = 3)
        ax = fig.add_subplot(111, projection = '3d')

        curve_x, curve_y, curve_z = self.curve()

        control_points_x = np.array([control_point[0] for control_point in self.control_points])
        control_points_y = np.array([control_point[1] for control_point in self.control_points])
        control_points_z = np.array([control_point[2] for control_point in self.control_points])

        x_min = min(np.min(curve_x), np.min(control_points_x))
        x_max = max(np.max(curve_x), np.max(control_points_x))
        x_range = np.abs(x_max - x_min)

        y_min = min(np.min(curve_y), np.min(control_points_y))
        y_max = max(np.max(curve_y), np.max(control_points_y))
        y_range = np.abs(y_max - y_min)

        z_min = min(np.min(curve_z), np.min(control_points_z))
        z_max = max(np.max(curve_z), np.max(control_points_z))
        z_range = np.abs(z_max - z_min)

        ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        ax.set_zlim(z_min - 0.05 * z_range, z_max + 0.05 * z_range)

        ax.plot(control_points_x, control_points_y, control_points_z, **CONTROL_POLYGON_KWARGS)
        ax.plot(curve_x, curve_y, curve_z, **CURVE_KWARGS)

        ax.axis('off')

        ax.view_init(elev = 45, azim = 0)  # note that this resets ax.dist to 10, so we can't use it below
        ax.dist = 7.5  # default is 10, so zoom in a little because there's no axis to take up the rest of the space

        ### ANIMATION ###

        frames = length * fps

        writer = anim.writers['ffmpeg'](fps = fps, bitrate = 2000)  # don't need a very high bitrate

        def animate(frame):
            ax.azim = 360 * frame / frames  # one full rotation
            return []  # must return the list of artists we modified (i.e., nothing, since all we did is rotate the view)

        ani = anim.FuncAnimation(fig, animate, frames = frames, blit = True)
        ani.save(f"{os.path.join(kwargs['target_dir'], kwargs['name'])}.mp4", writer = writer)

        plt.close()


def random_curve(number_of_unique_knots, polynomial_order = 2, dimensions = 3):
    """Return a BSplineCurve in some number of dimensions and polynomial order, generated from a random sample of knots and control points."""
    knot_multiplicites = rand.randint(1, polynomial_order + 1, size = number_of_unique_knots)

    # ensure interpolation on the edges of the control polygon
    knot_multiplicites[0] = polynomial_order + 1
    knot_multiplicites[-1] = polynomial_order + 1

    knot_vector = np.repeat(range(len(knot_multiplicites)), repeats = knot_multiplicites)

    basis = BSplineBasis(knot_vector = knot_vector, polynomial_order = polynomial_order)

    control_points = rand.random_sample((basis.number_of_basis_functions, dimensions))

    curve = BSplineCurve(basis, control_points)

    return curve


if __name__ == '__main__':
    plt_kwargs = dict(
        target_dir = OUT_DIR,
    )

    ## TESTING BSPLINE BASIS FUNCTIONS ##
    bsplines = (
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 0),
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 1),
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 2),
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 3),
        BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 4),
        BSplineBasis(knot_vector = [0, 0, 1, 2, 3, 4, 5], polynomial_order = 2),
        BSplineBasis(knot_vector = [0, 0, 1, 2, 3, 4, 5, 5], polynomial_order = 1),
        BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2),
        BSplineBasis(knot_vector = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 9, 9], polynomial_order = 3),
        BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 9], polynomial_order = 4),
    )
    for bspline in bsplines:
        print(bspline.info())
        bspline.plot_basis_functions(**plt_kwargs)
        bspline.plot_basis_functions(**plt_kwargs, img_format = 'pgf', fig_scale = 'half')
        print('-' * 80)

    ## CURVE FROM HUGHES ET. AL. 2004 ##
    paper_curve = BSplineCurve(BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2), [
        (0, 0),
        (0, 1),
        (1, -1),
        (1, .5),
        (2, .5),
        (2.5, -1),
        (3, 0),
        (3.5, 0),
    ])
    paper_curve.plot_curve_2D(name = 'paper_curve', **plt_kwargs)
    paper_curve.plot_curve_2D(name = 'paper_curve', **plt_kwargs, img_format = 'pgf', fig_scale = 'full')

    ## 3D VERSION OF PAPER CURVE ##
    fancy_curve = BSplineCurve(BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2), [
        (0, 0, 0),
        (0, 1, 1),
        (1, -1, 2),
        (1, .5, -.5),
        (2, .5, 0),
        (2.5, -1, 2),
        (3, 0, -2),
        (3.5, 0, 3),
    ])
    fancy_curve.plot_curve_3D(name = 'fancy_curve', **plt_kwargs)

    ## RANDOM CURVES ##
    uk_p_d = (
        (6, 2, 2),
        (8, 3, 2),
        (6, 2, 3),
        (10, 4, 3)
    )
    for uk, p, d in uk_p_d:
        random_curves = list(random_curve(uk, polynomial_order = p, dimensions = d) for _ in range(5))
        for ii, rc in enumerate(random_curves):
            print(rc.info())
            print('-' * 80)

            name = f'rc_uk={uk}_p={p}_d={d}__{ii}'

            with open(os.path.join(OUT_DIR, f'{name}.txt'), mode = 'w') as f:
                f.write(rc.info())

            if d == 2:
                rc.plot_curve_2D(name = name, **plt_kwargs)
            elif d == 3:
                rc.plot_curve_3D(name = name, **plt_kwargs)
