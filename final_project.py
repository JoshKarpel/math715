import itertools
import os

import utils

import numpy as np
import numpy.ma as ma
import numpy.random as rand
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # looks unused, but necessary for MPL to plot projection = '3d' axes
import matplotlib.animation as anim

OUT_DIR = os.path.join(os.getcwd(), 'out_surface_test')

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

    def info(self):
        out = [
            self.name,
            f'  Knot Vector: {self.knot_vector}',
            f'  Number of Basis Functions: {self.number_of_basis_functions}',
            f'  Polynomial Order: {self.polynomial_order}',
            f'  Number of Knots: {self.number_of_knots}',
            f'  Number of Unique Knots: {self.number_of_unique_knots}',
            f'  # of Basis Functions + Polynomial Order + 1: {len(self.basis_function_indices) + self.polynomial_order + 1}',
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

    @utils.memoize
    def basis_null(self, basis_function_index):
        """Return the zero-order basis function evaluated along the parameter space for a given basis function index."""
        if basis_function_index == self.basis_function_indices[-1]:
            comp = np.less_equal
        else:
            comp = np.less

        return np.where(np.greater_equal(self.xi, self.knot_vector[basis_function_index]) * comp(self.xi, self.knot_vector[basis_function_index + 1]),
                        1.0, 0.0)

    @utils.memoize
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

    def attach_basis_functions(self, ax):
        for basis_index in self.basis_function_indices:
            y = self.basis_function(basis_index, self.polynomial_order)  # get basis function
            y = ma.masked_equal(y, 0)  # mask zeros so they won't be displayed in the plot

            ax.plot(self.xi, y, label = rf'$N_{{ {basis_index}, {self.polynomial_order} }} $')

    def plot_basis_functions(self, fig_scale = 'full', title = True, legend_on_right = True, **kwargs):
        fig = utils.get_figure(fig_scale)
        ax = fig.add_subplot(111)

        self.attach_basis_functions(ax)

        ax.set_xlim(self.xi_min, self.xi_max)
        ax.set_ylim(0, 1.01)

        ax.set_xlabel(r'$\chi$', fontsize = 12)
        ax.set_ylabel(r'$N_{i,p}(\chi)$', fontsize = 12)

        if title:
            ax.set_title(fr'Basis Functions for $\Theta = \left\lbrace {",".join(str(s) for s in self.knot_vector)} \right\rbrace$, $p = {self.polynomial_order}$')

        if legend_on_right:
            ax.legend(bbox_to_anchor = (1.02, 1), loc = 'upper left', borderaxespad = 0., fontsize = 12, handlelength = 1, ncol = 1 + (len(self.basis_function_indices) // 15))
        else:
            ax.legend(loc = 'upper right', handlelength = 1)

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

    def attach_curve_2D(self, ax):
        curve_x, curve_y = self.curve()

        control_points_x = np.array([control_point[0] for control_point in self.control_points])
        control_points_y = np.array([control_point[1] for control_point in self.control_points])

        x_min = min(np.min(curve_x), np.min(control_points_x))
        x_max = max(np.max(curve_x), np.max(control_points_x))
        x_range = np.abs(x_max - x_min)

        y_min = min(np.min(curve_y), np.min(control_points_y))
        y_max = max(np.max(curve_y), np.max(control_points_y))
        y_range = np.abs(y_max - y_min)

        ax.set_xlim(x_min - .05 * x_range, x_max + .05 * x_range)
        ax.set_ylim(y_min - .05 * y_range, y_max + .05 * y_range)

        ax.plot(control_points_x, control_points_y, **CONTROL_POLYGON_KWARGS)
        ax.plot(curve_x, curve_y, **CURVE_KWARGS)

        ax.axis('off')

    def plot_curve_2D(self, fig_scale = 'full', **kwargs):
        """Only works in 2D..."""
        fig = utils.get_figure(fig_scale)
        ax = fig.add_subplot(111)

        self.attach_curve_2D(ax)

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
            print(frame, frames, frame / frames)
            ax.azim = 360 * frame / frames  # one full rotation
            return []  # must return the list of artists we modified (i.e., nothing, since all we did is rotate the view)

        ani = anim.FuncAnimation(fig, animate, frames = frames, blit = True)
        ani.save(f"{os.path.join(kwargs['target_dir'], kwargs['name'])}.mp4", writer = writer)

        plt.close()


class NURBSCurve(BSplineCurve):
    def __init__(self, basis, control_points, weights):
        super().__init__(basis, control_points)
        self.weights = np.array(weights)

    def curve(self):
        """Return the d-dimensional curve given by the (weighted) control points and basis functions."""
        return (sum(np.outer(basis_function * weight, control_point) for basis_function, control_point, weight in zip(self.basis, self.control_points, self.weights)).T /
                sum(basis_function * weight for basis_function, weight in zip(self.basis, self.weights)).T)


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


class BSplineSurface:
    def __init__(self, basis_1, basis_2, control_net):
        self.basis_1 = basis_1
        self.basis_2 = basis_2
        self.control_net = control_net

        self.xi_1_mesh, self.xi_2_mesh = np.meshgrid(self.basis_1.xi, self.basis_2.xi, indexing = 'ij')

    def surface(self):
        """Return the d-dimensional surface given by the control points and basis functions."""
        # return sum(np.outer(basis_function, control_point) for basis_function, control_point in zip(self.basis_1, self.basis)).T
        # return sum(np.outer(basis_function_1, self.control_net[ii, jj]) for ((ii, basis_function_1), (jj, basis_function_2)) in zip(enumerate(self.basis_1), enumerate(self.basis_2))).T
        # return sum(np.outer(basis_function_1, self.control_net[ii, jj]) + np.outer(basis_function_2, self.control_net[ii, jj]) for ((ii, basis_function_1), (jj, basis_function_2)) in zip(enumerate(self.basis_1), enumerate(self.basis_2))).T

        # x = np.zeros_like(self.xi_1_mesh)
        # y = np.zeros_like(self.xi_1_mesh)
        # z = np.zeros_like(self.xi_1_mesh)
        xyz = np.zeros((*self.xi_1_mesh.shape, 3))
        for (i, basis_function_i), (j, basis_function_j) in itertools.product(enumerate(self.basis_1), enumerate(self.basis_2)):
            print(i, basis_function_i)
            print(j, basis_function_j)
            print(self.control_net[i, j])
            # b1, b2 = np.meshgrid(basis_function_i, basis_function_j, indexing = 'ij')
            control_x, control_y, control_z = self.control_net[i, j]
            # print(b1.shape, b2.shape, np.array(self.control_net[i, j]).shape)
            # print((b1 * b2).shape)
            # z += np.outer(b1 * b2, self.control_net[i, j])
            # print(np.shape(z))
            print(np.outer(basis_function_i, basis_function_j))
            # x += np.outer(basis_function_i, basis_function_j) * control_x
            # y += np.outer(basis_function_i, basis_function_j) * control_y
            # z += np.outer(basis_function_i, basis_function_j) * control_z
            print(np.outer(basis_function_i, basis_function_j).shape)
            print(np.outer(np.outer(basis_function_i, basis_function_j), self.control_net[i, j]).shape)
            print(np.outer(np.outer(basis_function_i, basis_function_j), np.array(self.control_net[i, j])).shape)
            r = np.einsum('i,j,k->ijk', basis_function_i, basis_function_j, np.array(self.control_net[i, j]))
            print(r.shape)
            xyz += r

        # print(x, y, z)

        # return x, y, z
        return xyz

    def plot_surface_3D(self, length = 30, fps = 30, **kwargs):
        """Only works in 3D..."""
        fig = utils.get_figure(scale = 3)
        ax = fig.add_subplot(111, projection = '3d')

        # surface_x = self.xi_1_mesh
        # surface_y = self.xi_2_mesh
        # surface_x, surface_y, surface_z = self.surface()
        xyz = self.surface()

        # surface_x, surface_y = np.meshgrid(surface_x, surface_y)

        # print(np.shape(surface_x))
        # print(np.shape(surface_y))
        # print(np.shape(surface_z))

        control_points_x = np.array([control_point[0] for control_point in self.control_net.values()])
        control_points_y = np.array([control_point[1] for control_point in self.control_net.values()])
        control_points_z = np.array([control_point[2] for control_point in self.control_net.values()])

        # x_min = min(np.min(surface_x), np.min(control_points_x))
        # x_max = max(np.max(surface_x), np.max(control_points_x))
        # x_range = np.abs(x_max - x_min)
        #
        # y_min = min(np.min(surface_y), np.min(control_points_y))
        # y_max = max(np.max(surface_y), np.max(control_points_y))
        # y_range = np.abs(y_max - y_min)
        #
        # z_min = min(np.min(surface_z), np.min(control_points_z))
        # z_max = max(np.max(surface_z), np.max(control_points_z))
        # z_range = np.abs(z_max - z_min)
        #
        # ax.set_xlim(x_min - 0.05 * x_range, x_max + 0.05 * x_range)
        # ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        # ax.set_zlim(z_min - 0.05 * z_range, z_max + 0.05 * z_range)

        ax.scatter(control_points_x, control_points_y, control_points_z, depthshade = False, **CONTROL_POLYGON_KWARGS)

        # print(np.max(surface_x), np.max(surface_y), np.max(surface_z))
        # print(np.min(surface_x), np.min(surface_y), np.min(surface_z))
        # print(surface_x)
        # print(surface_y)
        # print(surface_z)
        xyz = np.reshape(xyz, (-1, 3))
        print(xyz.shape)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        ax.scatter(x, y, z)
        # ax.plot_trisurf(
        #     x, y, z,
        #     cmap = plt.get_cmap('viridis'),
        #     linewidth = 0,
        #     antialiased = True,
        # )
        # ax.plot_surface(surface_x, surface_y, surface_z, rstride = 1, cstride = 1)
        # ax.plot_trisurf(surface_x, surface_y, surface_z)
        # ax.plot_trisurf(surface_x, surface_y, surface_z, **CURVE_KWARGS)

        ax.axis('off')

        ax.view_init(elev = 45, azim = 0)  # note that this resets ax.dist to 10, so we can't use it below
        ax.dist = 7.5  # default is 10, so zoom in a little because there's no axis to take up the rest of the space

        plt.show()
        utils.save_current_figure(**kwargs)

        ### ANIMATION ###

        frames = length * fps

        writer = anim.writers['ffmpeg'](fps = fps, bitrate = 2000)  # don't need a very high bitrate

        def animate(frame):
            print(frame, frames, frame / frames)
            ax.azim = 360 * frame / frames  # one full rotation
            return []  # must return the list of artists we modified (i.e., nothing, since all we did is rotate the view)

        ani = anim.FuncAnimation(fig, animate, frames = frames, blit = True)
        ani.save(f"{os.path.join(kwargs['target_dir'], kwargs['name'])}.mp4", writer = writer)

        plt.close()


def figure_8(**kwargs):
    fig = utils.get_figure('full', aspect_ratio = .8)

    ax_original_curve = fig.add_subplot(221)
    ax_original_basis = fig.add_subplot(223)
    ax_new_curve = fig.add_subplot(222)
    ax_new_basis = fig.add_subplot(224)

    original_basis = BSplineBasis([0, 0, 0, 1, 1, 1], polynomial_order = 2)
    original_curve = BSplineCurve(original_basis,
                                  [
                                      (0, 0),
                                      (.5, 1),
                                      (1, 0)
                                  ])

    title_size = 10
    ax_original_curve.set_title(r'Original Curve: $\Theta = \left\lbrace 0, 0, 0, 1, 1, 1\right\rbrace, \; p = 2$', fontsize = title_size)
    ax_original_basis.set_title(r'Original Basis Functions', fontsize = title_size)

    new_basis = BSplineBasis([0, 0, 0, .5, 1, 1, 1], polynomial_order = 2)
    new_curve = BSplineCurve(new_basis,
                             [
                                 (0, 0),
                                 (.25, .5),
                                 (.75, .5),
                                 (1, 0)
                             ])

    ax_new_curve.set_title(r"$h$-Refined Curve: $\Theta' = \left\lbrace 0, 0, 0, 0.5, 1, 1, 1\right\rbrace, \; p = 2$", fontsize = title_size)
    ax_new_basis.set_title(r'$h$-Refined Basis Functions', fontsize = title_size)

    for ax, basis in ((ax_original_curve, original_curve), (ax_new_curve, new_curve)):
        basis.attach_curve_2D(ax)
        ax.set_xlim(-.05, 1.05)
        ax.set_ylim(-.05, 1.05)
        ax.grid(True, linewidth = .5)
        ax.axis('on')

        ax.set_xticks((0, 1 / 4, 1 / 2, 3 / 4, 1))
        ax.set_xticklabels((r'$0$',
                            r'$\frac{1}{4}$',
                            r'$\frac{1}{2}$',
                            r'$\frac{3}{4}$',
                            r'$1$',
                            ))
        ax.set_yticks((0, 1 / 2, 1))
        ax.set_yticklabels((r'$0$',
                            r'$\frac{1}{2}$',
                            r'$1$',
                            ))

    for ax, basis in ((ax_original_basis, original_basis), (ax_new_basis, new_basis)):
        basis.attach_basis_functions(ax)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'$\chi$')
        ax.grid(True, **GRID_KWARGS)

    ax_original_basis.set_ylabel(r'$N_{i, \,p}\left(\chi\right)$')

    plt.tight_layout()

    utils.save_current_figure('fig_8', **kwargs)

    plt.close()


def figure_9(**kwargs):
    fig = utils.get_figure('full', aspect_ratio = .8)

    ax_original_curve = fig.add_subplot(221)
    ax_original_basis = fig.add_subplot(223)
    ax_new_curve = fig.add_subplot(222)
    ax_new_basis = fig.add_subplot(224)

    original_basis = BSplineBasis([0, 0, 0, 1, 1, 1], polynomial_order = 2)
    original_curve = BSplineCurve(original_basis,
                                  [
                                      (0, 0),
                                      (.5, 1),
                                      (1, 0)
                                  ])

    title_size = 10
    ax_original_curve.set_title(r'Original Curve: $\Theta = \left\lbrace 0, 0, 0, 1, 1, 1 \right\rbrace, \; p = 2$', fontsize = title_size)
    ax_original_basis.set_title(r'Original Basis Functions', fontsize = title_size)
    ax_new_curve.set_title(r"$p$-Refined Curve: $\Theta' = \left\lbrace 0, 0, 0, 0, 1, 1, 1, 1 \right\rbrace, \; p = 3$", fontsize = title_size)
    ax_new_basis.set_title(r'$p$-Refined Basis Functions', fontsize = title_size)

    new_basis = BSplineBasis([0, 0, 0, 0, 1, 1, 1, 1], polynomial_order = 3)
    new_curve = BSplineCurve(new_basis,
                             [
                                 (0, 0),
                                 (1 / 3, 2 / 3),
                                 (2 / 3, 2 / 3),
                                 (1, 0)
                             ])

    for ax, basis in ((ax_original_curve, original_curve), (ax_new_curve, new_curve)):
        basis.attach_curve_2D(ax)
        ax.set_xlim(-.05, 1.05)
        ax.set_ylim(-.05, 1.05)
        ax.grid(True, linewidth = .5)
        ax.axis('on')

        ax.set_xticks((0, 1 / 3, 2 / 3, 1))
        ax.set_xticklabels((r'$0$',
                            r'$\frac{1}{3}$',
                            r'$\frac{2}{3}$',
                            r'$1$',
                            ))
        ax.set_yticks((0, 2 / 3, 1))
        ax.set_yticklabels((r'$0$',
                            r'$\frac{2}{3}$',
                            r'$1$',
                            ))

    for ax, basis in ((ax_original_basis, original_basis), (ax_new_basis, new_basis)):
        basis.attach_basis_functions(ax)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel(r'$\chi$')
        ax.grid(True, **GRID_KWARGS)

    ax_original_basis.set_ylabel(r'$N_{i, \,p}\left(\chi\right)$')

    plt.tight_layout()

    utils.save_current_figure('fig_9', **kwargs)

    plt.close()


def figure_10a(**kwargs):
    fig = utils.get_figure('half')
    ax = fig.add_subplot(111)

    basis = BSplineBasis([0, 0, 1, 1], polynomial_order = 1)

    basis.attach_basis_functions(ax)

    title_size = 10
    ax.set_xlabel(r'$\chi$', fontsize = title_size)
    ax.set_ylabel(r'$N_{i, \,p}\left(\chi\right)$', fontsize = title_size)
    ax.set_title(r'$\Theta = \left\lbrace 0, 0, 1, 1 \right\rbrace, \, p = 1$', fontsize = title_size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, **GRID_KWARGS)

    utils.save_current_figure('fig_10a', **kwargs)

    plt.close()


def figure_10b(**kwargs):
    fig = utils.get_figure(.425, aspect_ratio = 1 / utils.golden_ratio)

    # ax_upper = fig.add_subplot(211)
    # ax_lower = fig.add_subplot(212)

    ax_upper = fig.add_axes([.2, .6, .75, .25])
    ax_lower = fig.add_axes([.2, .1, .75, .25])

    axes = (ax_upper, ax_lower)

    basis_upper = BSplineBasis([0, 0, 1 / 3, 2 / 3, 1, 1], polynomial_order = 1)
    basis_lower = BSplineBasis([0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1], polynomial_order = 2)
    bases = (basis_upper, basis_lower)

    titles = (r"$\Theta' = \left\lbrace 0, 0, \frac{1}{3}, \frac{2}{3}, 1, 1 \right\rbrace, \, p = 1$",
              r"$\Theta'' = \left\lbrace 0, 0, 0, \frac{1}{3}, \frac{1}{3}, \frac{2}{3}, \frac{2}{3}, 1, 1, 1 \right\rbrace, \, p = 2$")

    method = (r'$h$-refinement \\ (knot insertion)',
              r'$p$-refinement \\ (order elevation)')

    for ax, basis, title, method in zip(axes, bases, titles, method):
        basis.attach_basis_functions(ax)

        title_size = 10
        ax.set_ylabel(r'$N_{i, \,p}\left(\chi\right)$', fontsize = title_size)
        ax.set_xlabel(r'$\chi$', fontsize = title_size)
        ax.set_title(title, fontsize = title_size)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, **GRID_KWARGS)

        ax.annotate(s = method, xy = (0.5, 1.275), xycoords = 'axes fraction', xytext = (10, 8), textcoords = 'offset points',
                    fontsize = title_size, )

        ax.annotate(s = '', xy = (0.5, 1.275), xycoords = 'axes fraction', xytext = (0, 25), textcoords = 'offset points',
                    arrowprops = dict(
                        width = 1, facecolor = 'black', headlength = 10,
                    ))

    # plt.tight_layout()

    utils.save_current_figure('fig_10b', tight = False, **kwargs)

    plt.close()


def figure_10c(**kwargs):
    fig = utils.get_figure(.425, aspect_ratio = 1 / utils.golden_ratio)

    # ax_upper = fig.add_subplot(211)
    # ax_lower = fig.add_subplot(212)

    ax_upper = fig.add_axes([.2, .6, .75, .25])
    ax_lower = fig.add_axes([.2, .1, .75, .25])

    axes = (ax_upper, ax_lower)

    basis_upper = BSplineBasis([0, 0, 0, 1, 1, 1], polynomial_order = 2)
    basis_lower = BSplineBasis([0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1], polynomial_order = 2)
    bases = (basis_upper, basis_lower)

    titles = (r"$\Theta' = \left\lbrace 0, 0, 0, 1, 1, 1 \right\rbrace, \, p = 2$",
              r"$\Theta'' = \left\lbrace 0, 0, 0, \frac{1}{3}, \frac{2}{3}, 1, 1, 1 \right\rbrace, \, p = 2$")

    method = (r'$p$-refinement \\ (order elevation)',
              r'$h$-refinement \\ (knot insertion)')

    for ax, basis, title, method in zip(axes, bases, titles, method):
        basis.attach_basis_functions(ax)

        title_size = 10
        ax.set_ylabel(r'$N_{i, \,p}\left(\chi\right)$', fontsize = title_size)
        ax.set_xlabel(r'$\chi$', fontsize = title_size)
        ax.set_title(title, fontsize = title_size)
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.grid(True, **GRID_KWARGS)

        ax.annotate(s = method, xy = (0.5, 1.275), xycoords = 'axes fraction', xytext = (10, 8), textcoords = 'offset points',
                    fontsize = title_size, )

        ax.annotate(s = '', xy = (0.5, 1.275), xycoords = 'axes fraction', xytext = (0, 25), textcoords = 'offset points',
                    arrowprops = dict(
                        width = 1, facecolor = 'black', headlength = 10,
                    ))

    # plt.tight_layout()

    utils.save_current_figure('fig_10c', tight = False, **kwargs)

    plt.close()


if __name__ == '__main__':
    plt_kwargs = dict(
        target_dir = OUT_DIR,
    )

    # for fmt in ('pdf', 'pgf'):
    #     kw = dict(img_format = fmt, **plt_kwargs)
    #
    #     figure_8(**kw)
    #     figure_9(**kw)
    #     figure_10a(**kw)
    #     figure_10b(**kw)
    #     figure_10c(**kw)
    #
    # ## TESTING BSPLINE BASIS FUNCTIONS ##
    # bsplines = (
    #     BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 0),
    #     BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 1),
    #     BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 2),
    #     BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 3),
    #     # BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 4),
    #     # BSplineBasis(knot_vector = [0, 0, 1, 2, 3, 4, 5], polynomial_order = 2),
    #     # BSplineBasis(knot_vector = [0, 0, 1, 2, 3, 4, 5, 5], polynomial_order = 1),
    #     BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2),
    #     # BSplineBasis(knot_vector = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 7, 7, 8, 8, 9, 9, 9], polynomial_order = 3),
    #     # BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 7, 8, 8, 9, 9, 9], polynomial_order = 4),
    # )
    # for bspline in bsplines:
    #     print(bspline.info())
    #     bspline.plot_basis_functions(**plt_kwargs)
    #     bspline.plot_basis_functions(**plt_kwargs, img_format = 'pgf', fig_scale = 'half', title = False, legend_on_right = False)
    #     print('-' * 80)
    #
    # paper_spline = BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2)
    # paper_spline.plot_basis_functions(**plt_kwargs)
    # paper_spline.plot_basis_functions(**plt_kwargs, img_format = 'pgf', fig_scale = 'full', title = False, legend_on_right = True)
    #
    # ## CURVE FROM HUGHES ET. AL. 2004 ##
    # paper_curve = BSplineCurve(BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2), [
    #     (0, 0),
    #     (0, 1),
    #     (1, -1),
    #     (1, .5),
    #     (2, .5),
    #     (2.5, -1),
    #     (3, 0),
    #     (3.5, 0),
    # ])
    # paper_curve.plot_curve_2D(name = 'paper_curve', **plt_kwargs)
    # paper_curve.plot_curve_2D(name = 'paper_curve', **plt_kwargs, img_format = 'pgf', fig_scale = 'full', title = False, legend_on_right = True)

    # ## NURBS CURVE ##
    # nurbs_curve = NURBSCurve(BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2), [
    #     (0, 0),
    #     (0, 1),
    #     (1, -1),
    #     (1, .5),
    #     (2, .5),
    #     (2.5, -1),
    #     (3, 0),
    #     (3.5, 0),
    # ], weights = [1, 2, 1, 3, .5, 1, 1, 1])
    # nurbs_curve.plot_curve_2D(name = 'nurbs_curve', **plt_kwargs)
    # nurbs_curve.plot_curve_2D(name = 'nurbs_curve', **plt_kwargs, img_format = 'pgf', fig_scale = 'full')

    # ## 3D VERSION OF PAPER CURVE ##
    # fancy_curve = BSplineCurve(BSplineBasis(knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5], polynomial_order = 2), [
    #     (0, 0, 0),
    #     (0, 1, 1),
    #     (1, -1, 2),
    #     (1, .5, -.5),
    #     (2, .5, 0),
    #     (2.5, -1, 2),
    #     (3, 0, -2),
    #     (3.5, 0, 3),
    # ])
    # fancy_curve.plot_curve_3D(name = 'fancy_curve', **plt_kwargs)

    # ## RANDOM CURVES ##
    # uk_p_d = (
    #     # (6, 2, 2),
    #     # (8, 3, 2),
    #     (4, 2, 3),
    #     (6, 2, 3),
    #     # (10, 4, 3),
    # )
    # for uk, p, d in uk_p_d:
    #     random_curves = list(random_curve(uk, polynomial_order = p, dimensions = d) for _ in range(5))
    #     for ii, rc in enumerate(random_curves):
    #         print(rc.info())
    #
    #         name = f'rc_uk={uk}_p={p}_d={d}__{ii}__' + rc.basis.name
    #
    #         # with open(os.path.join(OUT_DIR, f'{name}.txt'), mode = 'w') as f:
    #         #     f.write(rc.info())
    #
    #         # rc.basis.plot_basis_functions(**plt_kwargs)
    #         # rc.basis.plot_basis_functions(**plt_kwargs, img_format = 'pgf', fig_scale = 'half', title = False, legend_on_right = False)
    #
    #         if d == 2:
    #             rc.plot_curve_2D(name = name, **plt_kwargs)
    #         elif d == 3:
    #             rc.plot_curve_3D(name = name, **plt_kwargs)
    #
    #         print('-' * 80)

    ## BSPLINE SURFACE ##
    cn = {
        (0, 0): (0, 0, 0),
        (1, 0): (1, 0, 1),
        (0, 1): (0, 0.5, 2),
        (1, 1): (1, 1, 0.5),
        (2, 0): (2, 0, -0.5),
        (0, 2): (0, 1.5, -1),
        (2, 1): (0, 1, 1.5),
        (1, 2): (1, 0, -2),
        (2, 2): (2, 1, 1),
    }
    surf = BSplineSurface(
        basis_1 = BSplineBasis([0, 0, 1, 1, 2, 3], polynomial_order = 2, parameter_space_points = 101),
        basis_2 = BSplineBasis([0, 0, 1, 2, 3, 3], polynomial_order = 2, parameter_space_points = 101),
        control_net = cn,
    )
    surf.basis_1.plot_basis_functions(**plt_kwargs)
    surf.basis_2.plot_basis_functions(**plt_kwargs)
    # surf.surface()
    surf.plot_surface_3D(name = 'surface', **plt_kwargs)
