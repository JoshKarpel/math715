import os

from utils import *

import functools as ft

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.getcwd(), 'out')


def p1_disc(x, y, a):
    return ((x ** 2) - x - a) * (y ** 2)


def characterization_plot(a, x_bound = 5, y_bound = 5, points = 1000, **kwargs):
    x, y = np.linspace(-x_bound, x_bound, points), np.linspace(-y_bound, y_bound, points)

    x_mesh, y_mesh = np.meshgrid(x, y, indexing = 'ij')

    d_mesh = p1_disc(x_mesh, y_mesh, a)

    fig = get_figure('full')
    ax = fig.add_subplot(111)

    plt.set_cmap(plt.cm.get_cmap('seismic'))

    colormesh = ax.pcolormesh(x_mesh, y_mesh, d_mesh, vmin = -.5, vmax = .5, shading = 'gouraud')
    # plt.colorbar(colormesh, extend = 'both')

    plt.xlabel(r'$x$')
    plt.ylabel(r'$y$')
    plt.title(r'a = {}'.format(a))

    save_current_figure(name = 'a={}'.format(a), **kwargs)


if __name__ == '__main__':
    characterization_plot(-.25, x_bound = 10, y_bound = 10, points = 500,
                          target_dir = OUT_DIR)
