__author__ = 'Josh Karpel'

import functools as ft
import os

import matplotlib as mpl

mpl.use('Agg')

mpl_rcParams_update = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'mathtext.rm': 'serif',
}

mpl.rcParams.update(mpl_rcParams_update)

import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision = 3, linewidth = 200)


def ensure_dir_exists(path):
    """Ensure that the directory tree to the path exists."""
    split_path = os.path.splitext(path)
    if split_path[0] != path:  # path is file
        make_path = os.path.dirname(split_path[0])
    else:  # path is dir
        make_path = split_path[0]
    os.makedirs(make_path, exist_ok = True)


def save_current_figure(name = 'img', target_dir = None, img_format = 'pdf', scale_factor = 1):
    """Save the current matplotlib figure with the given name to the given folder."""
    if target_dir is None:
        target_dir = os.getcwd()
    path = os.path.join(target_dir, '{}.{}'.format(name, img_format))

    ensure_dir_exists(path)

    plt.savefig(path, dpi = scale_factor * plt.gcf().dpi, bbox_inches = 'tight')


def figsize(scale, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0):
    """
    Helper function for get_figure

    :param scale:
    :param fig_width_pts: get this from LaTeX using \the\textwidth
    :param aspect_ratio: height = width * ratio, defaults to golden ratio
    :return:
    """
    inches_per_pt = 1.0 / 72.27  # Convert pt to inch

    fig_width = fig_width_pts * inches_per_pt * scale  # width in inches
    fig_height = fig_width * aspect_ratio  # height in inches
    fig_size = [fig_width, fig_height]

    return fig_size


def get_figure(scale = 0.9, fig_width_pts = 498.66258, aspect_ratio = (np.sqrt(5.0) - 1.0) / 2.0):
    """
    Get a matplotlib figure object with the desired scale relative to a full-text-width LaTeX page.

    scale = 'full' -> scale = 0.95
    scale = 'half' -> scale = 0.475

    :param scale: width of figure in LaTeX \textwidths
    :param fig_width_pts: get this from LaTeX using \the\textwidth
    :param aspect_ratio: height = width * ratio, defaults to golden ratio
    :return:
    """
    if scale == 'full':
        scale = 0.95
    elif scale == 'half':
        scale = .475

    fig = plt.figure(figsize = figsize(scale, fig_width_pts = fig_width_pts, aspect_ratio = aspect_ratio))

    return fig


def hash_args_kwargs(*args, **kwargs):
    """Return the hash of a tuple containing the args and kwargs."""
    return hash(args + tuple(kwargs.items()))


def memoize(func):
    """Memoize a function by storing a dictionary of {inputs: outputs}."""
    memo = {}

    @ft.wraps(func)
    def memoizer(*args, **kwargs):
        key = hash_args_kwargs(*args, **kwargs)
        if key not in memo:
            memo[key] = func(*args, **kwargs)
        return memo[key]

    return memoizer
