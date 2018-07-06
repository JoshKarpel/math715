import os

import final_project as nurbs
import utils

import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.getcwd(), 'couch_club')
PLT_KWARGS = dict(
    scale_factor = 6,
    img_format = 'png',
    target_dir = OUT_DIR,
)


def add_axes(fig, left, right, bottom, top):
    return fig.add_axes([left, bottom, right - left, top - bottom])


def refinement(**kwargs):
    # fig = utils.get_figure('full')
    fig = plt.figure(figsize = (6, 6))

    ####################

    # ax = fig.add_axes([.2, .8, .75, .25])
    ax = add_axes(fig, .4, .6, .7, .9)

    basis = nurbs.BSplineBasis([0, 0, 1, 1], polynomial_order = 1)

    basis.attach_basis_functions(ax)

    title_size = 10
    ax.set_xlabel(r'$\chi$', fontsize = title_size)
    ax.set_ylabel(r'$N_{i, \,p}\left(\chi\right)$', fontsize = title_size)
    ax.set_title(r'$\Theta = \left\lbrace 0, 0, 1, 1 \right\rbrace, \, p = 1$', fontsize = title_size)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, **nurbs.GRID_KWARGS)

    ###########

    # ax_upper = fig.add_axes([.6, .4, .75, .25])
    # ax_lower = fig.add_axes([.6, .1, .75, .25])
    ax_lower = add_axes(fig, .1, .45, .1, .3)
    ax_upper = add_axes(fig, .1, .45, .4, .6)

    axes = (ax_upper, ax_lower)

    basis_upper = nurbs.BSplineBasis([0, 0, 1 / 3, 2 / 3, 1, 1], polynomial_order = 1)
    basis_lower = nurbs.BSplineBasis([0, 0, 0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 1, 1, 1], polynomial_order = 2)
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
        ax.grid(True, **nurbs.GRID_KWARGS)

        ax.annotate(s = method, xy = (-.4, 1.0), xycoords = 'axes fraction', xytext = (-40, 30), textcoords = 'offset points',
                    fontsize = title_size, )

        ax.annotate(s = '', xy = (-.4, 1.0), xycoords = 'axes fraction', xytext = (0, 25), textcoords = 'offset points',
                    arrowprops = dict(
                        width = 1, facecolor = 'black', headlength = 10,
                    ))

    #################

    ax_lower = add_axes(fig, .55, .9, .1, .3)
    ax_upper = add_axes(fig, .55, .9, .4, .6)

    axes = (ax_upper, ax_lower)

    basis_upper = nurbs.BSplineBasis([0, 0, 0, 1, 1, 1], polynomial_order = 2)
    basis_lower = nurbs.BSplineBasis([0, 0, 0, 1 / 3, 2 / 3, 1, 1, 1], polynomial_order = 2)
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
        ax.grid(True, **nurbs.GRID_KWARGS)

        ax.annotate(s = method, xy = (1.4, 1.0), xycoords = 'axes fraction', xytext = (-40, 30), textcoords = 'offset points',
                    fontsize = title_size, )

        ax.annotate(s = '', xy = (1.4, 1.0), xycoords = 'axes fraction', xytext = (0, 25), textcoords = 'offset points',
                    arrowprops = dict(
                        width = 1, facecolor = 'black', headlength = 10,
                    ))

    ###################

    utils.save_current_figure('fig_10', **kwargs)

    plt.close()


if __name__ == '__main__':
    nurbs.figure_8(**PLT_KWARGS)
    nurbs.figure_9(**PLT_KWARGS)
    refinement(**PLT_KWARGS)

    bsplines = (
        nurbs.BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 0),
        nurbs.BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 1),
        nurbs.BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 2),
        nurbs.BSplineBasis(knot_vector = [0, 1, 2, 3, 4, 5], polynomial_order = 3),
    )
    for bspline in bsplines:
        bspline.plot_basis_functions(
            legend_on_right = False,
            **PLT_KWARGS,
        )

    nurbs.BSplineBasis(
        knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5],
        polynomial_order = 2
    ).plot_basis_functions(
        legend_on_right = True,
        **PLT_KWARGS,
    )

    bspline_curve = nurbs.BSplineCurve(
        nurbs.BSplineBasis(
            knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5],
            polynomial_order = 2)
        ,
        [
            (0, 0),
            (0, 1),
            (1, -1),
            (1, .5),
            (2, .5),
            (2.5, -1),
            (3, 0),
            (3.5, 0),
        ]
    )
    bspline_curve.plot_curve_2D(name = 'bspline_curve', **PLT_KWARGS)

    nurbs_curve = nurbs.NURBSCurve(
        nurbs.BSplineBasis(
            knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5],
            polynomial_order = 2,
        ),
        [
            (0, 0),
            (0, 1),
            (1, -1),
            (1, .5),
            (2, .5),
            (2.5, -1),
            (3, 0),
            (3.5, 0),
        ],
        weights = [1, 2, 1, 3, .5, 1, 1, 1],
    )
    nurbs_curve.plot_curve_2D(name = 'nurbs_curve', **PLT_KWARGS)

    fancy_curve = nurbs.BSplineCurve(
        nurbs.BSplineBasis(
            knot_vector = [0, 0, 0, 1, 2, 3, 4, 4, 5, 5, 5],
            polynomial_order = 2,
        ),
        [
            (0, 0, 0),
            (0, 1, 1),
            (1, -1, 2),
            (1, .5, -.5),
            (2, .5, 0),
            (2.5, -1, 2),
            (3, 0, -2),
            (3.5, 0, 3),
        ],
    )
    fancy_curve.plot_curve_3D(name = 'fancy_curve', **PLT_KWARGS)
