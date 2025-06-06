# coding=utf-8


import math

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.path import Path
from matplotlib import cm

from braindecode.datasets.sensor_positions import (
    CHANNEL_10_20_APPROX,
    get_channelpos,
)


def ax_scalp(
    v,
    channels,
    ax=None,
    annotate=False,
    vmin=None,
    vmax=None,
    cmap=cm.coolwarm,
    scalp_line_width=1,
    scalp_line_style="solid",
    chan_pos_list=CHANNEL_10_20_APPROX,
    interpolation="bilinear",
    fontsize=8,
):
    """Draw a scalp plot.

    Draws a scalp plot on an existing axes. The method takes an array of
    values and an array of the corresponding channel names. It matches
    the channel names with an channel position list
    to project them correctly on the scalp.

    Parameters
    ----------
    v : 1d-array of floats
        The values for the channels
    channels : 1d array of strings
        The corresponding channel names for the values in ``v``
    ax : Axes, optional
        The axes to draw the scalp plot on. If not provided, the
        currently activated axes (i.e. ``gca()``) will be taken
    annotate : Boolean, optional
        Draw the channel names next to the channel markers.
    vmin, vmax : float, optional
        The display limits for the values in ``v``. If the data in ``v``
        contains values between -3..3 and ``vmin`` and ``vmax`` are set
        to -1 and 1, all values smaller than -1 and bigger than 1 will
        appear the same as -1 and 1. If not set, the maximum absolute
        value in ``v`` is taken to calculate both values.
    cmap : matplotlib.colors.colormap, optional
        A colormap to define the color transitions.
    scalp_line_width: float
        Line width for outline of scalp
    scalp_line_style: float
        Line style for outline of scalp
    chan_pos_list: iterable of tuples
        First entry should be 'angle' or 'cartesian',
        remaining entries 2-tuples of x and y.
    interpolation: str

    Returns
    -------
    ax : Axes
        the axes on which the plot was drawn

    Notes
    -----
    Code adapted from Wyrm [1]_ toolbox https://github.com/bbci/wyrm.

    References
    ----------

    .. [1] Schirrmeister, R. T., Springenberg, J. T., Fiederer, L. D. J.,
       Glasstetter, M., Eggensperger, K., Tangermann, M., Hutter, F. & Ball, T. (2017).
       Deep learning with convolutional neural networks for EEG decoding and
       visualization.
       Human Brain Mapping , Aug. 2017. Online: http://dx.doi.org/10.1002/hbm.23730
    """
    if ax is None:
        ax = plt.gca()
    assert len(v) == len(channels), "Should be as many values as channels"
    assert interpolation == "bilinear" or interpolation == "nearest"
    if vmin is None:
        # added by me (robintibor@gmail.com)
        assert vmax is None
        vmin, vmax = -np.max(np.abs(v)), np.max(np.abs(v))
    # what if we have an unknown channel?
    points = [get_channelpos(c, chan_pos_list) for c in channels]
    for c in channels:
        assert get_channelpos(c, chan_pos_list) is not None, (
            "Expect " + c + " to exist in positions"
        )
    z = [v[i] for i in range(len(points))]
    # calculate the interpolation
    #x = [i[0] for i in points]
    #y = [i[1] for i in points]

    x = [i[0] for i in points]
    y = [i[1] + 0.2 for i in points]  # 整体上移 0.1，可以按需要调整

    # interpolate the in-between values

    # 1. 控制插值区域放大比例
    scale = 1  # 比如扩大30%

    # 2. 找到中心点
    x_center = np.mean(x)
    y_center = np.mean(y)

    # 3. 缩放每个坐标点，扩大插值区域
    x_scaled = [(xi - x_center) * scale + x_center for xi in x]
    y_scaled = [(yi - y_center) * scale + y_center for yi in y]

    # 4. 使用放大的坐标范围创建插值网格
    #xx = np.linspace(min(x_scaled), max(x_scaled), 1000)
    #yy = np.linspace(min(y_scaled), max(y_scaled), 1000)

    #xx = np.linspace(-1.5, 1.5, 1000)
    #yy = np.linspace(-1.5, 1.5, 1000)

    xx = np.linspace(min(x), max(x), 1000)
    yy = np.linspace(min(y), max(y), 1000)
    if interpolation == "bilinear":
        xx_grid, yy_grid = np.meshgrid(xx, yy)
        f = interpolate.LinearNDInterpolator(list(zip(x, y)), z)
        zz = f(xx_grid, yy_grid)
        #zz = f(*np.meshgrid(xx, yy))
    else:
        assert interpolation == "nearest"
        f = interpolate.NearestNDInterpolator(list(zip(x, y)), z)
        assert len(xx) == len(yy)
        zz = np.ones((len(xx), len(yy)))
        for i_x in range(len(xx)):
            for i_y in range(len(yy)):
                # somehow this is correct. don't know why :(
                zz[i_y, i_x] = f(xx[i_x], yy[i_y])
                # zz[i_x,i_y] = f(xx[i_x], yy[i_y])
        assert not np.any(np.isnan(zz))

    # plot map
    image = ax.imshow(
        zz,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        #extent=[min(x), max(x), min(y), max(y)],
        extent=[min(xx), max(xx), min(yy), max(yy)],
        #extent=[-1.5, 1.5, -1.5, 1.5],
        origin="lower",
        interpolation=interpolation,
    )
    if scalp_line_width > 0:
        # paint the head
        ax.add_artist(
            plt.Circle(
                (0, 0),
                0.6,
                linestyle=scalp_line_style,
                linewidth=scalp_line_width,
                fill=False,
            )
        )
        # add a nose
        ax.plot(
            [-0.06, 0, 0.06],  # 横向长度缩小
            [0.6, 0.66, 0.6],  # 鼻尖稍微超过圆顶
            color="black",
            linewidth=scalp_line_width,
            linestyle=scalp_line_style,
        )
        # add ears
        _add_ears(ax, scalp_line_width, scalp_line_style)
    # add markers at channels positions
    # set the axes limits, so the figure is centered on the scalp
    #ax.set_ylim([-1.05, 1.15])
    #ax.set_xlim([-1.15, 1.15])

    ax.set_xlim([-0.75, 0.75])
    ax.set_ylim([-0.75, 0.85])

    # hide the frame and ticks
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    # draw the channel names
    if annotate:
        for i in zip(channels, list(zip(x, y))):
            ax.annotate(
                " " + i[0],
                i[1],
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=fontsize,
            )
    ax.set_aspect(1)
    return image


def _add_ears(ax, linewidth, linestyle):
    scale = 0.6  # 根据头部半径缩放

    start_x = np.cos(10 * np.pi / 180.0) * scale
    start_y = np.sin(10 * np.pi / 180.0) * scale
    end_x = np.cos(-15 * np.pi / 180.0) * scale
    end_y = np.sin(-15 * np.pi / 180.0) * scale

    verts = [
        (start_x, start_y),
        (start_x + 0.05 * scale, start_y + 0.05 * scale),
        (start_x + 0.1 * scale, start_y),
        (start_x + 0.11 * scale, (end_y * 0.7 + start_y * 0.3)),
        (end_x + 0.14 * scale, end_y),
        (end_x + 0.05 * scale, end_y - 0.05 * scale),
        (end_x, end_y),
    ]

    codes = [Path.MOVETO] + [Path.CURVE3] * (len(verts) - 1)
    path = Path(verts, codes)
    patch = patches.PathPatch(path, facecolor="none", linestyle=linestyle, linewidth=linewidth)
    ax.add_patch(patch)

    # Left ear: mirror over y-axis
    verts_left = [(-x, y) for x, y in verts]
    path_left = Path(verts_left, codes)
    patch_left = patches.PathPatch(path_left, facecolor="none", linestyle=linestyle, linewidth=linewidth)
    ax.add_patch(patch_left)

