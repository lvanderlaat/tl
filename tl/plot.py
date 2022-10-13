#!/usr/bin/env python


"""
"""


# Python Standard Library
import warnings

# Other dependencies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local files
from . import features


__author__ = 'Leonardo van der Laat'
__email__  = 'laat@umich.edu'


warnings.filterwarnings('ignore')


def misfits_histograms(df_true, df_pred, bins=50):
    diff = df_pred - df_true
    diff.z *= -1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2.1))

    fig.suptitle('misfit = pred - true')

    fig.subplots_adjust(
        left=.03,
        bottom=.19,
        right=.96,
        top=.8,
        wspace=.2,
        hspace=.2
    )

    ax1.set_xlabel('East [m]')
    ax2.set_xlabel('North [m]')
    ax3.set_xlabel('Depth [m]')

    for ax, dim in zip([ax1, ax2, ax3], 'x y z'.split()):
        d = diff[dim].values
        ax.axvline(d.mean(), lw=0.8, ls='--', c='r')
        ax.set_title(f'$\mu_{dim} = {int(d.mean())} \pm {int(d.std())}$', fontsize=10)
        xmin = np.quantile(d, 0.005)
        xmax = np.quantile(d, 0.995)
        ax.hist(d[np.where((d >= xmin) & (d <= xmax))], bins=bins)

        ax.get_yaxis().set_visible(False)
        ax.get_yaxis().set_ticks([])
        for position in 'left right top'.split():
            ax.spines[position].set_visible(False)
    return fig


def square_subplots(fig):
    ax1 = fig.get_axes()[0]
    rows, cols = ax1.get_subplotspec().get_gridspec().get_geometry()

    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom

    wspace = fig.subplotpars.wspace
    hspace = fig.subplotpars.hspace

    figw, figh = fig.get_size_inches()

    axw = figw*(r-l)/(cols+(cols-1)*wspace)
    axh = figh*(t-b)/(rows+(rows-1)*hspace)
    axs = min(axw, axh)
    w = (1-axs/figw*(cols+(cols-1)*wspace))/2.5
    h = (1-axs/figh*(rows+(rows-1)*hspace))/2.5

    fig.subplots_adjust(bottom=h, top=1-h, left=w, right=1-w)
    return fig


def locs_test_sample(
    df_true, df_pred, sta, n_events=100, s=10,
    figwidth=4.6, width=0.65, bottom=0.05,
):
    # Just show a number of events
    df1 = df_true.copy()
    df2 = df_pred.copy()

    if len(df1) > n_events:
        index = np.linspace(0, len(df_true)-1, n_events, dtype=int)

        df1.sort_values(by='misfit', inplace=True, ignore_index=True)
        df2.sort_values(by='misfit', inplace=True, ignore_index=True)

        df1 = df1.loc[index]
        df2 = df2.loc[index]


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.5), sharex=True)
    fig.subplots_adjust(bottom=bottom, hspace=0)

    ax1.set_ylabel('Cartesian northing [m]')
    ax2.set_xlabel('Cartesian easting [m]')
    ax2.set_ylabel('Depth [m]')

    vmin = np.quantile(df_true.misfit, 0.05)
    vmax = np.quantile(df_true.misfit, 0.95)

    # Map
    ax1.scatter(df1.x, df1.y, c='k', marker='x', label='True', cmap='inferno')
    scatter = ax1.scatter(
        df2.x, df2.y, s=s,
        c=df2.misfit, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        label='Predicted', cmap='inferno'
    )
    ax1.legend()

    # Profile
    ax2.scatter(df1.x, df1.z, c='k', marker='x', cmap='inferno')
    ax2.scatter(
        df2.x, df2.z, s=s,
        c=df2.misfit, norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap='inferno',
    )

    # Colorbar
    cbar_ax = fig.add_axes([0.86, 0.33, 0.01, 0.3])
    fig.colorbar(scatter, cax=cbar_ax, label='Misfit [m]')

    for i, row in df1.iterrows():
        ax1.plot([df1.loc[i].x, df2.loc[i].x], [df1.loc[i].y, df2.loc[i].y], c='k', lw=1)
        ax2.plot([df1.loc[i].x, df2.loc[i].x], [df1.loc[i].z, df2.loc[i].z], c='k', lw=1)

    # Stations
    ax1.scatter(sta.x, sta.y, s=25, c='k', marker='^')
    ax2.scatter(sta.x, sta.z, s=25, c='k', marker='^')

    # Layout
    # adjust_map_section(fig, ax1, ax2, figwidth, width, bottom)
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    square_subplots(fig)
    return fig


def locations(df, sta, s=1, figwidth=4.6, width=0.65, bottom=0.05):

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.6, 5.5), sharex=True)
    fig.subplots_adjust(bottom=bottom, hspace=0)

    ax1.set_ylabel('Cartesian northing [m]')
    ax2.set_xlabel('Cartesian easting [m]')
    ax2.set_ylabel('Depth [m]')

    # Map
    ax1.scatter(df.x, df.y, c='r', s=s)
    ax2.scatter(df.x, df.z, c='r', s=s)

    # Stations
    ax1.scatter(sta.x, sta.y, s=25, c='k', marker='^')
    ax2.scatter(sta.x, sta.z, s=25, c='k', marker='^')

    for ax in [ax1, ax2]:
        ax.grid('on')

    # Layout
    # adjust_map_section(fig, ax1, ax2, figwidth, width, bottom)

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    square_subplots(fig)
    return fig


def adjust_maps_sections(fig, ax1, ax2, ax3, ax4, figwidth, width, bottom):
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    zmin, zmax = ax3.get_ylim()

    range_x = xmax - xmin
    range_y = ymax - ymin
    range_z = zmax - zmin

    range_v = range_y + range_z

    pos_1 = ax1.get_position()
    pos_2 = ax2.get_position()
    pos_3 = ax3.get_position()
    pos_4 = ax4.get_position()

    height_1 = range_y/range_x * width/2
    height_3 = range_z/range_x * width/2

    y0_3 = bottom
    y0_1 = y0_3 + height_3

    x0_2 = pos_1.x0 + width/2

    ax1.set_position([pos_1.x0, y0_1, width/2, height_1])
    ax3.set_position([pos_1.x0, y0_3, width/2, height_3])

    ax2.set_position([x0_2, y0_1, width/2, height_1])
    ax4.set_position([x0_2, y0_3, width/2, height_3])

    figheight = range_v/range_x*figwidth
    figheight -= 0.48*figheight
    margin = (1-width)/2

    fig.set_size_inches(figwidth, figheight)

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_aspect('equal')

    fig.subplots_adjust(
        # left=margin,
        bottom=bottom,
        # # right=1-margin,
        # top=height_1+height_3+bottom,
        hspace=.0,
        wspace=0.05
    )
    # ax1.get_shared_x_axes().join(ax1, ax2, ax3, ax4)
    # ax1.get_shared_y_axes().join(ax1, ax2)
    # ax3.get_shared_y_axes().join(ax3, ax4)
    fig.tight_layout()

    return


def locs_true_pred(
    df_true, df_pred, sta, s=0.05, figwidth=7.4, width=0.5, bottom=0.1,
    cmap='inferno',
):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.subplots_adjust(
        bottom=bottom,
        hspace=.0,
        wspace=0
    )

    ax1.set_ylabel('Cartesian northing [m]')
    ax3.set_ylabel('Elevation [m a.s.l.]')

    vmin = np.quantile(df_true.misfit, 0.05)
    vmax = np.quantile(df_true.misfit, 0.95)

    # True locations
    # Map
    scatter = ax1.scatter(
        df_true.x, df_true.y, s=s, c=df_true.misfit,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax,), cmap=cmap, rasterized=True
    )
    # Profile
    ax3.scatter(
        df_true.x, df_true.z, s=s, c=df_true.misfit,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, rasterized=True
    )

    # Predicted locations
    # Map
    ax2.scatter(
        df_pred.x, df_pred.y, s=s, c=df_pred.misfit,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, rasterized=True
    )
    # Profile

    ax4.scatter(
        df_pred.x, df_pred.z, s=s, c=df_pred.misfit,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap, rasterized=True
    )

    cbar_ax = fig.add_axes([0.91, 0.33, 0.01, 0.3])
    fig.colorbar(scatter, cax=cbar_ax, label='Misfit [m]')

    for ax in fig.get_axes():
        ax.grid('on')
        ax.set_rasterization_zorder(0)

    for ax in [ax1, ax2]:
        ax.scatter(
            sta.x, sta.y, s=50, c='k', marker='^')
        ax.set_xticklabels([])
    for ax in [ax3, ax4]:
        ax.scatter(
            sta.x, sta.z, s=50, c='k', marker='^')
        ax.set_xlabel('Cartesian easting [m]')
    for ax in [ax2, ax4]:
        ax.set_yticklabels([])


    adjust_maps_sections(fig, ax1, ax2, ax3, ax4, figwidth, width, bottom)

    ax1.set_title('True locations')
    ax2.set_title('Predicted locations')
    return fig


def adjust_map_section(fig, ax1, ax2, figwidth, width, bottom):
    xmin, xmax = ax1.get_xlim()
    ymin, ymax = ax1.get_ylim()
    zmin, zmax = ax2.get_ylim()

    range_x = xmax - xmin
    range_y = ymax - ymin
    range_z = zmax - zmin

    range_v = range_y + range_z

    pos_1 = ax1.get_position()
    pos_2 = ax2.get_position()

    height_1 = range_y/range_x * width
    height_2 = range_z/range_x * width

    y0_2 = bottom
    y0_1 = y0_2 + height_2

    ax1.set_position([pos_1.x0, y0_1, width, height_1])
    ax2.set_position([pos_1.x0, y0_2, width, height_2])

    ax1.set_xticklabels([])

    ax1.set_aspect('equal')
    ax2.set_aspect('equal')

    figheight = range_v/range_x*figwidth
    figheight -= 0.2*figheight
    margin = (1-width)/2

    fig.set_size_inches(figwidth, figheight)
    fig.subplots_adjust(
        left=margin,
        bottom=bottom,
        right=1-margin,
        top=height_1+height_2+bottom,
        hspace=.0
    )
    # fig.tight_layout()
    return


def synth_misfit(
    dfh, dfv, sta, y_plot, z_plot, xh, yh, xv, zv, step,
    cmap='viridis', vmin=None, vmax=None, levels=[1e2, 1e3, 5e3],
    figwidth=4.6, width=0.65, bottom=0.05,
):

    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.subplots_adjust(bottom=bottom, hspace=0)
    ax1.set_ylabel('Cartesian northing [km]')
    ax2.set_xlabel('Cartesian easting [km]')
    ax2.set_ylabel('Depth [km]')

    im = ax1.imshow(
        np.rot90(dfh.misfit.values.reshape((len(yh), len(xh))).T),
        cmap=cmap,
        norm=colors.LogNorm(),
        extent=[
            dfh.x.min()-step/2,
            dfh.x.max()+step/2,
            dfh.y.min()-step/2,
            dfh.y.max()+step/2,
        ],
        vmin=vmin, vmax=vmax,
    )
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Misfit [m]')

    contour = ax1.tricontour(dfh.x, dfh.y, dfh.misfit, colors='k', levels=levels)
    ax1.clabel(contour, contour.levels, fmt='%0.f m')


    im = ax2.imshow(
        dfv.misfit.values.reshape((len(zv), len(xv))),
        cmap=cmap,
        norm=colors.LogNorm(),
        extent=[
            dfv.x.min()-step/2,
            dfv.x.max()+step/2,
            dfv.z.min()-step/2,
            dfv.z.max()+step/2,
        ],
        vmin=vmin, vmax=vmax,
    )
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    plt.colorbar(im, cax=cax, label='Misfit [m]')

    contour = ax2.tricontour(dfv.x, dfv.z, dfv.misfit, colors='k', levels=levels)
    ax1.clabel(contour, contour.levels, fmt='%0.f m')

    ax1.axhline(y_plot, c='k', ls='--')
    ax2.axhline(z_plot, c='k', ls='--')

    # Stations
    ax1.scatter(sta.x, sta.y, s=70, c='k', marker='^', ec='w', lw=0.7, zorder=100)
    ax2.scatter(sta.x, sta.z, s=70, c='k', marker='^', ec='w', lw=0.7, zorder=100)

    adjust_map_section(fig, ax1, ax2, figwidth, width, bottom)

    return fig


def amp_dist_mag(df, bands, channels, cha=['HHZ', 'EHZ'],  s=10, cols=2, rows=2):
    fc = 0.9
    fig = plt.figure(figsize=(7.4, 6))

    position = range(1, len(bands)+1, 1)

    for i, band in enumerate(bands[:4]):
        meta = features.to_dataframe(channels[channels.channel.isin(cha)], [band])

        columns = 'x y z magnitude'.split() + list(meta.key)

        df_band = df[columns]
        distance_keys = []
        for _, row in meta.iterrows():
            key = row.station + '_dist'
            distance_keys.append(key)
            df_band[key] = np.sqrt(
                (df_band.x - row.x)**2 + (df_band.y - row.y)**2 + (df_band.z - row.z)**2
            )

        _dfs = []
        for key, distance_key in zip(meta.key, distance_keys):
            _df = df_band[[key, distance_key, 'magnitude']]
            _df.columns = ['amplitude', 'distance', 'magnitude']
            _dfs.append(_df)

        df_ = pd.concat(_dfs, ignore_index=True)
        df_.distance *= 1e-3
        df_.sort_values(by='amplitude', ascending=False, inplace=True)

        vmin = df_.amplitude.quantile(.005)
        vmax = df_.amplitude.quantile(.995)

        ax = fig.add_subplot(rows, cols, position[i])
        ax.set_xlabel('Magnitude')
        ax.set_ylabel('Distance [km]')
        if len(bands) > 1:
            ax.set_title(f'{band[0]} - {band[1]} Hz')

        scatter = ax.scatter(
            df_.magnitude, df_.distance, c=df_.amplitude,
            s=s, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap='turbo',
            rasterized=True, zorder=-10
        )
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(scatter, cax=cax, label='Amplitude [$\mu m/s$]')
        ax.set_facecolor((fc, fc, fc))
    fig.tight_layout()

    return fig


def amp_timeseries(df):
    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(
        left=.1,
        bottom=.1,
        right=.9,
        top=.9,
        wspace=.2,
        hspace=.2
    )

    ax = fig.add_subplot(111)
    ax.set_ylabel('Amplitude [$\mu m/s$]')
    for column in df.columns:
        if '_' in column:
            ax.plot(df[column], lw=0.3, alpha=0.5, rasterized=True)
    ax.set_yscale('log')
    return fig


def ratio_timeseries(df, meta):
    df = df[meta.key]

    fig = plt.figure(figsize=(6, 4))
    fig.subplots_adjust(
        left=.1,
        bottom=.1,
        right=.9,
        top=.9,
        wspace=.2,
        hspace=.2
    )
    ax = fig.add_subplot(111)
    ax.set_ylabel('Amplitude ratio')

    n_features = len(meta)
    for i in range(0, n_features-1):
        for j in range(i+1, n_features):
            feature_i = meta.iloc[i]
            feature_j = meta.iloc[j]

            # Do not compute ratios between the same station
            if feature_i.station == feature_j.station:
                continue
            pair = f'{feature_i.key}_{feature_j.key}'
            ratio = df[feature_i.key] / df[feature_j.key]
            ax.scatter(df.index, ratio, lw=0, alpha=0.7, s=0.1, rasterized=True)
    ax.set_yscale('log')
    return fig


def loc_timeseries(df):
    fig, axes = plt.subplots(figsize=(7.4, 6), nrows=3, ncols=1, sharex=True)
    fig.subplots_adjust(hspace=0)

    axes[0].set_ylabel('Cartesian easting [m]')
    axes[1].set_ylabel('Cartesian northing [m]')
    axes[2].set_ylabel('Elevation [m]')

    for ax, dim in zip(axes, ['x', 'y', 'z']):
        ax.scatter(df.index, df[dim], s=10, rasterized=True)
        ax.grid('on')

    return fig


if __name__ == '__main__':
    pass
