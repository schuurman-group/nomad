"""
Module of all plotting routines used in FMSinterpreter.
This should be a stand-alone module which handles all calls to
Matplotlib. The Figure object can be used to handle multiple
plots on single or multiple canvases.
Note that just by importing this module, the default theme will be
reset according to set_theme().
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
from cycler import cycler
plt.rc('axes', axisbelow=True)
plt.rc('legend', edgecolor='none')


def set_theme(color=['#2ca02c', '#1f77b4', '#d62728', '#9467bd', '#ff7f0e',
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
              linestyle=['-', '--', ':', '-.'],
              marker=['', '.', 'o', 's', '^', '*', '+', 'x'],
              order=['color', 'linestyle', 'marker']):
    """Sets colours, linetypes and markers to a given theme.
    The pre-set values are the Matplotlib default color scheme followed
    by the same scheme with linestyles, then markers. Changing the keyword
    'order' will change the cycle order. By default, colours are cycled
    through first, followed by linestyles, then markers.
    For a black and white theme, use color='k'.
    """
    # set custom default color/line/marker cycle
    def_order = ['color', 'linestyle', 'marker']
    if isinstance(order, str):
        def_order.remove(order)
        order += def_order
    elif len(order) < 3:
        for i in order:
            def_order.remove(i)
        order += def_order
    plt.rc('axes', prop_cycle=(
        cycler(order[2], locals()[order[2]]) *
        cycler(order[1], locals()[order[1]]) *
        cycler(order[0], locals()[order[0]])
                               ))

    # set custom default colormap
    vir = plt.cm.get_cmap('viridis_r')
    clisti = vir(np.linspace(0, 1, vir.N - 28))
    clistn = np.array([np.linspace(1, clisti[0,i], 29) for i in range(4)]).T
    clist = np.vstack((clistn[:-1], clisti))
    wir = col.LinearSegmentedColormap.from_list('wiridis', clist)
    plt.register_cmap(cmap=wir)
    plt.rc('image', cmap='wiridis')


def Figure(object):
    """An object which handles matplotlib objects needed for figures."""
    def __init__(self, subplots=(1,1), **kwargs):
        self.subplots = subplots
        self.fig, axarr = plt.subplots(*subplots, **kwargs)
        self.axarr = np.atleast_2d(axarr)

        # get array of subplot indices
        max0, max1 = subplots
        inds = np.mgrid[:max0, :max1]
        inds = np.rollaxis(inds, 0, 3)
        self.inds = inds.reshape((max0*max1, 2))

    def lineplot(self, x, y, isub=(0,0), **kwargs):
        """Plots a line from data y vs. x in the subplot position isub."""
        ax = self.axarr[isub[0], isub[1]]
        ax.plot(x, y, **kwargs)

    def scatter(self, x, y, isub=(0,0), **kwargs):
        """Plots a scatter plot of x and y positions in the subplot
        position isub."""
        ax = self.axarr[isub[0], isub[1]]
        ax.scatter(x, y, **kwargs)

    def heatmap(self, x, y, z, isub=(0,0), **kwargs):
        """Plots a heatmap of z vs. x and y in the subplot position isub."""
        ax = self.axarr[isub[0], isub[1]]
        ax.pcolormesh(x, y, z, rasterized=True, **kwargs)

    def contour(self, x, y, z, isub=(0,0), **kwargs):
        """Plots a contour plot of z vs. x and y in the subplot
        position isub."""
        lvl = np.arange(np.floor(min(z)), np.ceil(max(z)), 0.5)
        ax = self.axarr[isub[0], isub[1]]
        cs = ax.contour(x, y, z, lvl, **kwargs)
        ax.setp(cs.collections[1::2], alpha=0.5)
        ax.clabel(cs, cs.levels[::2], inline=1, fmt=r'%.2f')

    def contourf(self, x, y, z, isub=(0,0), **kwargs):
        """Plots a filled countour plot of z vs. x and y in the subplot
        position isub."""
        lvls = np.linspace(np.min(z), np.max(z), 150)
        ax = self.axarr[isub[0], isub[1]]
        ax.contourf(x, y, z, levels=lvls, zorder=-9, **kwargs)
        ax.set_rasterization_zorder(-1)

    def set_xlabel(self, label, isub=None):
        """Sets the x-axis label on subplot(s) in position isub. If no isub
        is given, the label is set for all subplots."""
        for i in _get_ind(isub):
            ax = self.axarr[i[0], i[1]]
            ax.set_xlabel(label)

    def set_ylabel(self, label, isub=None):
        """Sets the y-axis label on subplot(s) in position isub. If no isub
        is given, the label is set for all subplots."""
        for i in _get_ind(isub):
            ax = self.axarr[i[0], i[1]]
            ax.set_xlabel(label)

    def set_xlim(self, lim, isub=None):
        """Sets the x-axis limit on subplot(s) in position isub. If no isub
        is given, the label is set for all subplots."""
        for i in _get_ind(isub):
            ax = self.axarr[i[0], i[1]]
            ax.set_xlim(lim)

    def set_ylim(self, lim, isub=None):
        """Sets the y-axis limit on subplot(s) in position isub. If no isub
        is given, the label is set for all subplots."""
        for i in _get_ind(isub):
            ax = self.axarr[i[0], i[1]]
            ax.set_ylim(lim)

    def set_legend(self, labels, isub=None):
        """Sets the legend labels on subplots(s) in position isub. If no isub
        is given, the label is set for all subplots.
        In the future, isub=None default behaviour should set a legend
        outside of all subplots.
        """
        for i in _get_ind(isub):
            ax = self.axarr[i[0], i[1]]
            ax.set_legend(labels, loc='best')

    def _get_ind(self, ind):
        """Returns list of all indices if None, otherwise returns a
        2D array."""
        if ind is None:
            return self.inds
        else:
            return np.atleast_2d(ind)


def lineplot(x, y, err=None, xlabel='x', ylabel='y', xlim=None, ylim=None,
             legend=None, **kwargs):
    """Plots a line from data y vs. x in a single frame."""
    fig, ax = plt.subplots()
    if err is not None:
        for i in range(y.shape[1]):
            ax.fill_between(x, y[:,i] + err[:,i], y[:,i] - err[:,i], alpha=0.2)
    ax.plot(x, y, **kwargs)

    _ax_set(ax, xlabel, ylabel, _get_lim(x,xlim), _get_lim(y,ylim), legend)
    return fig, ax


def scatter(x, y, xlabel='x', ylabel='y', xlim=None, ylim=None,
            legend=None, transp=None, **kwargs):
    """Plots a scatter plot of x and y positions in a single frame."""
    fig, ax = plt.subplots()
    if transp is None:
        colours = 'k'
    else:
        colours = np.zeros((len(transp), 4))
        colours[:,3] = transp
    ax.scatter(x, y, color=colours, lw=0, **kwargs)

    _ax_set(ax, xlabel, ylabel, _get_lim(x,xlim), _get_lim(y,ylim), legend)
    return fig, ax


def contour(x, y, z, xlabel='x', ylabel='y', xlim=None, ylim=None,
            legend=None, **kwargs):
    """Plots a contour plot of z vs. x and y in a single frame."""
    lvl = np.arange(np.floor(np.min(z)), np.ceil(np.max(z)), 0.5)
    fig, ax = plt.subplots()
    cs = ax.contour(x, y, z, lvl, **kwargs)
    plt.setp(cs.collections[1::2], alpha=0.5)
    ax.clabel(cs, cs.levels[::2], inline=1, fmt=r'%.2f')

    _ax_set(ax, xlabel, ylabel, _get_lim(x,xlim), _get_lim(y,ylim), legend)
    return fig, ax


def contourf(x, y, z, xlabel='x', ylabel='y', xlim=None, ylim=None,
             legend=None, **kwargs):
    """Plots a filled countour plot of z vs. x and y in a single frame."""
    fig, ax = plt.subplots()
    lvls = np.linspace(np.min(z), np.max(z), 150)
    l1 = ax.contourf(x, y, z, levels=lvls, zorder=-9, **kwargs)
    ax.set_rasterization_zorder(-1)

    _ax_set(ax, xlabel, ylabel, _get_lim(x,xlim), _get_lim(y,ylim), legend)
    fig.colorbar(l1)
    return fig, ax


def heatmap(x, y, z, xlabel='x', ylabel='y', xlim=None, ylim=None,
            legend=None, **kwargs):
    """Plots a heatmap of z vs. x and y in a single frame."""
    fig, ax = plt.subplots()
    l1 = ax.pcolormesh(x, y, z, rasterized=True, **kwargs)

    _ax_set(ax, xlabel, ylabel, _get_lim(x,xlim), _get_lim(y,ylim), legend)
    fig.colorbar(l1)
    return fig, ax


def energyplot(lbl, y, wid=1, sep=1, rot=90, maxe=None, grid=False):
    """Plots a connected bar plot used to represent potential energies."""
    fig, ax = plt.subplots()

    yplot = np.repeat(y, 2, axis=0)
    x1 = np.arange(len(y)) * (wid + sep)
    x2 = np.insert(x1 + wid, range(len(x1)), x1)

    ax.plot(x2, yplot)
    for i in range(len(y[0])):
        ax.hlines(y[:,i], x1, x1 + wid, linewidth=2, zorder=3)

    ax.set_xticks(x1 + 0.5*wid)
    ax.set_xticklabels(lbl, rotation=rot)
    ax.yaxis.grid(grid)

    _ax_set(ax, ylabel='Energy / eV', xlim=(x2[0] - wid, x2[-1] + wid),
            ylim=(-0.1) if maxe is None else (-0.1, maxe))
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('left')
    return fig, ax


def save(fname, figure=None, dpi=300):
    """Saves a provided figure (or the current figure) to
    a given filename.
    Note that the 'dpi' flag only affects raster filetypes and
    plots with rasterized=True (e.g. heatmaps).
    """
    if figure is None:
        plt.savefig(fname, bbox_inches='tight', dpi=dpi)
    else:
        figure.savefig(fname, bbox_inches='tight', dpi=dpi)


def _ax_set(ax, xlabel=None, ylabel=None, xlim=None, ylim=None, legend=None):
    """Sets basic properties of a given plot axis."""
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if legend is not None:
        ax.legend(legend, loc='best')


def _get_lim(q, qlim):
    """Gets the limits for a coordinate q."""
    if qlim == 'range':
        return min(q), max(q)
    else:
        return qlim


set_theme()
