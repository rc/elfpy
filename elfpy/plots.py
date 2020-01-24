import os.path as op
import numpy as np
import matplotlib.pyplot as plt

from elfpy.filters import _parse_list_of_ints
from elfpy.dataio import _get_filename

class Cycler(list):
    def __init__(self, sequence):
        list.__init__(self, sequence)
        self.n_max = len(sequence)

    def __getitem__(self, ii):
        return list.__getitem__(self, ii % self.n_max)

def define_plot_attributes():
    colors = Cycler([(0.0, 0.0, 0.0),
                     (0.6, 0.0, 1.0),
                     (1.0, 1.0, 0.0), (1.0, 0.0, 0.8),
                     (0.5, 1.0, 0.5), (0.5, 1.0, 1.0),
                     (0.8, 0.0, 0.2), (1.0, 0.0, 0.0),
                     (0.0, 0.0, 0.4), (0.0, 0.0, 1.0), (0.0, 0.75, 0.0),])

    markers = Cycler(['o', 'v', 's','^', '<',  'D', '>', 'x', 'p', 'h', '+'])

    linestyles = Cycler(['-', '--', '-.', ':'])

    color_vector=[]
    for ii in range(0,6):
        color_vector = np.append(color_vector, colors)
    color_vector = np.reshape(color_vector, (66,3))

    return markers, linestyles, color_vector

markers, linestyles, color_vector = define_plot_attributes()

def make_colors(num):
    """
    Make `num` continuously changing rainbow-like RGB colors.
    """
    def g(n):
        """
        Map sine [-1.0 .. 1.0] => color byte [0 .. 255].
        """
        return 255 * (n + 1) / 2.0

    def f(start, stop, num):

        interval = (stop - start) / num

        for n in range(num):
            coefficient = start + interval * n
            yield g(np.sin(coefficient * np.pi))

    red = f(0.5, 1.5, num)
    green = f(1.5, 3.5, num)
    blue = f(1.5, 2.5, num)

    rgbs = [('#%02x%02x%02x' % rgb) for rgb in zip(red, green, blue)]
    return rgbs

plot_options = {
    'linewidth' : 1,
}

data_options = {
    'sampling' : 1,
    'use_markers' : 1,
}

def make_legend_text(args, ks):
    """
    Make a text of a legend.

    Parameters
    ----------
    - args : string
    - ks : float, float

    Returns
    -------
    - leg : string
    """

    leg = []
    for ii, arg in enumerate(args):
        output(arg)
        output(ks[ii][0])
        output(ks[ii][1])
        leg.append('%s,\n $E_0 = %.6f $, $E_1 = %.6f $'
                   % (op.splitext(arg)[0], ks[ii][0], ks[ii][1]))
    return leg

def _plot_curve(ax, dx, dy, xlabel, ylabel, label='',
                color=None, title=None, iline=None):
    il = len(ax.lines) if iline is None else iline
    color = color_vector[il, :3] if color is None else color
    marker = markers[il] if data_options['use_markers'] else None

    ax.plot(dx, dy, label=label, color=color, marker=marker, **plot_options)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is not None:
        plt.title(title)

def _get_ax(fig_num, ax):
    fig = plt.figure(fig_num)

    if ax is None:
        fig.clf()
        ax = fig.add_subplot(111)

    return ax

def _get_label(data, label):
    if label:
        if label == '%':
            label = data.name

        else:
            label = ': '.join((data.name, label))

    return label

def _get_data(dx, dy):
    ic = data_options['sampling']

    length = dx.shape[0]
    num = int(np.ceil(float(length) / ic))
    ii = np.linspace(0, length - 1, num, dtype=np.int)

    return dx[ii], dy[ii]

def set_sampling(data, sampling=1, **kwargs):
    data_options['sampling'] = sampling

def use_markers(data, use=1, **kwargs):
    data_options['use_markers'] = use

def set_plot_option(data, option='', value=0.0, **kwargs):
    plot_options[option] = value

def plot_strain_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    dx, dy = _get_data(data.time, data.strain)
    _plot_curve(ax, dx, dy, 'time [s]', 'strain [1]',
                label=label, title='strain-time')
    return ax

def plot_stress_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    dx, dy = _get_data(data.time, data.stress)
    _plot_curve(ax, dx, dy, 'time [s]', 'stress [MPa]',
                label=label, title='stress-time')
    return ax

def plot_stress_strain(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    dx, dy = _get_data(data.strain, data.stress)
    _plot_curve(ax, dx, dy, 'strain [1]', 'stress [MPa]',
                label=label, title='stress-strain')
    return ax

def plot_raw_strain_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    dx, dy = _get_data(data.time, data.raw_strain)
    _plot_curve(ax, dx, dy, 'time [s]', 'strain [1]',
                label=label, title='raw strain-time')
    return ax

def plot_raw_stress_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    dx, dy = _get_data(data.time, data.raw_stress)
    _plot_curve(ax, dx, dy, 'time [s]', 'stress [MPa]',
                label=label, title='raw stress-time')
    return ax

def plot_raw_stress_strain(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    dx, dy = _get_data(data.raw_strain, data.raw_stress)
    _plot_curve(ax, dx, dy, 'strain [1]', 'stress [MPa]',
                label=label, title='raw stress-strain')
    return ax

def _plot_cycles_colors(data, ax, label, ics):
    colors = make_colors(len(ics))

    nc = len(data.cycles)
    for ii, ic in enumerate(ics):
        ic = ic if ic >= 0 else nc + ic
        irange = data.cycles[ic]
        dx, dy = _get_data(data.strain[irange], data.stress[irange])
        lab = label + '_%d' % ic
        _plot_curve(ax, dx, dy, 'strain [1]', 'stress [MPa]', label=lab,
                    color=colors[ii], title='stress-strain cycles', iline=ii)

def plot_cycles_colors(data, fig_num=1, ax=0, label='', odd=1, even=1,
                       cut_last=0):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)

    if not (odd or even): return ax

    ics = data.get_cycle_indices(odd, even, cut_last)
    _plot_cycles_colors(data, ax, label, ics)

    return ax

def plot_cycles_colors_list(data, fig_num=1, ax=0, label='', ics=[0]):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)

    _plot_cycles_colors(data, ax, label, ics)

    return ax
plot_cycles_colors_list._elfpy_arg_parsers = {'ics' : _parse_list_of_ints}

def plot_cycles_time(data, fig_num=1, ax=0):
    ax = _get_ax(fig_num, ax)
    ymin, ymax = ax.axis()[2:]

    for ii in data.cycles:
        plt.vlines(data.full_time[ii.start], ymin, ymax)
        plt.vlines(data.full_time[ii.stop], ymin, ymax, colors='b')

    return ax

def mark_ultimate_strain(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)

    plt.plot(data.time[data.iult], data.ultimate_strain, 'k*',
             ms=20, label=label)

    return ax

def mark_ultimate_stress(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)

    plt.plot(data.time[data.iult], data.ultimate_stress, 'k*',
             ms=20, label=label)

    return ax

def mark_ultimate_values(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)

    plt.plot(data.ultimate_strain, data.ultimate_stress, 'k*',
             ms=20, label=label)

    return ax

def _plot_region_line(data, irange, color, linewidth, label):
    if irange is None: return

    ii = [irange.start, irange.stop - 1]
    plt.plot(data.strain[ii], data.stress[ii], color,
             linewidth=linewidth, label=label)

def _mark_regions(data, iranges, ax, label=''):
    label = _get_label(data, label)

    n_colors = len(iranges)
    if n_colors > 2:
        colors = make_colors(n_colors)

    else:
        colors = ['b', 'r']

    for ii, irange in enumerate(iranges):
        _plot_region_line(data, irange, colors[ii], 5, label)

def mark_strain_regions(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    _mark_regions(data, data.strain_regions_iranges, ax, label)

    return ax

def mark_stress_regions(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    _mark_regions(data, data.stress_regions_iranges, ax, label)

    return ax

def _plot_fit_line(data, irange, coefs, color, linewidth, label):
    if coefs is None: return

    ii = [irange.start, irange.stop - 1]
    strain = data.strain[ii]
    stress = coefs[0] * strain + coefs[1]
    plt.plot(strain, stress, color, linewidth=linewidth, label=label)

def _mark_regions_fits(data, iranges, lin_fits, ax):
    n_colors = len(lin_fits)
    if n_colors > 2:
        colors = make_colors(n_colors)

    else:
        colors = ['b', 'r']

    for ii, (ik, fit) in enumerate(lin_fits):
        irange = iranges[ik]
        _plot_fit_line(data, irange, fit, colors[ii], 5, '%d' % ik)

def mark_strain_regions_fits(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    _mark_regions_fits(data, data.strain_regions_iranges,
                       data.strain_regions_lin_fits, ax)

    return ax

def mark_stress_regions_fits(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    _mark_regions_fits(data, data.stress_regions_iranges,
                       data.stress_regions_lin_fits, ax)

    return ax

def mark_cycles_fits(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)

    colors = make_colors(len(data.cycles_lin_fits))

    for ii, (ic, fit) in enumerate(data.cycles_lin_fits):
        _plot_fit_line(data, data.cycles[ic], fit, colors[ii], 5, '%d' % ic)

    return ax

def plot_cycles_fits_stiffness(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)

    dx, dy = [], []
    for ii, (ic, fit) in enumerate(data.cycles_lin_fits):
        dx.append(ic)
        dy.append(fit[0])

    _plot_curve(ax, dx, dy, 'cycle [1]', 'stiffness [MPa]',
                label=label, title='stiffness per cycle')

    return ax

def mark_strain_of_stress(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)

    for ic, (strain, stress) in enumerate(data.strains_of_stresses):
        plt.plot(strain, stress, 'k*', ms=20, label=label)

    return ax

def show(datas, **kwargs):
    for ifig in plt.get_fignums():
        fig = plt.figure(ifig)
        fig.tight_layout()

    plt.show()

def save_figure(datas, fig_num=1, suffix='', filename=''):
    fig = plt.figure(fig_num)

    if not filename:
        ax = fig.gca()
        title = ax.title.get_text()
        filename = title.replace(' ', '_')

    if not suffix:
        suffix = 'png'

    filename = _get_filename(datas, filename, '', suffix)

    fig.savefig(filename, bbox_inches='tight')
