import os.path as op
import numpy as np
import matplotlib.pyplot as plt

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
        print arg
        print ks[ii][0]
        print ks[ii][1]
        leg.append('%s,\n $E_0 = %.6f $, $E_1 = %.6f $'
                   % (op.splitext(arg)[0], ks[ii][0], ks[ii][1]))
    return leg

def _plot_curve(ax, dx, dy, xlabel, ylabel, label='', title=None):
    il = len(ax.lines)
    ax.plot(dx, dy, label=label, color=color_vector[il,:3], marker=markers[il])
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
        label = ': '.join((data.name, label))

    return label

def plot_strain_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    _plot_curve(ax, data.time, data.strain, 'time [s]', 'strain [1]',
                label=label, title='strain-time')
    return ax

def plot_stress_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    _plot_curve(ax, data.time, data.stress, 'time [s]', 'stress [MPa]',
                label=label, title='stress-time')
    return ax

def plot_stress_strain(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    _plot_curve(ax, data.strain, data.stress, 'strain [1]', 'stress [MPa]',
                label=label, title='stress-strain')
    return ax

def plot_raw_strain_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    _plot_curve(ax, data.time, data.raw_strain, 'time [s]', 'strain [1]',
                label=label, title='raw strain-time')
    return ax

def plot_raw_stress_time(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    _plot_curve(ax, data.time, data.raw_stress, 'time [s]', 'stress [MPa]',
                label=label, title='raw stress-time')
    return ax

def plot_raw_stress_strain(data, fig_num=1, ax=0, label=''):
    ax = _get_ax(fig_num, ax)
    label = _get_label(data, label)
    _plot_curve(ax, data.raw_strain, data.raw_stress,
                'strain [1]', 'stress [MPa]',
                label=label, title='raw stress-strain')
    return ax

def plot_cycles_time(data, fig_num=1, ax=0):
    ax = _get_ax(fig_num, ax)
    ymin, ymax = ax.axis()[2:]

    for ii in data.cycles:
        plt.vlines(data.full_time[ii.start], ymin, ymax)

    return ax

def show(**kwargs):
    plt.show()

def save_figure(fig_num=1, suffix='', filename=''):
    fig = plt.figure(fig_num)
    if not filename:
        if not suffix:
            suffix = 'png'

        ax = fig.gca()
        title = ax.title.get_text()
        filename = title.replace(' ', '_') + '.' + suffix

    else:
        if suffix:
            filename += '.' + suffix

    fig.savefig(filename)
