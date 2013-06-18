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

def plot_strain_time(data, fig_num=1, ax=False):
    fig = plt.figure(fig_num)

    if ax is None:
        fig.clf()
        ax = fig.add_subplot(111)
        plt.title(data.name)

    ax.plot(data.time, data.strain)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('strain [1]')

    return ax

def plot_stress_time(data, fig_num=1, ax=False):
    fig = plt.figure(fig_num)

    if ax is None:
        fig.clf()
        ax = fig.add_subplot(111)
        plt.title(data.name)

    ax.plot(data.time, data.stress)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('stress [MPa]')

    return ax

def plot_stress_strain(data, fig_num=1, ax=False):
    fig = plt.figure(fig_num)

    if ax is None:
        fig.clf()
        ax = fig.add_subplot(111)
        plt.title(data.name)

    ax.plot(data.strain, data.stress)
    ax.set_xlabel('strain [1]')
    ax.set_ylabel('stress [MPa]')

    return ax
