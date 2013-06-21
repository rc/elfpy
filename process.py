#!/usr/bin/env python
"""
Analyze results of mechanical measurements.

Examples
--------

$ python process.py data.txt -f 'smooth_strain:smooth_stress:detect_strain_cycles:reset_strain:reset_stress' -p 'plot_stress_strain,1:plot_stress_time,2:plot_strain_time,2,1'
"""
from optparse import OptionParser
import glob
import copy
import os.path as op
import matplotlib.pyplot as plt

from elfpy.base import output
from elfpy.filters import parse_filter_pipeline
from elfpy.dataio import read_file_info, Data
import elfpy.dataio as dataio
import elfpy.plots as pl

def get_commands(options):
    """
    Get filter and plot commands from options.
    """
    if options.command_file:
        fd = open(options.command_file, 'r')
        cmds = fd.readlines()
        fd.close()

        filter_cmds = []
        plot_cmds = []
        save_cmds = []
        ii = 0
        appends = [plot_cmds.append, save_cmds.append]
        append = filter_cmds.append
        for cmd in cmds:
            cmd = cmd.strip()

            if cmd.startswith('-'):
                append = appends[ii]
                ii += 1
                continue

            elif cmd.startswith('#') or (len(cmd) == 0):
                continue

            append(cmd)

        filter_cmds = ':'.join(filter_cmds)
        plot_cmds = ':'.join(plot_cmds)
        save_cmds = ':'.join(save_cmds)

    else:
        filter_cmds = None
        plot_cmds = None
        save_cmds = None

    if filter_cmds:
        if options.filters:
            filter_cmds = filter_cmds + ':' + options.filters

    else:
        filter_cmds = options.filters

    if plot_cmds:
        if options.plots:
            plot_cmds = plot_cmds + ':' + options.plots

    else:
        plot_cmds = options.plots

    if save_cmds:
        if options.saves:
            save_cmds = save_cmds + ':' + options.saves

    else:
        save_cmds = options.saves

    return filter_cmds, plot_cmds, save_cmds

def read_all_data(filenames):
    directory = op.split(__file__)[0]
    areas = read_file_info(op.join(directory, 'cross_sections.txt'))
    lengths = read_file_info(op.join(directory, 'init_lengths.txt'))

    datas = []
    for i_file, filename in enumerate(filenames):
        data = Data.from_file(filename)
        data.set_initial_values(lengths=lengths, areas=areas)
        datas.append(data)

    return datas

def run_pipeline(filters, plots, saves, datas):
    """
    Apply filters and then plots to datas.
    """
    for ii, flt in enumerate(filters):
        fun, kwargs = flt

        aux = ', '.join(['%s=%s' % kw for kw in kwargs.iteritems()])
        output('applying: %s(%s) ...' % (fun.__name__, aux))

        for ir, data in enumerate(datas):
            output('processing: %s ...' % data.name)

            # The filter action modifies data in-place.
            data = fun(data, **kwargs)

            output('...done')
        output('...done')

        datas[ir] = data

    ax = None
    for ii, plot in enumerate(plots):
        fun, kwargs = plot
        kwargs = copy.copy(kwargs)

        aux = ', '.join(['%s=%s' % kw for kw in kwargs.iteritems()])
        output('applying: %s(%s) ...' % (fun.__name__, aux))

        shared_ax = kwargs.pop('ax', False)
        ax = ax if shared_ax else None

        is_legend = False
        for ir, data in enumerate(datas):
            output('plotting: %s ...' % data.name)

            is_legend = is_legend or kwargs.get('label', '')
            ax = fun(data, ax=ax, **kwargs)

            output('...done')
        output('...done')

        if is_legend:
            plt.legend()

    for ii, save in enumerate(saves):
        fun, kwargs = save

        aux = ', '.join(['%s=%s' % kw for kw in kwargs.iteritems()])
        output('executing: %s(%s) ...' % (fun.__name__, aux))

        fun(datas, **kwargs)

        output('...done')

usage = '%prog [options] filenames\n' + __doc__.rstrip()

_help = {
    'filters' : 'filters that should be applied to data files',
    'plots' : 'plots that should be created for data files',
    'saves' : 'commands to save results into files',
    'no_show' : 'do not show figures',
    'command_file' : 'file with filter commands followed by plot commands.'
    ' The two groups has to be separated by a line with one or several "-"'
    ' characters. The filter commands are pre-pended to commands passed'
    ' using --filters. The plot commands are pre-pended to commands passed'
    ' using --plots.',
}

def main():
    parser = OptionParser(usage=usage, version="%prog ")
    parser.add_option('-f', '--filters',
                      metavar='filter1,arg1,...,argN:filter2,...',
                      action='store', type='string', dest='filters',
                      default=None, help=_help['filters'])
    parser.add_option('-p', '--plots',
                      metavar='plot1,fig_num,arg1,...,argN:plot2,...',
                      action='store', type='string', dest='plots',
                      default=None, help=_help['plots'])
    parser.add_option('-s', '--saves',
                      metavar='save1,arg1,...,argN:save2,...',
                      action='store', type='string', dest='saves',
                      default=None, help=_help['saves'])
    parser.add_option('-n', '--no-show',
                      action='store_false', dest='show',
                      default=True, help=_help['no_show'])
    parser.add_option('-c', '--command-file',
                      metavar='filename',
                      action='store', type='string', dest='command_file',
                      default=None, help=_help['command_file'])
    cmdl_options, args = parser.parse_args()

    expanded_args = []
    for arg in args:
        expanded_args.extend(glob.glob(arg))
    args = expanded_args

    if len(args) == 0:
        parser.print_help()
        return

    filter_cmds, plot_cmds, save_cmds = get_commands(cmdl_options)

    filters = parse_filter_pipeline(filter_cmds)
    plots = parse_filter_pipeline(plot_cmds, get=vars(pl).get, name='plots')
    vv = vars(dataio)
    vv.update(vars(pl))
    saves = parse_filter_pipeline(save_cmds, get=vv.get, name='saves')
    if cmdl_options.show:
        saves = [(pl.show, {})] + saves

    datas = read_all_data(args)

    run_pipeline(filters, plots, saves, datas)

if __name__ == '__main__':
    main()
