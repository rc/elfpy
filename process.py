#!/usr/bin/env python
"""
Analyze results of mechanical measurements by applying filters, plots, and save
commands to data files. The available commands can be listed using the --list
(or -l) option.

The commands can be specified using the command line options, and using a
command file. As the command line commands are applied after the command file
commands, they can be used for overriding.

All filters pass around a data object, that is modified in place.
All plot commands accept a matplotlib figure number and and axes object.
All save commands accept a list of all the data objects.

After parsing of all commands, the following algorithm is used:

1. Read all data files into a list of data objects. Each data object
   corresponds to a single data file.
2. Apply each filter to each data object.
3. Apply each plot command to each data object.
4. Apply each save command to the list of all data objects.

Command Syntax
--------------

Each command consists of a comma-separated list of the command name followed by
the command arguments. The arguments are positional and can have the following
types:

- integer
- float
- string (written without quotation marks, e.g. strain, stress)
- list of floats or integers (e.g. [1; 3; 5] - the items are delimited by
  semicolons so that lists to not interfere with argument parsing)

The command names as well as their possible arguments and argument types can be
listed using the --list (or -l) option. If some arguments are unspecified,
their listed default values are used.

The command file has to have the following structure::

  all filter commands

  -----

  all plot commands

  -----

  all save commands

Lines beginning with '#' are comment lines. Lines beginnings with '-' separate
the command sections.

Examples
--------

Let us assume that the measurements are in text files in the data/ directory.

- Plot filtered and raw stress and strain:

$ python process.py data/*.txt -f 'smooth_strain : smooth_stress' -p 'use_markers, 0 : plot_strain_time, 1, 0, filtered : plot_raw_strain_time, 1, 1, raw : plot_stress_time, 2, 0, filtered : plot_raw_stress_time, 2, 1, raw'

- Detect ultimate stress and strain in the last strain load cycle, plot it on a
  stress-strain curve and save it to a text file:

$ python process.py data/*.txt -f 'smooth_strain : smooth_stress : select_cycle, -1 : get_ultimate_values' -p 'use_markers, 0 : plot_stress_strain, 1, 0, stress-strain : mark_ultimate_values, 1, 1' -s 'save_ultimate_values : save_figure, 1' -n

- Corresponding command file::

    # Beginning of example command file.

    # Filters.
    smooth_strain
    smooth_stress
    select_cycle, -1
    get_ultimate_values

    -----

    # Plot commands.
    use_markers, 0

    plot_stress_strain, 1, 0, stress-strain
    mark_ultimate_values, 1, 1

    -----

    # Save commands.
    save_ultimate_values
    save_figure, 1

    # End of example command file.
"""
from argparse import Action, ArgumentParser, RawDescriptionHelpFormatter
import glob
import copy
import os.path as op
import matplotlib.pyplot as plt

from elfpy.base import output
from elfpy.filters import parse_filter_pipeline, list_commands
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

def read_all_data(filenames, options):
    directory = op.split(__file__)[0]
    areas = read_file_info(op.join(directory, options.cross_sections_filename))
    lengths = read_file_info(op.join(directory, options.init_lengths_filename))

    datas = []
    cols = options.columns
    for i_file, filename in enumerate(filenames):
        data = Data.from_file(filename, sep=options.separator,
                              header_rows=options.header_rows,
                              icycles=cols.get('cycle', None),
                              itime=cols.get('time', 2),
                              idispl=cols.get('displ', 1),
                              iforce=cols.get('force', 0))
        data.set_initial_values(lengths=lengths, areas=areas)
        datas.append(data)

    return datas

def run_pipeline(filters, plots, saves, datas, cmdl_options):
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

        shared_ax = kwargs.pop('ax', None)
        if shared_ax is not None: # True plot command.
            ax = ax if shared_ax else None

        is_legend = False
        for ir, data in enumerate(datas):
            output('plotting: %s ...' % data.name)

            is_legend = is_legend or kwargs.get('label', '')
            _ax = fun(data, ax=ax, **kwargs)

            output('...done')
            if _ax is None:
                if len(datas) > 1:
                    output('non-plot command, skipping other data')
                    break

            else:
                ax = _ax

        output('...done')

        if is_legend:
            plt.legend(loc=cmdl_options.legend_location)

    for ii, save in enumerate(saves):
        fun, kwargs = save

        aux = ', '.join(['%s=%s' % kw for kw in kwargs.iteritems()])
        output('executing: %s(%s) ...' % (fun.__name__, aux))

        fun(datas, **kwargs)

        output('...done')

class PlotParsAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        pars = {}
        for pair in values.split(';'):
            key, val = pair.split('=')
            pars[key] = eval(val)
        setattr(namespace, self.dest, pars)

_help = {
    'filenames' : 'files with measurement data',
    'list' : 'list all available filters, plots and save commands',
    'separator' : 'data separator character [default: %(default)s]',
    'header_rows' : 'number of data header rows [default: %(default)s]',
    'legend_location' : 'matplotlib legend location code'
    ' [default: %(default)s]',
    'columns' : 'indices of time, displacement, force and cycle columns'
    ' in data [default: %(default)s]',
    'filters' : 'filters that should be applied to data files',
    'plots' : 'plots that should be created for data files',
    'saves' : 'commands to save results into files',
    'init_lengths' : 'text file with initial specimen lengths'
    ' (<data file name> <value> per line, # is comment) [default: %(default)s]',
    'cross_sections' : 'text file with initial specimen cross sections'
    ' (<data file name> <value> per line, # is comment) [default: %(default)s]',
    'rc' : 'matplotlib resources',
    'no_show' : 'do not show figures',
    'command_file' : 'file with filter commands followed by plot commands.'
    ' The two groups has to be separated by a line with one or several "-"'
    ' characters. The filter commands are pre-pended to commands passed'
    ' using --filters. The plot commands are pre-pended to commands passed'
    ' using --plots.',
}

def main():
    output.level = 0

    parser = ArgumentParser(description=__doc__.rstrip(),
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('filenames', metavar='filename', nargs='+',
                        help=_help['filenames'])
    parser.add_argument('-l', '--list',
                        action='store_true', dest='list',
                        default=False, help=_help['list'])
    parser.add_argument('--separator',
                        metavar='separator',
                        action='store', dest='separator',
                        default=' ', help=_help['separator'])
    parser.add_argument('--header-rows',
                        metavar='int', type=int,
                        action='store', dest='header_rows',
                        default=2, help=_help['header_rows'])
    parser.add_argument('--legend-location',
                        metavar='int', type=int,
                        action='store', dest='legend_location', default=0,
                        help=_help['legend_location'])
    ac = parser.add_argument('--columns',
                             metavar='key=val,...',
                             action=PlotParsAction, dest='columns',
                             default='time=2;displ=1;force=0;cycle=None',
                             help=_help['columns'])
    parser.add_argument('-f', '--filters',
                        metavar='filter1,arg1,...,argN:filter2,...',
                        action='store', dest='filters',
                        default=None, help=_help['filters'])
    parser.add_argument('-p', '--plots',
                        metavar='plot1,fig_num,arg1,...,argN:plot2,...',
                        action='store', dest='plots',
                        default=None, help=_help['plots'])
    parser.add_argument('-s', '--saves',
                        metavar='save1,arg1,...,argN:save2,...',
                        action='store', dest='saves',
                        default=None, help=_help['saves'])
    parser.add_argument('--init-lengths',
                        metavar='filename', action='store',
                        dest='init_lengths_filename',
                        default='init_lengths.txt',
                        help=_help['init_lengths'])
    parser.add_argument('--cross-sections',
                        metavar='filename', action='store',
                        dest='cross_sections_filename',
                        default='cross_sections.txt',
                        help=_help['cross_sections'])
    parser.add_argument('--rc', metavar='key=val;...',
                        action=PlotParsAction, dest='rc',
                        default={}, help=_help['rc'])
    parser.add_argument('-n', '--no-show',
                        action='store_false', dest='show',
                        default=True, help=_help['no_show'])
    parser.add_argument('-c', '--command-file',
                        metavar='filename',
                        action='store', dest='command_file',
                        default=None, help=_help['command_file'])
    cmdl_options = parser.parse_args()

    if not isinstance(cmdl_options.columns, dict):
        ac(parser, cmdl_options, ac.default)

    expanded_args = []
    for arg in cmdl_options.filenames:
        expanded_args.extend(glob.glob(arg))
    args = expanded_args

    vv = vars(dataio)
    vv.update(vars(pl))

    if cmdl_options.list:
        list_commands()
        list_commands(namespace=vars(pl), name='plots')
        list_commands(namespace=vv, name='saves', arg0_name='datas')
        return

    if len(args) == 0:
        parser.print_help()
        return

    plt.rcParams['font.size'] = 20
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['lines.linewidth'] = 3

    plt.rcParams.update(cmdl_options.rc)

    filter_cmds, plot_cmds, save_cmds = get_commands(cmdl_options)

    filters = parse_filter_pipeline(filter_cmds)
    plots = parse_filter_pipeline(plot_cmds, get=vars(pl).get, name='plots')
    saves = parse_filter_pipeline(save_cmds, get=vv.get, name='saves')
    if cmdl_options.show:
        saves = saves + [(pl.show, {})]

    datas = read_all_data(args, cmdl_options)

    run_pipeline(filters, plots, saves, datas, cmdl_options)

if __name__ == '__main__':
    main()
