#!/usr/bin/env python
r"""
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
- string, written with or without quotation marks, e.g. 'strain', stress
          special characters such as '%' must be quoted
- list of floats or integers, for example [1, 3, 5]

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

- Plot filtered and raw stress and strain::

    elfpy-process data/PER_*.txt --separator='\s+' --columns='time=2,displ=1,force=0,cycle=None' -f "smooth_strain : smooth_stress" -p "use_markers, 0 : plot_strain_time, 1, 0, filtered : plot_raw_strain_time, 1, 1, raw : plot_stress_time, 2, 0, filtered : plot_raw_stress_time, 2, 1, raw"

- Detect ultimate stress and strain in the last strain load cycle, plot it on a
  stress-strain curve and save it to a text file::

    elfpy-process data/PER_*.txt --separator='\s+' --columns='time=2,displ=1,force=0,cycle=None' -f "smooth_strain : smooth_stress : select_cycle, -1 : get_ultimate_values" -p "use_markers, 0 : plot_stress_strain, 1, 0, 'stress-strain' : mark_ultimate_values, 1, 1" -s "save_ultimate_values : save_figure, 1" -n

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

    plot_stress_strain, 1, 0, 'stress-strain'
    mark_ultimate_values, 1, 1

    -----

    # Save commands.
    save_ultimate_values
    save_figure, 1

    # End of example command file.
"""
import glob
import copy
import matplotlib.pyplot as plt

import soops as so

from elfpy.base import output
from elfpy.filters import parse_filter_pipeline, list_commands
from elfpy.dataio import read_file_info, Data
import elfpy.dataio as dataio
import elfpy.plots as pl
from elfpy.devices import devices_table

opts = so.Struct(
    list = (False, 'list all available filters, plots and save commands'),
    machine = (tuple(devices_table.keys()),
               'measurement machine name. Determines data columns.'),
    columns = (
        [None, ''],
        """indices of time, displacement, force and cycle columns in data.
        If given, overrides default machine settings.""",
        dict(metavar='KEY=VAL,...')
    ),
    separator = (
        [None, ','],
        r"""CSV data separator character(s), use '\s+' for whitespace. If given,
        overrides default machine settings. """
    ),
    header_rows = (1, 'number of data header rows'),
    filters = ([None, ''], 'filters that should be applied to data files',
               dict(metavar='FILTER1,ARG1,...,ARGN:FILTER2,...')),
    plots = ([None, ''], 'plots that should be created for data files',
             dict(metavar='PLOT1,FIG_NUM,ARG1,...,ARGN:PLOT2,...')),
    saves = ([None, ''], 'commands to save results into files',
             dict(metavar='SAVE1,ARG1,...,ARGN:SAVE2,...')),
    init_lengths = (
        'init_lengths.txt',
        """text file with initial specimen lengths
        (<data file name> <value> per line, # is comment)""",
        dict(metavar='FILENAME')
    ),
    cross_sections = (
        'cross_sections.txt',
        """text file with initial specimen cross sections
        (<data file name> <value> per line, # is comment)""",
        dict(metavar='FILENAME')
    ),
    legend_location = (0, 'matplotlib legend location code'),
    plot_rc_params = ('', 'matplotlib resources', dict(metavar='KEY=VAL,...')),
    show = (True, 'do not show figures'),
    shell = (False, 'run shell'),
    debug = (False, 'debug on error'),
    command_file = (
        [None, ''],
        """file with filter commands followed by plot commands. The two groups
        have to be separated by a line with one or several '-' characters. The
        filter commands are pre-pended to commands passed using --filters. The
        plot commands are pre-pended to commands passed using --plots.""",
        dict(metavar='FILENAME')
    ),
    filenames = (None, 'files with measurement data', dict(nargs='*')),
)

def get_commands(options):
    """
    Get filter and plot commands from options.
    """
    filter_cmds = []
    plot_cmds = []
    save_cmds = []
    if options.command_file:
        with open(options.command_file, 'r') as fd:
            cmds = fd.readlines()

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

    if options.filters:
        filter_cmds.extend(options.filters.split(':'))

    if options.plots:
        plot_cmds.extend(options.plots.split(':'))

    if options.saves:
        save_cmds.extend(options.saves.split(':'))

    return filter_cmds, plot_cmds, save_cmds

def read_all_data(filenames, options):
    areas = read_file_info(options.cross_sections)
    lengths = read_file_info(options.init_lengths)

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

def run_pipeline(filters, plots, saves, datas, options):
    """
    Apply filters and then plots to datas.
    """
    for ii, flt in enumerate(filters):
        fun, kwargs = flt

        aux = ', '.join(['%s=%s' % kw for kw in kwargs.items()])
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

        aux = ', '.join(['%s=%s' % kw for kw in kwargs.items()])
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
            plt.legend(loc=options.legend_location)

    for ii, save in enumerate(saves):
        fun, kwargs = save

        aux = ', '.join(['%s=%s' % kw for kw in kwargs.items()])
        output('executing: %s(%s) ...' % (fun.__name__, aux))

        fun(datas, **kwargs)

        output('...done')

def parse_args(args=None):
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    so.build_arg_parser(parser, opts, aliases=dict(
        list='-l',
        filters='-f',
        plots='-p',
        saves='-s',
        show='-n',
        command_file='-c',
    ))
    options = parser.parse_args(args=args)
    options.plot_rc_params = so.parse_as_dict(options.plot_rc_params)
    if options.columns is not None:
        options.columns = so.parse_as_dict(options.columns)

    else:
        options.columns = devices_table[options.machine].converted_columns

    if options.separator is None:
        options.separator = devices_table[options.machine].separator

    return parser, options

def main():
    output.level = 0

    parser, options = parse_args()
    if options.debug:
        from soops import debug; debug()

    expanded_args = []
    for arg in options.filenames:
        expanded_args.extend(glob.glob(arg))
    args = expanded_args

    vv = vars(dataio)
    vv.update(vars(pl))

    if options.list:
        list_commands()
        list_commands(namespace=vars(pl), name='plots')
        list_commands(namespace=vv, name='saves', arg0_name='datas')
        return

    if len(args) == 0:
        parser.print_help()
        return

    plt.rcParams['font.size'] = 12
    plt.rcParams['lines.linewidth'] = 1
    plt.rcParams.update(options.plot_rc_params)

    filter_cmds, plot_cmds, save_cmds = get_commands(options)

    filters = parse_filter_pipeline(filter_cmds)
    plots = parse_filter_pipeline(plot_cmds, get=vars(pl).get, name='plots')
    saves = parse_filter_pipeline(save_cmds, get=vv.get, name='saves')
    if options.show:
        saves = saves + [(pl.show, {})]

    datas = read_all_data(args, options)

    run_pipeline(filters, plots, saves, datas, options)

    if options.shell:
        from soops import shell; shell()

if __name__ == '__main__':
    main()
