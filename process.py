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
import elfpy.plots as pl

usage = '%prog [options] filenames\n' + __doc__.rstrip()

_help = {
    'filters' : 'filters that should be aplied to data files',
    'plots' : 'plots that should be created for data files',
}

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

def run_pipeline(filters, plots, datas):
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

        for ir, data in enumerate(datas):
            output('plotting: %s ...' % data.name)

            ax = fun(data, ax=ax, **kwargs)

            output('...done')
        output('...done')

    plt.show()

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
    cmdl_options, args = parser.parse_args()

    expanded_args = []
    for arg in args:
        expanded_args.extend(glob.glob(arg))
    args = expanded_args

    if len(args) == 0:
        parser.print_help()
        return

    filters = parse_filter_pipeline(cmdl_options.filters)
    plots = parse_filter_pipeline(cmdl_options.plots,
                                  get=vars(pl).get, name='plots')
    datas = read_all_data(args)

    run_pipeline(filters, plots, datas)

if __name__ == '__main__':
    main()
