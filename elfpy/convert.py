"""
Convert csv files.

Examples
--------

Convert all csv files in a given directory 'data', put results to
data-converted::

    elfpy-convert data
"""
import os.path as op
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import soops as so

from .devices import TestingMachine

opts = so.Struct(
    machine = ('mtl32_2020', 'measurement machine name',
                    dict(metavar='NAME')),
    pattern = ('*.csv', 'pattern of data file names'),
    init_lengths = ([None, ''], 'if given, add initial lengths to this file',
                    dict(metavar='FILENAME')),
    output_dir = ([None, ''],
                  'output directory [default: <data_dir>-converted]'),
    data_dir = (None, 'csv data directory'),
)

def parse_args(args=None):
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    so.build_arg_parser(parser, opts)
    options = parser.parse_args(args=args)

    if options.output_dir is None:
        options.output_dir = op.dirname(options.data_dir) + '-converted'

    return options

def main():
    options = parse_args()

    so.ensure_path(options.output_dir + op.sep)
    inodir = partial(op.join, options.output_dir)

    machine = TestingMachine.any_by_name(options.machine)

    out = pd.DataFrame(columns=['Name', 'Max. force', 'Displacement', 'Time'])
    dgroups = {}
    for ii, filename in enumerate(sorted(
            so.locate_files(options.pattern, root_dir=options.data_dir)
    )):
        filename = op.relpath(filename)
        print(filename)

        mdf = machine.read_data(filename)
        dirname, name = op.split(filename)
        group = name[:2]

        basename = op.basename(filename)
        if options.init_lengths:
            key = op.splitext(basename)[0]
            length = machine.get_init_length(mdf)
            print(f'{key}: initial length: {length}')
            with open(options.init_lengths, 'a') as fd:
                fd.write(f'{key} {length:.2f}\n')

        df = machine.convert(mdf)

        df.to_csv(inodir(basename), float_format='%.6f')

        imax = df['Force [N]'].argmax()
        row = df.loc[imax]
        out.loc[ii] = [filename, row['Force [N]'], row['Displacement [mm]'],
                       row['Time [s]']]

        groups = dgroups.setdefault(dirname, {})
        datas = groups.setdefault(group, {})
        datas[op.basename(filename)] = df

    print(out)
    out.to_csv(inodir('max-forces.csv'))

    fig, ax = plt.subplots()
    ax.plot(out['Displacement'], ls='', marker='o', label='Displacement')
    ax.plot(out['Max. force'], ls='', marker='o', label='Max. force')
    ax.grid()
    ax.set_xticks(np.arange(len(out)))
    ax.set_xticklabels(map(op.basename, out['Name']))
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    ax.legend()
    plt.tight_layout()

    figname = 'max-forces.png'
    fig.savefig(inodir(figname), bbox_inches='tight')

    fig, ax = plt.subplots()
    for dirname, groups in sorted(dgroups.items()):
        for group, dfs in sorted(groups.items()):
            ax.cla()
            colors = plt.cm.viridis(np.linspace(0, 1, len(dfs) + 1))
            for ii, (name, df) in enumerate(sorted(dfs.items())):
                print(dirname, group, name)

                ax.plot(df['Time [s]'], df['Force [N]'], label=name,
                        color=colors[ii], lw=3)

            ax.legend()
            ax.set_title(dirname)
            plt.tight_layout()

            figname = dirname.replace(op.sep, '_') + '_' + group + '.png'
            fig.savefig(inodir(figname), bbox_inches='tight')

if __name__ == '__main__':
    main()
