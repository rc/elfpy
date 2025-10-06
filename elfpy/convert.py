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

opts = so.Struct(
    pattern = ('*.csv', 'pattern of data file names'),
    output_dir = ([None, ''],
                  'output directory [default: <data_dir>-converted]'),
    data_dir = (None, 'csv data directory'),
)

def load(filename):
    df = pd.read_csv(filename, skiprows=5)
    df = df.rename(columns=lambda x: x.strip())
    dirname, name = op.split(filename)
    return df, dirname, name[:2]

def convert(df_in):
    df = pd.DataFrame()
    df['Time [s]'] = df_in['Time sec']
    df['Cycle'] = df_in['CY-X1']
    df['Force1 [N]'] = df_in['X1L N']
    df['Force2 [N]'] = df_in['X2L N']
    df['Force [N]'] = np.minimum(df['Force1 [N]'], df['Force2 [N]'])
    df['Displacement [mm]'] = df_in['X1Disp mm'] + df_in['X2Disp mm']
    df['Elongation [mm]'] = df['Displacement [mm]'] - df['Displacement [mm]'][0]

    return df

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

    out = pd.DataFrame(columns=['Name', 'Max. force', 'Displacement', 'Time'])
    dgroups = {}
    for ii, filename in enumerate(sorted(
            so.locate_files(options.pattern, root_dir=options.data_dir)
    )):
        filename = op.relpath(filename)
        print(filename)

        df, dirname, group = load(filename)
        df = convert(df)

        oname = inodir(op.basename(filename))
        with open(oname, 'w') as fd:
            fd.write('\n')
            df.to_csv(fd, float_format='%.6f')

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
