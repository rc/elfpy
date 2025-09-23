"""
Run as:

  python3 convert_csv.py folie/data '*.csv'
"""
import os
import sys
import fnmatch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

args = sys.argv[1:]

def get_files(root_dir, pattern):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in sorted(fnmatch.filter(filenames, pattern)):
            yield os.path.join(dirpath, filename)

def load(filename):
    df = pd.read_csv(filename, skiprows=5)
    df = df.rename(columns=lambda x: x.strip())
    dirname, name = os.path.split(filename)
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

    #df = df.set_index('Time [s]')

    return df

print(args)

out = pd.DataFrame(columns=['Name', 'Max. force', 'Displacement', 'Time'])
dgroups = {}
for ii, filename in enumerate(sorted(get_files(args[0], args[1]))):
    print(filename)

    df, dirname, group = load(filename)
    df = convert(df)

    oname = 'new-' + filename
    if not os.path.exists(os.path.dirname(oname)):
        os.makedirs(os.path.dirname(oname))
    with open(oname, 'w') as fd:
        fd.write('\n')
        df.to_csv(fd, float_format='%.6f')

    #print(df.head())

    #df.plot(x='Time [s]')
    #plt.show()

    imax = df['Force [N]'].argmax()
    row = df.loc[imax]
    out.loc[ii] = [filename, row['Force [N]'], row['Displacement [mm]'],
                   row['Time [s]']]

    groups = dgroups.setdefault(dirname, {})
    datas = groups.setdefault(group, {})
    datas[os.path.basename(filename)] = df

print(out)
out.to_csv('results.csv')

fig = plt.figure(1)
fig.clf()
ax = fig.gca()
ax.plot(out['Displacement'], ls='', marker='o', label='Displacement')
ax.plot(out['Max. force'], ls='', marker='o', label='Max. force')
ax.grid()
ax.set_xticks(np.arange(len(out)))
ax.set_xticklabels(map(os.path.basename, out['Name']))
for tick in ax.get_xticklabels():
    tick.set_rotation(90)
ax.legend()
plt.tight_layout()

figname = 'max_forces.png'
fig.savefig(figname, bbox_inches='tight')

for dirname, groups in sorted(dgroups.items()):
    for group, dfs in sorted(groups.items()):
        fig = plt.figure(1)
        fig.clf()
        ax = fig.gca()
        colors = plt.cm.viridis(np.linspace(0, 1, len(dfs) + 1))
        for ii, (name, df) in enumerate(sorted(dfs.items())):
            print(dirname, group, name)

            ax.plot(df['Time [s]'], df['Force [N]'], label=name,
                    color=colors[ii], lw=3)

        ax.legend()
        ax.set_title(dirname)
        plt.tight_layout()

        figname = dirname.replace(os.path.sep, '_') + '_' + group + '.png'
        fig.savefig(figname, bbox_inches='tight')
