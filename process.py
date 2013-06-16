#!/usr/bin/env python
# c: 28.01.2009; r: 03.07.2009

import sys, re, glob
import os.path as op
import numpy as np
import pylab as p
from optparse import OptionParser
import matplotlib.font_manager as fm
from base import Config, Object, pause, output


def read_file_info(filename):
    """
    Reading of the file.

    Parameters
    ----------
    - filename : string

    Returns
    -------
    - info : array
        The values of cross-sectional area and length of the specimens
        according to key.
    """
    fd = open(filename, 'r')
    info = {}
    for line in fd:
        if line and (line[0] not in ['#', '\n']):
            key, val = line.split()
            info[key] = float(val)
    fd.close()
    return info

def split_chunks(strain, time, options, eps_r = 0.01, split = False,
                         append = False):
    """
    Automatic separation of individual cycles in the case of cyclic
    loading. The process is based on the finding of the region between
    individual cycles where the first time derivative of strain is smaller than
    value of eps (close enough to zero).

    Parameters
    ----------
    - strain : array
    - time : array
    - options : list
    - eps_r : float, default = 0.01
    - split : bool, default = False
    - append : bool, default = False

    Returns
    -------
    - aii : array
        The indexes where the first time derivation of strain equal zero
        (local maxima resp. minima).
    - chunks : array
        The sequences of adjacent indexes where the first time
        derivation of strain equal zero.
    - ii : array
        The indexes where the first time derivation of strain equal zero,
        in the case of one cycle ii = [0].
    """
    dstrain = np.diff(strain) / np.diff(time)
    # dstrain = first time derivation of strain
    eps = eps_r * (dstrain.max() - dstrain.min())
    # eps = sensitivity of data approximation
    ii = np.where( np.abs( dstrain ) < eps )[0]
    # region with dstrain < eps, region where the stress-strain curve is partly linear
    print 'ii', ii

    if options.one_cycle:
        if not ii.any():
            ii = [0]


    # finding of indexes where the first derivation is equal zero = local maxima resp. minima
    if split:
        dd = np.ediff1d(ii, to_end = 2)
        ir = np.where(dd > 1)[0]
        chunks = []
        ic0 = 0
        for ic in ir:
            if append:
                chunk = np.r_[ii[ic0:ic+1], ii[ic] + 1]
                ic0 = ic + 1
            else:
                chunk = ii[ic0:ic+1]
                ic0 = ic + 1
            chunks.append(chunk)
        if options.cycles:
            chunks = chunks[:-2]
        if options.cut_first_cycle:
            chunks = chunks[2:]
        aii = np.concatenate(chunks)
        return aii, chunks
    else:
        return ii

def split_curve(stress, strain, eps_r ,  def_ss=0, def_ls = [0,0],
                       split = False, append = False):

    """
    Semi-automatic finding of toe and linear regions (i.e. the region of small
    deformations and large deformations).

    Parameters
    ----------
    - stress : array
    - strain : array
    - eps_r : float
    - def_ss : float, default = 0
    - def_ls : [float,float], default = [0,0]
    - split : bool, default = False
    - append : bool, default = False

    Returns
    -------
    - aii : array
        The indexes where the stress-strain curve is linear.
    - chunks : array
        The sequences of adjacent indexes where the stress-strain.
        curve is linear.
    - ii : array
        The indexes of region with small and large deformations.
    """
    dstress= np.diff(stress, n=1)/ np.diff(strain, n=1)
    ddstress = np.diff(dstress)

    eps = eps_r * (ddstress.max() - ddstress.min())
    p1 = np.where(dstress >= 0)[0]
    p2 = np.ediff1d(p1, to_end = 2 )
    p3 = np.where(p2 > 1)[0]
    if p3[0] == 0 or p3[0] == 1:
        index_value = p1[-1]
    else:
        index_value = p1[p3][0]
    print 'index_value', index_value
    bii= np.where(np.abs(ddstress) < eps)[0]
    index = np.where (bii < index_value) [0]
    ii = bii[index]
    print 'ii', ii

    if split:
        dd = np.ediff1d(ii, to_end = 2)
        ir = np.where(dd > 1)[0]
        chunks = []
        ic0 = 0
        print 'ii', ii
        for ic in ir:
            if append:
                chunk = np.r_[ii[ic0:ic+1], ii[ic] + 1]
                ic0 = ic + 1
            else:
                chunk = ii[ic0:ic+1]
                ic0 = ic + 1
            chunks.append(chunk)
        print 'chunks', chunks
        if def_ss == 0 :
            pass
        else:
            chunks[0] = np.where(strain <= def_ss)[0]
            # defined small strain 0-0.15
            #print 'def_ss == limit, chunks',  chunks, chunks.__len__()
        if def_ls == [0.0,0.0]:
            pass
        else:
            if chunks.__len__() > 1:
                chunks[-1] = np.where( (strain >= def_ls[0])
                & (strain <= def_ls[1]) )[0]
            else:
                chunks.append(np.where( (strain >= def_ls[0])
                & (strain <= def_ls[1]) )[0])
        aii = np.concatenate(chunks)
        return aii, chunks
    else:
        return ii

def get_ultimate_values(strain, stress):
    """
    Automatic determination of ultimate values, i.e. ultimate stress and
    ultimate strain.

    Parameters
    ----------
    - stress : array
    - strain : array

    Returns
    -------
    - ultim_stress : float
    - ultim_strain : float
    """

    dstress= np.diff(stress, n=1)/ np.diff(strain, n=1)

    ii = np.where(dstress < 0)[0]
    if len(ii) == 0:
        print 'warning: stress does not decrease'
        index_ultimate_strength = len(stress)-1
    else:
        index_ultimate_strength = np.where(stress[ii] > 0.1*stress.max())[0]
        if len(index_ultimate_strength ) == 0:
            index_ultimate_strength = ii[0]
            print 'warning: ultimate stress is less then 0.1*max stress'
        else:
            index_ultimate_strength = ii[index_ultimate_strength[0]]
    print 'index_ultimate strength', index_ultimate_strength

    ultim_strain=strain[index_ultimate_strength]
    ultim_stress=stress[index_ultimate_strength]
    print 'ultim_strain, ultim_stress', ultim_strain, ultim_stress

    p.plot([ultim_strain], [ultim_stress], 'g+', ms=10)

    return ultim_strain, ultim_stress


def fit_data(filename, options, lengths, areas, isPlot = 2):
    """
    Determination of Young's modulus of elasticity in the descending part of
    the cycle for individual cycles.

    Parameters
    ----------
    - filename : string
    - options : array
    - lengths : dict
    - areas : dict
    - isPlot : integer, default = 2
        - isPlot=0 no drawing
        - isPlot=1 draw only results
        - isPlot=2 draw all

    Returns
    -------
    - fits : array
        The values of Young's moduli of elasticity
    - strain[i0:i1+1] : array
    - force[i0:i1+1] : array
    """

    fd = open(filename, 'r')
    tdata = fd.readlines()
    fd.close()
    header = '\n'.join(tdata[:2])
    print header

    tdata = tdata[2:]

    print 'length:', len(tdata)

    data = []
    for row in tdata:
        split_row = row.replace( ',', '.' ).split( ';' )

        new = [float( ii ) for ii in split_row]
        data.append(new)


    data = np.array(data, dtype = np.float64)
    print 'shape:', data.shape

    force, strain, time = data[:,0], data[:,1], data[:,2]
    name = op.splitext(filename)[0]
    lenght0 = lengths[name]
    print lenght0


    area = 1.0
    for key, _area in areas.iteritems():
        if re.match(key, name):
            area = _area
            break
    print area

    strain /= lenght0
    force /= area

    if isPlot:
        p.figure(1)
        p.plot(time, force, 'g')
        p.figure(2)
        p.plot(time, strain)

    ms = strain.min()

    ii, chunks = split_chunks(strain, time, options, split = True,
                                      append = True)
    chunks.append(np.array([data.shape[0] - 1]))
    print 'ii:', ii
    print 'chunks:', chunks

    print len(chunks)
    print '# cycles:', len(chunks) / 2

    if isPlot:
        p.plot(time[ii], np.zeros( len( ii ) ), 'ro' )

    ic = 0

    if isPlot == 2:
        p.ion()
        f = p.figure()
        p.ioff()
        ax = f.add_subplot(111)
        ax.plot(strain, force, 'b')

    fits = []
    while ic < len(chunks) - 1:
        i0 = chunks[ic][-1]
        i1 = chunks[ic+1][0]

        x = strain[i0:i1+1]
        y = force[i0:i1+1]
        out = np.polyfit(x, y, 1)

        if isPlot == 2:
            print i0, i1
            h0 = ax.plot(x, y, 'r')
            p.setp(h0, linewidth = 3)
            h1 = ax.plot(x, out[0] * x + out[1], 'g')
            p.setp(h1, linewidth = 3)
            ax.axis([x.min(), x.max(), y.min(), y.max()])
            print out
            f.canvas.draw()
            f.canvas.draw()
            aa = raw_input()
            del ax.lines[-1]
            del ax.lines[-1]

        fits.append(out)
        ic += 2

    fits = np.array(fits)[:,0]

    # Last chunk is up to rupture: treat it separately
    # in fit_stress_strain_lines().
    i0, i1 = chunks[-2][-1], chunks[-1][0]
    if i0 == i1:
        i0 = chunks[-3][-1]
        i1 = chunks[-1][0]
    if isPlot:
        p.figure(4)
        p.plot(fits)
        p.show()
    return fits, strain[i0:i1+1], force[i0:i1+1]

def fit_stress_strain_lines( fig, filename, strain, stress, options, cc,
                                    color_vector, markers, isPlot = False ):
    """
    Determination of Young's modulus of elasticity in the region of small as
    well as large deformations (toe and linear region of stress-strain curve).

    Parameters
    ----------
    - fig : integer
    - filename : string
    - strain : array
    - stress : array
    - options : array
    - cc : integer
    - color_vector : array
    - markers : array
    - isPlot : bool, default = False

    Returns
    -------
    - out0[0] : float
        The value of Young's modulus of elasticity in the region of
        small deformations
    - out1[0] : float
        The value of Young's modulus of elasticity in the region of
        large deformations
    - [h0, h1] : [object, object]
    - h_data : object
    """

    x, y = strain, stress

    ii, chunks = split_curve(stress, strain, eps_r = options.sensitivity, def_ss = options.def_ss,
                                    def_ls = options.def_ls, split = True,
                                    append = True)
    print 'ii, chunks',  ii, chunks

    imin0=chunks[0][0]
    imin1=chunks[0][-1]

    imax0=chunks[-1][0]
    imax1=chunks[-1][-1]+1

    out0 = np.polyfit(x[imin0:imin1+1], y[imin0:imin1+1], 1)
    out1 = np.polyfit(x[imax0:imax1+1], y[imax0:imax1+1], 1)

    p.figure(fig)
    if isPlot:
        p.clf()

    if options.cut_strain == 0.0:
        if options.sampling == 0:
            h_data = p.plot(x, y, color=color_vector[cc,:3], marker=markers[cc],
                             markersize = 3, label=filename)
        else:
            h_data = p.plot(x[::options.sampling], y[::options.sampling], color=color_vector[cc,:3], marker=markers[cc],
                             markersize = 3, label=filename)
    elif options.cut_strain > x[-1]:
        print 'Warning: value of strain out of range'
        h_data = p.plot(x, y, color=color_vector[cc,:3], marker=markers[cc],
                             markersize = 3, label=filename)
    else:
        indexes_cut_strain = np.where( x >= options.cut_strain)
        index_cut_strain = indexes_cut_strain[0][0]
        print 'index_cut_strain', index_cut_strain
        h_data = p.plot(x[:index_cut_strain], y[:index_cut_strain],
                                 color=color_vector[cc,:3], marker=markers[cc],
                                 markersize = 3, label=filename)

    p.plot(x[[imin0,imin1]], y[[imin0,imin1]], 'gs')
    p.plot(x[[imax0,imax1]], y[[imax0,imax1]], 'rs')
    h0 = p.plot(x[:imin1+1], out0[0] * x[:imin1+1] + out0[1],
                 'g', linewidth = 1.5)
    h1 = p.plot(x[imax0:imax1+1], out1[0] * x[imax0:imax1+1] + out1[1],
                 'r', linewidth = 1.5)
    if isPlot:
        p.show()
    return out0[0], out1[0], [h0, h1], h_data

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


usage = """%prog [options] files file_out"""

default_options = {
    'def_ss' : 0,
    'def_ls': [0, 0],
    'legend_fontsize' : 10,
    'file_name_out' : 'results',
    'one_cycle' : False,
    'cycles' : False,
    'cut_first_cycle' : False,
    's' : False,
    'mean_val' : False,
    'cut_strain' : False,
    'ultim_val' : False,
    'sampling' : 0,
    'sensitivity' : 0.01,
    'mean_plot' : False
}

help = {
    'def_ss' :
    'the final value of strain for small deformations [default: %s]'  % default_options['def_ss'],
    'def_ls' :
    'the interval of strain for large deformations [default: %s]'  % (default_options['def_ls'],),
    'legend_fontsize' :
    'a fontsize of the legends [default: %s]'  % default_options['legend_fontsize'],
    'file_name_out' :
    'output file name [default: %s]' % default_options['file_name_out'],
    'one_cycle' :
    'data contain only one cycle',
    'cycles' :
    'data contain only cycles',
    'cut_first_cycle' :
    'the first cycle will be cut from the evaluation',
    's' :
    'save the pictures with evaluated curves and files with mechanical properties',
    'mean_val' :
    'count the mean value of moduli of elasticity of all inserted stress-strain curves',
    'cut_strain' :
    'cut the region of defined strain [default: %s]' % default_options['cut_strain'],
    'ultim_val' :
    'get ultimate values of stress and strain',
    'sampling' :
    'the sampling interval',
    'sensitivity' :
    'the sensitivity of approximation [default: %s]' % default_options['sensitivity'],
    'mean_plot' :
    'the mean plot of stress-strain curve [default: %s]' % default_options['mean_plot'],
    'conf_filename' :
    'use configuration file',
}

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

def parse_def_ls(option, opt, value, parser):
    if value is not None:
        vals = [float(r) for r in value.split(',')]
        assert (len(vals) == 2)

        setattr(parser.values, option.dest, vals)

def main():
    """
    The following command line options are available:

    - def_ss : the final value of strain for small deformations
    - def_ls : the interval of strain for large deformations
    - legend_fontsize : a font size of the legends
    - file_name_out : output file name
    - one_cycle : data contain only one cycle
    - cycles : data contain only cycles
    - cut_first_cycle : the first cycle will be cut from the evaluation
    - s : save the pictures with evaluated curves and files with mechanical
      properties
    - mean_val : count the mean value of moduli of elasticity of all inserted
      stress-strain curves
    - cut_strain : cut the region of defined strain
    - ultim_val : get ultimate values of stress and strain
    - sampling : the sampling interval
    - sensitivity : the sensitivity of approximation
    - conf : use configuration file
    - mean_plot : plot the mean stress-strain curve
    """

    parser = OptionParser(usage=usage, version="%prog ")
    parser.add_option("", "--def-ss", type=float, metavar='float',
                      action="store", dest="def_ss",
                      default=None, help=help['def_ss'])
    parser.add_option("", "--def-ls", type='str', metavar='float,float',
                      action="callback", dest="def_ls",
                      callback=parse_def_ls, help=help['def_ls'])
    parser.add_option("-o", "", metavar='string',
                      action="store", dest="file_name_out",
                      default=None, help=help['file_name_out'])
    parser.add_option("", "--legend-fontsize", type=int, metavar='int',
                      action="store", dest="legend_fontsize",
                      default=None, help=help['legend_fontsize'])
    parser.add_option("", "--one-cycle",
                      action="store_true", dest="one_cycle",
                      default=None, help=help['one_cycle'])
    parser.add_option("", "--cycles",
                      action="store_true", dest="cycles",
                      default=None, help=help['cycles'])
    parser.add_option("", "--cut-first-cycle",
                      action="store_true", dest="cut_first_cycle",
                      default=None, help=help['cut_first_cycle'])
    parser.add_option("-s", "",
                      action="store_true", dest="s",
                      default=None, help=help['s'])
    parser.add_option("", "--mean-val",
                      action="store_true", dest="mean_val",
                      default=None, help=help['mean_val'])
    parser.add_option("", "--cut-strain", type=float, metavar='float',
                      action="store", dest="cut_strain",
                      default=None, help=help['cut_strain'])
    parser.add_option("", "--ultim-val",
                      action="store_true", dest="ultim_val",
                      default=None, help=help['ultim_val'])
    parser.add_option("", "--sampling", type=int, metavar='int',
                      action="store", dest="sampling",
                      default=None, help=help['sampling'])
    parser.add_option("", "--sensitivity", type=float, metavar='float',
                      action="store", dest="sensitivity",
                      default=None, help=help['sensitivity'])
    parser.add_option("-c", "--conf", metavar='filename',
                      action="store", dest="conf_filename",
                      default=None, help=help['conf_filename'])
    parser.add_option("", "--mean-plot",
                      action="store_true", dest="mean_plot",
                      default=None, help=help['mean_plot'])
    cmdl_options, args = parser.parse_args()

    expanded_args = []
    for arg in args:
        expanded_args.extend(glob.glob(arg))
    args = expanded_args

    file_number = len(args)
    if file_number == 0:
        parser.print_help()
        return

    can_override = set()
    for key, default in default_options.iteritems():
        val = getattr(cmdl_options, key)
        if val is None:
            setattr(cmdl_options, key, default)
        else:
            can_override.add(key)

    if cmdl_options.conf_filename is not None:
        config = Config.from_file(cmdl_options.conf_filename,
                                  defaults=default_options)
    else:
        conf = {'options' : {'default' : default_options},
                'options_default' : default_options}
        config = Config.from_conf(conf)

    config.override(cmdl_options, can_override)
    options = Object(name='options', **(config.options['default']))

    filename_out = options.file_name_out

    markers, linestyles, color_vector = define_plot_attributes()

    fp = fm.FontProperties()
    fp.set_size(options.legend_fontsize)

    directory = op.split(__file__)[0]

    areas = read_file_info(op.join(directory,'cross_sections.txt'))
    lengths = read_file_info(op.join(directory,'init_lengths.txt'))

    isFinal = False
    isLast = False
    isPlot = 0


    if not options.one_cycle:
        p.figure(1)
        p.clf()
    if not options.cycles:
        p.figure(5)
        p.clf()


    all_fits = {}
    list_fits = []
    ks = []
    h_datas = []
    ult_strain=[]
    ult_stress=[]
    strain_all = []
    stress_all = []
    strain_size = []
    k_fig = 5
    for i_file in range( 0, file_number ):
        filename = args[i_file]
        print 'file:', filename

        if filename in config.options:
            specific_options = Object(name='options', **(config.options[filename]))
        else:
            specific_options = options

        fits, strain, stress = fit_data(filename, specific_options, lengths, areas,
                                               isPlot = isPlot )
        if specific_options.mean_plot:
            strain_size.append(strain.shape[0])
            strain_all.append(strain)
            stress_all.append(stress)
            print 'strain_all', strain_all, strain_size

        if not specific_options.cycles:
            k0, k1, h_fit, h_data = fit_stress_strain_lines(k_fig, filename, strain,
                                                                   stress, specific_options, i_file,
                                                                   color_vector, markers,
                                                                   isPlot = isPlot )
            ks.append( (k0, k1) )
            h_datas.extend(h_data)

            if options.ultim_val:
                ultim_strain, ultim_stress = get_ultimate_values(strain, stress)
                ult_strain.append(ultim_strain)
                ult_stress.append(ultim_stress)

        if i_file == 0:
            avg_fits = np.zeros_like(fits)
        avg_fits += fits
        all_fits[filename] = fits
        list_fits.append(fits)
    avg_fits /= float(file_number)

    if specific_options.mean_plot:
        index_length = np.min(strain_size)
        print index_length
        strain_mean = np.mean(strain_all, 0)
        stress_mean = np.mean(stress_all, 0)
        p.figure(61)
        p.plot(strain_mean, stress_mean, marker = 'p', color = 'k')

    if options.ultim_val:
        ult_strain = np.array(ult_strain, dtype = np.float64)
        ult_strain_average = np.sum(ult_strain, 0) / ult_strain.shape[0]
        ult_strain_dev = np.std(ult_strain, 0)
        print 'ultimate_strain', ult_strain_average, '\pm', ult_strain_dev

        ult_stress = np.array(ult_stress, dtype = np.float64)
        ult_stress_average = np.sum(ult_stress, 0) / ult_stress.shape[0]
        ult_stress_dev = np.std(ult_stress, 0)
        print 'ultimate_stress', ult_stress_average, '\pm', ult_stress_dev



    list_fits.insert(0, avg_fits)
    a_fits = np.array(list_fits, dtype = np.float64).T
    print a_fits.shape
    if options.s:
        print 'output file:', filename_out

        fd = open(op.splitext( filename_out)[0] + '_moduli_cycles.txt',
                        'w' )
        for row in a_fits:
            fd.write(' '.join([('%.3e' % ii) for ii in row]))
            fd.write('\n')
        fd.close()

    if not options.cycles:
        if options.mean_val:
            ks = np.array(ks, dtype = np.float64)
            ks_average = np.sum(ks, 0) / ks.shape[0]
            ks_dev = np.std(ks, 0)
            print ks
            print ks_average
            print ks_dev
            if options.s:
                np.savetxt(op.splitext( filename_out )[0] + '_moduli.txt', ks)

            fig = p.figure(k_fig)
            p.xlabel( 'strain' )
            p.ylabel( 'stress [MPa]' )
            texts = [r'$E_0 = %.2e\ \pm\ %.2e$' % (ks_average[0], ks_dev[0]),
                    r'$E_1 = %.2e\ \pm\ %.2e$' % (ks_average[1], ks_dev[1])]
            leg = make_legend_text(args, ks)
            fig.legend(h_datas, leg, loc = (0.6, 0.35), prop=fp)
            p.legend(h_fit, texts, loc = 'lower right', prop=fp)

            if options.s:
                fig_name = op.splitext(filename_out )[0] + '_stress_strain.pdf'
                p.savefig(fig_name, dpi = 300)
        else:
            ks = np.array(ks, dtype = np.float64)
            print ks
            if options.s:
                np.savetxt(op.splitext(filename_out)[0] + '_moduli.txt', ks)
            p.figure(k_fig)

            leg = make_legend_text(args, ks)
            p.xlabel( 'strain' )
            p.ylabel( 'stress [MPa]' )
            p.legend(h_datas, leg, loc = 'upper right', prop=fp)
            if options.s:
                fig_name = op.splitext(filename_out)[0] + '_stress_strain.pdf'
                p.savefig(fig_name, dpi = 300)

    if isLast:
        to = avg_fits.shape[0]
    else:
        to = avg_fits.shape[0] - 1

    if not options.one_cycle:
        p.figure( 1 )
        cycle = np.arange( 1, to + 1 )
        if not isFinal:
            leg = []
            ii=0
            for name, fits in all_fits.iteritems():
                p.plot(cycle, fits[:to], color=color_vector[ii,:3], marker=markers[ii], markersize = 3, label=op.splitext(name)[0])
                ii=ii+1
            if options.mean_val:
                p.plot(cycle, avg_fits[:to], 'ro')
            else:
                pass
        if options.mean_val:
            p.errorbar(cycle, avg_fits[:to],
                    yerr = np.std(a_fits[:to,1:], 1),
                    marker = 'o', mfc = 'red', label='mean_value $\pm$ SD')
        else:
            pass

        p.legend(loc = 'upper right' , prop=fp)
        p.xlim(0, a_fits.shape[0] + 1)
        p.xlabel('cycle number')
        p.ylabel('modulus of elasticity [MPa]')

    if options.s:
        if isFinal:
            fig_name = op.splitext(filename_out)[0] + '.pdf'
        else:
            fig_name = op.splitext(filename_out)[0] + '_all.pdf'
        p.savefig(fig_name, dpi = 300)
    p.show()

if __name__ == '__main__':
    main()
