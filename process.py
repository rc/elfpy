#!/usr/bin/env python
# c: 28.01.2009; r: 19.05.2009
 

import sys, re
import os.path as op
import numpy as nm
import pylab as p
from optparse import OptionParser

#~ lengths = { # mm
    #~ 'carp_001' : 22,
    #~ 'carp_002' : 22,
    #~ 'carp_003' : 22,
    #~ 'carp_004' : 22,
    #~ 'carp_005' : 22,
    #~ 'carp_006' : 22,
    #~ 'carp_007' : 22,
    #~ 'carp_008' : 22,
    #~ 'carp_009' : 22,
    #~ 'carp_010' : 22,
    #~ 'carp_011' : 22,
    #~ 'carp_012' : 22,
    #~ 'carp_013' : 22,
    #~ 'CSub_01'  : 18.0,
    #~ 'CSub_02'  : 23.0,
    #~ 'CSub_03'  : 16.0,
    #~ 'CSub_04'  : 13.0,
    #~ 'CSub_05'  : 21.0,
    #~ 'CTh_01'   : 28.0,
    #~ 'CTh_02'   : 32.0,
    #~ 'DSub_01'  : 25.0,
    #~ 'DSub_02'  : 18.0,
    #~ 'DTh_01'   : 25.0,
    #~ 'NXSub_01' : 22.0,
    #~ 'NXSub_02' : 19.0,
    #~ 'NXSub_03' : 22.0,
    #~ 'equine_hoof_BU_I_K1' : 6.00,
    #~ 'equine_hoof_BU_II_K1' : 6.00,
    #~ 'equine_hoof_BU_I_K2' : 5.90,
    #~ 'equine_hoof_BU_III_K1' : 6.20,
    #~ 'equine_hoof_SS_I_K1' : 5.20,
    #~ 'equine_hoof_SS_II_K1' : 5.70,
    #~ 'equine_hoof_SS_III_K1' : 5.80,
    #~ 'equine_hoof_BS_I_K1' : 5.60,
    #~ 'equine_hoof_BS_II_K1' : 3.90,
    #~ 'equine_hoof_BS_III_K1' : 4.20,
    #~ 'EP124_ED600_PDGEBA_r125_bubbles' : 22.00,
    #~ 'VSM_01' : 10.00,
    #~ 'VSM_02' : 9.00,
    #~ 'VSM_03' : 7.00,
    #~ 'EP121_ED600_PDGEBA_r100' : 18.00,
    #~ 'EP121_ED600_PDGEBA_r100_bubbles' : 12.00,
    #~ 'EP122_ED600_PDGEBA_r106' : 18.00,
    #~ 'EP123_ED600_PDGEBA_r112' :  18.00,
    #~ 'EP123_ED600_PDGEBA_r112_bubbles' : 18.00,
    #~ 'EP124_ED600_PDGEBA_r125_bubbles_H2O' : 12.00,
    #~ 'EP125_ED600_PDGEBA_r150_bubbles' : 18.00,
    #~ 'EP126_ED600_PDGEBA_r200' : 12.00,
    #~ 'EP126_ED600_PDGEBA_r200_bubbles' : 18.00,
#~ }

#~ areas = { # mm^2
    #~ 'carp_001' : 7.600,
    #~ 'carp_002' : 10.620,
    #~ 'carp_003' : 8.640,
    #~ 'carp_004' : 9.000,
    #~ 'carp_005' : 10.740,
    #~ 'carp_006' : 8.700,
    #~ 'carp_007' : 13.260,
    #~ 'carp_008' : 12.360,
    #~ 'carp_009' : 12.960,
    #~ 'carp_010' : 24.240,
    #~ 'carp_011' : 41.020,
    #~ 'carp_012' : 9.540,
    #~ 'carp_013' : 7.920,
    #~ 'CSub_[0-9]+'  : 11.7,
    #~ 'NXSub_[0-9]+' : 17.3,
    #~ 'equine_hoof_BU_I_K1' : 40.17,
    #~ 'equine_hoof_BU_II_K1' : 45.60,
    #~ 'equine_hoof_BU_I_K2' : 45.25,
    #~ 'equine_hoof_BU_III_K1' : 44.98,
    #~ 'equine_hoof_SS_I_K1' : 48.35,
    #~ 'equine_hoof_SS_II_K1' : 46.25,
    #~ 'equine_hoof_SS_III_K1' : 51.34,
    #~ 'equine_hoof_BS_I_K1' : 54.08,
    #~ 'equine_hoof_BS_II_K1' : 48.79,
    #~ 'equine_hoof_BS_III_K1' : 39.96,
    #~ 'EP124_ED600_PDGEBA_r125_bubbles' : 9.17,
    #~ 'VSM_01' : 10.0,
    #~ 'VSM_02' : 10.0,
    #~ 'VSM_03' : 10.0,
    #~ 'EP121_ED600_PDGEBA_r100' : 9.13,
    #~ 'EP121_ED600_PDGEBA_r100_bubbles' : 9.85,
    #~ 'EP122_ED600_PDGEBA_r106' : 8.99,
    #~ 'EP123_ED600_PDGEBA_r112' :  9.35,
    #~ 'EP123_ED600_PDGEBA_r112_bubbles' : 9.93,
    #~ 'EP124_ED600_PDGEBA_r125_bubbles_H2O' : 11.02,
    #~ 'EP125_ED600_PDGEBA_r150_bubbles' :  10.69,
    #~ 'EP126_ED600_PDGEBA_r200' : 9.42,
    #~ 'EP126_ED600_PDGEBA_r200_bubbles' : 9.63,
#~ }

def read_file_info(filename):
    fd = open(filename, 'r')
    info = {}
    for line in fd:
        if line and (line[0] not in ['#', '\n']):
            key, val = line.split()
            info[key] = float(val)
    fd.close()
    return info

def splitChunks( strain, time, options, epsR = 0.01, split = False, append = False):
    dstrain = nm.diff( strain ) / nm.diff( time )
    eps = epsR * (dstrain.max() - dstrain.min())
    ii = nm.where( nm.abs( dstrain ) < eps )[0]
    print 'ii', ii
    
    if options.one_cycle:
        if not ii.any():
            ii = [0]
        
    #import pdb; pdb.set_trace()
     
    if split:
        dd = nm.ediff1d( ii, to_end = 2 )
        ir = nm.where( dd > 1 )[0]
        chunks = []
        ic0 = 0
#        print ii
        for ic in ir:
            if append:
                chunk = nm.r_[ii[ic0:ic+1], ii[ic] + 1]
                ic0 = ic + 1
            else:
                chunk = ii[ic0:ic+1]
                ic0 = ic + 1
            chunks.append( chunk )
#        print chunks
        if options.cycles:
            chunks = chunks[:-2]
        if options.cut_first_cycle:
            chunks = chunks[2:]
        aii = nm.concatenate( chunks )
        return aii, chunks
    else:
        return ii

def splitCurve( stress, strain, epsR = 0.02,  def_ss=0, def_ls = [0,0], split = False, append = False ):
    
    dstress= nm.diff(stress, n=1)/ nm.diff(strain, n=1)
    ddstress = nm.diff(dstress)
    #import pdb; pdb.set_trace()
    
    eps = epsR * (ddstress.max() - ddstress.min())
    p1 = nm.where(dstress >= 0)[0]
    p2 = nm.ediff1d(p1, to_end = 2 )
    p3 = nm.where(p2 > 1)[0]
    if p3[0] == 0:
        index_value = p1[-1]
    else:
        index_value = p1[p3][0]
    #bii= nm.where( (nm.abs(ddstress) < eps)  & (dstress[:-1] >= 0) )[0]
    bii= nm.where(nm.abs(ddstress) < eps)[0]
    index = nm.where (bii < index_value) [0]
    ii = bii[index]
    print 'ii', ii
    #import pdb; pdb.set_trace()
    
    if split:
        dd = nm.ediff1d( ii, to_end = 2 )
        ir = nm.where( dd > 1 )[0]
        chunks = []
        ic0 = 0
        print 'ii', ii
        for ic in ir:
            if append:
                chunk = nm.r_[ii[ic0:ic+1], ii[ic] + 1]
                ic0 = ic + 1
            else:
                chunk = ii[ic0:ic+1]
                ic0 = ic + 1
            chunks.append( chunk )
        #print chunks
        if def_ss == 0 :
            pass
            #print 'def_ss == 0, chunks',  chunks
        else:
            chunks[0] = nm.where( strain <= def_ss )[0]  # defined small strain 0-0.15
            #print 'def_ss == limit, chunks',  chunks, chunks.__len__()
        if def_ls == [0.0,0.0]:
            pass
        else:
            if chunks.__len__() > 1:
                chunks[-1] = nm.where( (strain >= def_ls[0]) & (strain <= def_ls[1]) )[0]
            else:
                chunks.append(nm.where( (strain >= def_ls[0]) & (strain <= def_ls[1]) )[0])
        #print 'chunks', chunks
        aii = nm.concatenate( chunks )
        return aii, chunks
    else:
        return ii

def fitdata( fileName, options, lengths, areas, isPlot = 2):
    """isPlot=0 nekresli nic, isPlot=1 vysledek, isPlot=2 vsechny"""

    fd = open( fileName, 'r' )
    tdata = fd.readlines()
    fd.close()
    header = '\n'.join( tdata[:2] )
    print header

    tdata = tdata[2:]

    print 'length:', len( tdata )

    data = []
    for row in tdata:
    #    print row
        splitRow = row.replace( ',', '.' ).split( ';' )
    #    print splitRow

        new = [float( ii ) for ii in splitRow]
        data.append( new )

    
    
    data = nm.array( data, dtype = nm.float64 )
    print 'shape:', data.shape

    force, strain, time = data[:,0], data[:,1], data[:,2]
    name = op.splitext( fileName )[0]
    lenght0 = lengths[name]
    print lenght0

#stress, strain, time = data[:,0]/cross_sections[name], data[:,1]/lengths[name], data[:,2]


    area = 1.0
    for key, _area in areas.iteritems():
        if re.match( key, name ):
            area = _area
            break
    print area
    
##     print force
##     print strain
    strain /= lenght0
    force /= area
##     print force
##     print strain
    if isPlot:
        p.figure( 1 )
        p.plot( time, force, 'g' )
        p.figure( 2 )
        p.plot( time, strain )
    
    ms = strain.min()
        
    ii, chunks = splitChunks( strain, time, options, split = True, append = True)
    chunks.append( nm.array( [data.shape[0] - 1] ) )
    print 'ii:', ii
    print 'chunks:', chunks
    
    print len(chunks)
    print '# cycles:', len( chunks ) / 2

    if isPlot:
        p.plot( time[ii], nm.zeros( len( ii ) ), 'ro' )

    ic = 0

    if isPlot == 2:
        p.ion()
        f = p.figure()
        p.ioff()
        ax = f.add_subplot( 111 )
        ax.plot( strain, force, 'b' )

    fits = []
    while ic < len( chunks ) - 1:
        i0 = chunks[ic][-1]
        i1 = chunks[ic+1][0]

        x = strain[i0:i1+1]
        y = force[i0:i1+1]
        out = nm.polyfit( x, y, 1 )

        if isPlot == 2:
            print i0, i1
            h0 = ax.plot( x, y, 'r' )
            p.setp( h0, linewidth = 3 )
            h1 = ax.plot( x, out[0] * x + out[1], 'g' )
            p.setp( h1, linewidth = 3 )
            ax.axis( [x.min(), x.max(), y.min(), y.max()] )
            print out
            f.canvas.draw()
            f.canvas.draw()
            aa = raw_input()
            del ax.lines[-1]
            del ax.lines[-1]

        fits.append( out )
        ic += 2

    fits = nm.array( fits )[:,0]

    ##
    # Last chunk is up to rupture: treat it separately
    # in fitStressStrainLines().
    i0, i1 = chunks[-2][-1], chunks[-1][0]
    if i0==i1:
        i0=chunks[-3][-1]
        i1=chunks[-1][0]
    if isPlot:
        p.figure( 4 )
        p.plot( fits )
        p.show()
    #import pdb; pdb.set_trace()
    return fits, strain[i0:i1+1], force[i0:i1+1]

def fitStressStrainLines( fig, fileName, strain, stress, options,
                          isPlot = False ):
    x, y = strain, stress
    #import pdb; pdb.set_trace()    
    
    ii, chunks = splitCurve( stress, strain, def_ss=options.def_ss, def_ls=options.def_ls, split = True, append = True)
    #ii, chunks = splitCurve( stress, strain, epsR = 0.01,  def_ss = 0 , def_ls = [0,0], split = True, append = True)
    print 'ii, chunks',  ii, chunks
        
    imin0=chunks[0][0]
    imin1=chunks[0][-1]
        
    imax0=chunks[-1][0]
    imax1=chunks[-1][-1]+1
    
    out0 = nm.polyfit( x[imin0:imin1+1], y[imin0:imin1+1], 1 )
    out1 = nm.polyfit( x[imax0:imax1+1], y[imax0:imax1+1], 1 )
    #import pdb; pdb.set_trace()
    
    p.figure( fig )
    if isPlot:
        p.clf()
    p.plot( x, y, 'b-o', markersize = 3 )
    p.plot( x[[imin0,imin1]], y[[imin0,imin1]], 'gs' )
    p.plot( x[[imax0,imax1]], y[[imax0,imax1]], 'rs' )
    h0 = p.plot( x[:imin1+1], out0[0] * x[:imin1+1] + out0[1],
                 'g', linewidth = 1.5 )
    h1 = p.plot( x[imax0:imax1+1], out1[0] * x[imax0:imax1+1] + out1[1],
                 'r', linewidth = 1.5 )
    
    axLegend = p.legend( [h0, h1], ['0', '1'],
                         loc = 'upper right')
    if isPlot:
        p.show()

    return out0[0], out1[0], axLegend


usage = """%prog [options] files file_out"""

help = {
    #~ 'filename' :
    #~ 'basename of output file(s) [default: %default]',
    #~ 'output_format' :
    #~ 'output file format (supported by the matplotlib backend used) '\
    #~ '[default: %default]',
    'def_ss' :
    'the final value of strain for small deformations [default: %default]',
    'def_ls' :
    'the interval of strain for large deformations [default: %default]',
    'file_name_out' :
    'output file name [default: %default]',
    'one_cycle' :
    'data contain only one cycle',
    'cycles' :
    'data contain only cycles',
    'cut_first_cycle' :
    'the first cycle will be cut from the evaluation',
    's' :
    'save the pictures with evaluated curves and files with mechanical properties',
 }

def main():
        
    parser = OptionParser(usage=usage, version="%prog ")
    parser.add_option("", "--def-ss", type=float, metavar='float',
                      action="store", dest="def_ss",
                      default='0', help=help['def_ss'])
    parser.add_option("", "--def-ls", metavar='float,float',
                      action="store", dest="def_ls",
                      default='0,0', help=help['def_ls'])
    parser.add_option("-o", "", metavar='string',
                      action="store", dest="file_name_out",
                      default='results', help=help['file_name_out'])
    parser.add_option("", "--one-cycle", 
                      action="store_true", dest="one_cycle",
                      default=False, help=help['one_cycle'])
    parser.add_option("", "--cycles", 
                      action="store_true", dest="cycles",
                      default=False, help=help['cycles'])
    parser.add_option("", "--cut-first-cycle", 
                      action="store_true", dest="cut_first_cycle",
                      default=False, help=help['cut_first_cycle'])
    parser.add_option("-s", "", 
                      action="store_true", dest="s",
                      default=False, help=help['s'])
    options, args = parser.parse_args()
    
    options.def_ls = [float(r) for r in  options.def_ls.split(',')]
    print options
    #raw_input()
    
    
    directory = op.split(__file__)[0]
        
    areas = read_file_info(op.join(directory,'cross_sections.txt'))
    lengths = read_file_info(op.join(directory,'init_lengths.txt'))
    #~ print 'cross-sections (mm^2), initial lengths (mm):' 
    #~ for key, val in areas.items():
        #~ print '%-10s : % 8.3f, % 8.3f' % (key, val, lengths[key])

    
    isFinal = False
    isLast = False
    isPlot = 0
        
    fileNumber = len(args)
    if fileNumber == 0:
        parser.print_help()
        return
        
    fileNameOut = options.file_name_out
    
    if not options.one_cycle:
        p.figure(1)
        p.clf()
    if not options.cycles:
        p.figure(5)
        p.clf()

    
    #~ p.figure(1)
    #~ p.clf()
    #~ p.figure(5)
    #~ p.clf()


    allFits = {}
    listFits = []
    ks = []
    kFig = 5
    for iFile in range( 0, fileNumber ):
        fileName = args[iFile]
        print 'file:', fileName
        fits, strain, stress = fitdata( fileName, options, lengths, areas, isPlot = isPlot )
        #import pdb; pdb.set_trace()
        if not options.cycles:
            k0, k1, axLegend = fitStressStrainLines( kFig, fileName, strain, stress, options,
                                                 isPlot = isPlot )
            ks.append( (k0, k1) )
        
        if iFile == 0:
            avgFits = nm.zeros_like( fits )
        avgFits += fits
        allFits[fileName] = fits
        listFits.append( fits )
    avgFits /= float( fileNumber )

    listFits.insert( 0, avgFits )
    aFits = nm.array( listFits, dtype = nm.float64 ).T
    print aFits.shape
    
    if options.s:
        print 'output file:', fileNameOut
        #print 'output file:', op.splitext( fileNameOut )[0] + '.txt'
        #nm.savetxt(op.splitext( fileNameOut )[0] + '_moduli.txt', ks)
    
        #fd = open( fileNameOut, 'w' )
        fd = open( op.splitext( fileNameOut )[0] + '_moduli_cycles.txt', 'w' ) 
        for row in aFits:
            fd.write( ' '.join( [('%.3e' % ii) for ii in row] ) )
            fd.write( '\n' )
        fd.close()

    if not options.cycles:
        ks = nm.array( ks, dtype = nm.float64 )
        ksAverage = nm.sum( ks, 0 ) / ks.shape[0]
        ksDev = nm.std( ks, 0 )
        print ks
        print ksAverage
        print ksDev
        if options.s:
            nm.savetxt(op.splitext( fileNameOut )[0] + '_moduli.txt', ks)
    
        p.figure( kFig )
        p.xlabel( 'strain' )
        p.ylabel( 'stress [MPa]' )
        texts = [r'$E_0 = %.2e\ \pm\ %.2e$' % (ksAverage[0], ksDev[0]),
                r'$E_1 = %.2e\ \pm\ %.2e$' % (ksAverage[1], ksDev[1])]
        tt = p.getp( axLegend, 'texts' )
        p.setp( tt[0], 'text', texts[0], 'fontsize', 10 )
        p.setp( tt[1], 'text', texts[1], 'fontsize', 10 )
        if options.s:
            figName = op.splitext( fileNameOut )[0] + '_stress_strain.pdf'
            p.savefig( figName, dpi = 300 )
    
    if isLast:
        to = avgFits.shape[0]
    else:
        to = avgFits.shape[0] - 1
          
    if not options.one_cycle:
        p.figure( 1 )
        cycle = nm.arange( 1, to + 1 )
        if not isFinal:
            leg = []
            for name, fits in allFits.iteritems():
                p.plot( cycle, fits[:to] )
                leg.append( name )
            import matplotlib.font_manager as fm
            fp = fm.FontProperties()
            fp.set_size(10)
            p.legend( leg , loc = 'upper right' , prop=fp)
            #p.legend( leg , loc = 7, prop=fp)
            p.plot( cycle, avgFits[:to], 'ro' )
        p.errorbar( cycle, avgFits[:to],
                    yerr = nm.std( aFits[:to,1:], 1 ),
                    marker = 'o', mfc = 'red' )
        p.xlim( 0, aFits.shape[0] + 1 )
        p.xlabel( 'cycle number' )
        p.ylabel( 'modulus of elasticity [MPa]' )

    if options.s:
        if isFinal:
            figName = op.splitext( fileNameOut )[0] + '.pdf'
        else:
            figName = op.splitext( fileNameOut )[0] + '_all.pdf'
        p.savefig( figName, dpi = 300 )
    p.show()
    
if __name__ == '__main__':
    main()
