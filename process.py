#!/usr/bin/env python
# c: 28.01.2008, r: 11.02.2008
import sys, re
import os.path as op
import numpy as nm
import pylab as p

lengths = { # mm
    'carp_001' : 22,
    'carp_002' : 22,
    'carp_003' : 22,
    'carp_004' : 22,
    'carp_005' : 22,
    'carp_006' : 22,
    'carp_007' : 22,
    'carp_008' : 22,
    'carp_009' : 22,
    'carp_010' : 22,
    'carp_011' : 22,
    'carp_012' : 22,
    'carp_013' : 22,
}

areas = { # mm^2
    'carp_001' : 7.600,
    'carp_002' : 10.620,
    'carp_003' : 8.640,
    'carp_004' : 9.000,
    'carp_005' : 10.740,
    'carp_006' : 8.700,
    'carp_007' : 13.260,
    'carp_008' : 12.360,
    'carp_009' : 12.960,
    'carp_010' : 24.240,
    'carp_011' : 41.020,
    'carp_012' : 9.540,
    'carp_013' : 7.920,
}




def splitChunks( strain, time, epsR = 0.01, split = False, append = False ):
    dstrain = nm.diff( strain ) / nm.diff( time )
    eps = epsR * (dstrain.max() - dstrain.min())
    ii = nm.where( nm.abs( dstrain ) < eps )[0]

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
        aii = nm.concatenate( chunks )
        return aii, chunks
    else:
        return ii
    
def fitdata( fileName, isPlot = 0 ):
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

    ii, chunks = splitChunks( strain, time, split = True, append = True )
    chunks.append( nm.array( [data.shape[0] - 1] ) )
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

    if isPlot:
        p.figure( 4 )
        p.plot( fits )
        p.show()
    return fits, strain[i0:i1+1], force[i0:i1+1]

def fitStressStrainLines( fig, fileName, strain, stress,
                          isPlot = False ):
    x, y = strain, stress
    ihalf = y.shape[0] / 2
    imax1 = nm.argmax( y[ihalf:] ) + ihalf - 1
    imin0 = nm.argmin( y[:ihalf] ) + 1

    dimax = 20 * y.shape[0] / 100
    imax0 = imax1 - dimax
    imin1 = imin0 + dimax

    out0 = nm.polyfit( x[imin0:imin1+1], y[imin0:imin1+1], 1 )
    out1 = nm.polyfit( x[imax0:imax1+1], y[imax0:imax1+1], 1 )

    p.figure( fig )
    if isPlot:
        p.clf()
    p.plot( x, y, 'b-o', markersize = 3 )
    p.plot( x[[imin0,imin1]], y[[imin0,imin1]], 'gs' )
    p.plot( x[[imax0,imax1]], y[[imax0,imax1]], 'rs' )
    h0 = p.plot( x[:ihalf], out0[0] * x[:ihalf] + out0[1],
                 'g', linewidth = 1.5 )
    h1 = p.plot( x[ihalf:], out1[0] * x[ihalf:] + out1[1],
                 'r', linewidth = 1.5 )
    axLegend = p.legend( [h0, h1], ['0', '1'],
                         loc = 'upper left' )
    if isPlot:
        p.show()

    return out0[0], out1[0], axLegend

def main():
    isFinal = False
    isLast = False
    isPlot = 0

    fileNumber = len( sys.argv ) - 1
    fileNameOut = sys.argv[fileNumber]

    allFits = {}
    listFits = []
    ks = []
    kFig = 5
    for iFile in range( 1, fileNumber ):
        fileName = sys.argv[iFile]
        print 'file:', fileName
        fits, strain, stress = fitdata( fileName, isPlot = isPlot )
        k0, k1, axLegend = fitStressStrainLines( kFig, fileName, strain, stress,
                                                 isPlot = isPlot )
        ks.append( (k0, k1) )
        
        if iFile == 1:
            avgFits = nm.zeros_like( fits )
        avgFits += fits
        allFits[fileName] = fits
        listFits.append( fits )
    avgFits /= float( fileNumber - 1 )

    listFits.insert( 0, avgFits )
    aFits = nm.array( listFits, dtype = nm.float64 ).T
    print aFits.shape

    print 'output file:', fileNameOut
    fd = open( fileNameOut, 'w' )
    for row in aFits:
        fd.write( ' '.join( [('%.3e' % ii) for ii in row] ) )
        fd.write( '\n' )
    fd.close()

    ks = nm.array( ks, dtype = nm.float64 )
    ksAverage = nm.sum( ks, 0 ) / ks.shape[0]
    ksDev = nm.std( ks, 0 )
    print ks
    print ksAverage
    print ksDev

    p.figure( kFig )
    p.xlabel( 'strain' )
    p.ylabel( 'stress [MPa]' )
    texts = [r'$E_0 = %.2e\ \pm\ %.2e$' % (ksAverage[0], ksDev[0]),
             r'$E_1 = %.2e\ \pm\ %.2e$' % (ksAverage[1], ksDev[1])]
    tt = p.getp( axLegend, 'texts' )
    p.setp( tt[0], 'text', texts[0], 'fontsize', 14 )
    p.setp( tt[1], 'text', texts[1], 'fontsize', 14 )
    figName = op.splitext( fileNameOut )[0] + '_stress_strain.pdf'
    p.savefig( figName, dpi = 300 )
    
    if isLast:
        to = avgFits.shape[0]
    else:
        to = avgFits.shape[0] - 1

    p.figure( 1 )
    cycle = nm.arange( 1, to + 1 )
    if not isFinal:
        leg = []
        for name, fits in allFits.iteritems():
            p.plot( cycle, fits[:to] )
            leg.append( name )
        p.legend( leg )
        p.plot( cycle, avgFits[:to], 'ro' )
    p.errorbar( cycle, avgFits[:to],
                yerr = nm.std( aFits[:to,1:], 1 ),
                marker = 'o', mfc = 'red' )
    p.xlim( 0, aFits.shape[0] + 1 )
    p.xlabel( 'cycle number' )
    p.ylabel( 'modulus of elasticity [MPa]' )

    if isFinal:
        figName = op.splitext( fileNameOut )[0] + '.pdf'
    else:
        figName = op.splitext( fileNameOut )[0] + '_all.pdf'
    p.savefig( figName, dpi = 300 )
    p.show()
    
    
if __name__ == '__main__':
    main()
