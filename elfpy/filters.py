"""
Data filters.
"""
import inspect
import numpy as np

import soops as so

from elfpy.base import type_as, output

def parse_filter_pipeline(commands, get=None, name='filters', ikw=1):
    """
    Parse commands string defining a pipeline.
    """
    if commands is None: return []

    if get is None: get = globals().get

    output('parsing %s...' % name)

    filters = []
    for ic, cmd in enumerate(commands):
        output('cmd %d: %s' % (ic, cmd))

        aux = so.parse_as_list(cmd)
        filter_name = aux[0]
        filter_args = aux[1:]

        fun = get(filter_name)
        if fun is None:
            raise ValueError('filter "%s" does not exist!' % filter_name)
        args, varargs, keywords, defaults, *rest = inspect.getfullargspec(fun)

        if defaults is None:
            defaults = []

        if len(defaults) < len(filter_args):
            raise ValueError('filter "%s" takes only %d arguments!'
                             % (filter_name, len(defaults)))

        # Process args after data.
        kwargs = {}
        for ia, arg in enumerate(args[ikw:]):
            if ia < len(filter_args):
                farg = filter_args[ia]
                if isinstance(farg, list):
                    kwargs[arg] = [type_as(ii, defaults[ia][0]) for ii in farg]

                else:
                    kwargs[arg] = type_as(farg, defaults[ia])

            else:
                kwargs[arg] = defaults[ia]

        output('using arguments:', kwargs)

        filters.append((fun, kwargs))

    output('...done')

    return filters

def list_commands(namespace=None, name='filters', arg0_name='data', ikw=1):
    """
    List all available commands in a given namespace.
    """
    if namespace is None: namespace = globals()

    head = 'available %s' % name
    output(head)
    output('-' * len(head))
    output.level += 1

    names = sorted(namespace.keys())
    for name in names:
        fun = namespace[name]
        if not inspect.isfunction(fun): continue
        if name.startswith('_'): continue

        args, varargs, keywords, defaults, *rest = inspect.getfullargspec(fun)
        if not len(args) or (args[0] != arg0_name): continue

        if defaults is not None:
            args_str = ', '.join(['%s=%s' % (args[ii], defaults[ii - ikw])
                                  for ii in range(ikw, len(args))])
        else:
            args_str = ''

        output('%s(%s)' % (name, args_str))

    output.level -= 1
    output('.')

def smooth_strain(data, window_size=35, order=3):
    data.filter_strain(window_size, order)

    return data

def smooth_stress(data, window_size=35, order=3):
    data.filter_stress(window_size, order)

    return data

def reset_strain(data):
    data._strain = None

    return data

def reset_stress(data):
    data._stress = None

    return data

def set_zero_displacement_for_force(data, force=0.0):
    """
    Set zero displacement for the given force.
    """
    data._strain = None

    ii = np.where(data.raw_force >= force)[0][0]
    u1 = data.raw_displ[ii]
    f1 = data.raw_force[ii] - force
    if f1 > 0.0:
        u0 = data.raw_displ[ii - 1]
        f0 = data.raw_force[ii - 1] - force

        uu = (f1 * u0 - f0 * u1) / (f1 - f0)

    else:
        uu = u1

    data.new_length0 = data.length0 + uu
    data.raw_displ = data.raw_displ - uu
    data._raw_strain = data.raw_displ / data.new_length0

    output('zero displacement set for force %f, index %d, u=%f'
           % (force, ii, uu))
    output('length0 = %f, new length0 = %f'
           % (data.length0, data.new_length0))
    output('new u0 = %f, new u1 = %f, f0 = %f, f1 = %f'
           % (data.raw_displ[ii-1], data.raw_displ[ii],
              data.raw_force[ii-1], data.raw_force[ii]))

    return data

def _update_cycles_attrs(data, cycles):
    data.cycles = np.array(cycles)
    data.cycles_lengths = np.array([cc.stop - cc.start for cc in data.cycles])
    data.cycles_dt = np.array([data.time[ii.stop-1] - data.time[ii.start]
                               for ii in data.cycles])

    return data

def use_data_cycles(data):
    """
    Separation of individual cycles using `icycles` field in the data file.

    Notes
    -----
    Sets `cycles`, `cycles_lengths` attributes of `data`. Then, for example,
    `data.strain[data.cycles[ii]]` gives the strain in the ii-th cycle.
    """
    if data.icycles is None:
        raise ValueError('"cycle" column index in data is not set!')

    dcycles = data.raw_data[:, data.icycles]
    ii = np.where(np.ediff1d(dcycles[:-1], to_begin=-1, to_end=-2))[0]

    cycles = [slice(ii[ir], ii[ir+1]) for ir in range(len(ii) - 1)]

    data = _update_cycles_attrs(data, cycles)
    return data

def detect_strain_cycles(data):
    """
    Automatic separation of individual cycles in the case of cyclic
    displacement-induced loading. The process is based on the finding the
    indices where the first time derivative of strain changes its sign.

    Notes
    -----
    Sets `cycles`, `cycles_lengths` attributes of `data`. Then, for example,
    `data.strain[data.cycles[ii]]` gives the strain in the ii-th cycle.
    """
    # First time derivative of strain.
    dstrain = np.diff(data.strain) / np.diff(data.time)

    # Sign change.
    sign = np.sign(dstrain)
    ii = np.where(np.abs(np.ediff1d(sign, to_begin=2, to_end=2)) >= 1)[0]

    cycles = [slice(ii[ir], ii[ir+1]) for ir in range(len(ii) - 1)]

    data = _update_cycles_attrs(data, cycles)
    return data

def detect_strain_cycles2(data, eps=0.01):
    """
    Automatic separation of individual cycles in the case of cyclic
    displacement-induced loading. The process is based on the finding the
    indices where the first time derivative of strain is almost zero. The
    ranges where the first time derivative of strain is almost zero are left
    out.

    Notes
    -----
    Sets `cycles`, `cycles_lengths` attributes of `data`. Then, for example,
    `data.strain[data.cycles[ii]]` gives the strain in the ii-th cycle.
    """
    # First time derivative of strain.
    dstrain = np.diff(data.strain) / np.diff(data.time)

    aeps = eps * (dstrain.max() - dstrain.min())
    ii = np.where(np.abs(dstrain) < aeps)[0]

    runs = np.ediff1d(ii, to_end=2)
    ir = np.where(runs > 1)[0]

    cycles = [slice(ii[ir[ic]], ii[ir[ic] + 1]) for ic in range(len(ir) - 1)]
    if cycles[-1].stop < len(dstrain):
        cycles.append(slice(cycles[-1].stop, len(dstrain)))

    data = _update_cycles_attrs(data, cycles)
    return data

def print_cycles_info(data):
    """
    Print various information about cycles.
    """
    n_cycles = len(data.cycles)
    output('number of cycles:', n_cycles)
    if not n_cycles:
        return data

    slengths = sorted(set(data.cycles_lengths))
    lhist, lbins = np.histogram(data.cycles_lengths,
                                bins=slengths + [slengths[-1] + 1])
    output('histogram of cycle lengths (length: count [duration interval]):')
    for ib, lbin in enumerate(lbins[:-1]):
        ii = np.where(data.cycles_lengths == lbin)[0]
        cdt = data.cycles_dt[ii]
        output('  % 5d: % 5d [%10.5e, %10.5e]'
               % (lbin, lhist[ib], cdt.min(), cdt.max()))

    i0 = data.cycles_lengths.min()
    i1 = np.where(data.cycles_lengths == i0)[0]
    output('shortest cycle length: %d (in %d cycle(s))' % (i0, len(i1)))
    cdt = data.cycles_dt[i1]
    output('  duration in [%10.5e, %10.5e]' % (cdt.min(), cdt.max()))

    i0 = data.cycles_lengths.max()
    i1 = np.where(data.cycles_lengths == i0)[0]
    output('longest cycle length: %d (in %d cycle(s))' % (i0, len(i1)))
    cdt = data.cycles_dt[i1]
    output('  duration in [%10.5e, %10.5e]' % (cdt.min(), cdt.max()))

    output('data min., mean, max. in the longest cycle(s):')
    for ii in i1:
        strain = data.strain[data.cycles[ii]]
        stress = data.stress[data.cycles[ii]]
        output('  cycle:', ii)
        output('    strain: %10.5e, %10.5e, %10.5e'
               % (strain.min(), strain.mean(), strain.max()))
        output('    stress: %10.5e, %10.5e, %10.5e'
               % (stress.min(), stress.mean(), stress.max()))

    return data

def remove_short_cycles(data, min_length=2):
    """
    Remove cycles with length smaller than `min_length`.

    Notes
    -----
    Modifies `cycles`, `cycles_lengths` attributes of `data`.
    """
    ii = np.where(data.cycles_lengths >= min_length)[0]
    data.cycles = data.cycles[ii]
    data.cycles_lengths = data.cycles_lengths[ii]

    return data

def select_cycle(data, cycle=-1, min_length=0):
    """
    Select current cycle.

    Notes
    -----
    Calls automatically :func:`detect_strain_cycles()` if needed. Sets `irange`
    attribute of `data`.
    """
    if not len(data.cycles):
        data = detect_strain_cycles(data)

    data.icycle = cycle
    try:
        data.irange = data.cycles[cycle]

    except IndexError:
        output('cycle %d is not present, using the last one!' % cycle)
        data.icycle = -1
        data.irange = data.cycles[-1]

    if min_length > 0:
        if data.icycle >= 0:
            islice = slice(data.icycle, None)
            sign = 1

        else:
            islice = slice(data.icycle, None, -1)
            sign = -1

        ic = np.where(data.cycles_lengths[islice] >= min_length)[0]
        if len(ic):
            data.icycle += sign * ic[0]
            output('cycle with min. length %d found: %d (length %d)' %
                   (min_length, data.icycle, data.cycles_lengths[data.icycle]))

        else:
            max_ic = data.cycles_lengths[islice].argmax()
            data.icycle += sign * max_ic
            output('no cycle with min. length %d, using the longest cycle %d'
                   ' (length %d)!' %
                   (min_length, data.icycle, data.cycles_lengths[data.icycle]))

        data.irange = data.cycles[data.icycle]

    return data

def get_ultimate_values(data, eps=0.1):
    """
    Get ultimate stress and strain.
    """
    stress = data.stress
    dstress = np.diff(stress, n=1)/ np.diff(data.strain, n=1)

    ii = np.where(dstress < 0)[0]
    if len(ii) == 0:
        output('warning: stress does not decrease')
        iult = stress.shape[0] - 1

    else:
        iult = np.where(stress[ii] > (eps * stress.max()))[0]
        if len(iult) == 0:
            iult = ii[0]
            output('warning: ultimate stress is less then %f*max stress' % eps)

        else:
            iult = ii[iult[0]]

    data.iult = iult
    output('index of ultimate strength:', iult)

    data.ultimate_strain = data.strain[iult]
    data.ultimate_stress = stress[iult]
    output('ultim. strain, ultim. stress:',
           data.ultimate_strain, data.ultimate_stress)

    return data

def detect_linear_regions(data, eps_r=0.01, run=10):
    """
    Detect linear-like regions of stress-strain curve (i.e. the regions of
    small and large deformations). The first and last regions are identified
    with small and large deformation linear regions.

    Notes
    -----
    Sets `strain_regions` and `strain_regions_iranges` attributes of `data`.
    """
    stress = data.stress
    window_size = max(int(0.001 * stress.shape[0]), 35)

    ds = savitzky_golay(stress, window_size, 3, 1)
    de = savitzky_golay(data.strain, window_size, 3, 1)
    dstress = ds / de
    ddstress = savitzky_golay(dstress, window_size, 3, 1)

    p1 = np.where(dstress >= 0)[0]
    p2 = np.ediff1d(p1, to_end=2)
    p3 = np.where(p2 > 1)[0]
    if p3[0] == 0 or p3[0] == 1:
        index_value = p1[-1]
    else:
        index_value = p1[p3][0]
    output('index_value:', index_value) # Usually equal to data.iult.

    ddstress = ddstress[:index_value]
    addstress = np.abs(ddstress)
    eps = eps_r * addstress.max()
    ii = np.where(addstress < eps)[0]
    idd = np.ediff1d(ii)
    ir = np.where(idd > 1)[0]

    run_len = int((run * index_value) / 100.)

    regions = []
    ic0 = 0
    for ic in ir:
        region = slice(ii[ic0], ii[ic] + 1)
        ic0 = ic + 1
        if (region.stop - region.start) >= run_len:
            regions.append(region)

    output('%d region(s)' % len(regions))

    data.strain_regions_iranges = regions
    data.strain_regions = [(data.strain[ii.start], data.strain[ii.stop])
                           for ii in data.strain_regions_iranges]

    return data

def _find_irange(values, val0, val1, msg='wrong range'):
    """
    Find the first consecutive range [i0, i1] in `values` such that
    values[i0] is the first value such that `val0 <= values[i0]` and
    values[i1] is the last value such that `values[i1] <= val1`.
    """
    assert(val0 < val1)

    i0 = np.where((values[:-1] <= val0) & (val0 <= values[1:]))[0]
    i1 = np.where((values[:-1] <= val1) & (val1 <= values[1:]))[0]
    if len(i0) and len(i1):
        irange = slice(i0[0] + 1, i1[-1] + 1)

    else:
        raise ValueError('%s! ([%.2e, %.2e] in [%.2e, %.2e])'
                         % (msg, val0, val1, values.min(), values.max()))

    output('required: [%s, %s], found: [%s, %s]'
           % ((val0, val1, values[irange.start], values[irange.stop - 1])))

    return irange

def _find_iranges(values, ranges, msg='wrong range'):
    """
    Find `ranges` in `values` by calling :func:`_find_irange()` for each couple
    in `ranges`.
    """
    assert((len(ranges) % 2) == 0)

    ranges = np.asarray(ranges).reshape((-1, 2))

    iranges = []
    for rng in ranges:
        irange = _find_irange(values, rng[0], rng[1], msg)
        iranges.append(irange)

    return iranges

def set_strain_regions(data, def_s0=-1.0, def_s1=-1.0,
                       def_l0=-1.0, def_l1=-1.0):
    """
    Set the regions of small and large deformations.

    If positive, [def_s0, def_s1] strain range is set as small deformations,
    [def_l0, def_l1] as large deformations.

    Notes
    -----
    Sets `strain_regions` and `strain_regions_iranges` attributes of `data`.
    """
    data.strain_regions_iranges = []

    if (def_s0 >= 0.0) and (def_s1 >= 0):
        irange = _find_irange(data.strain, def_s0, def_s1,
                              'wrong small deformation range')
        data.strain_regions_iranges.append(irange)

    if (def_l0 >= 0.0) and (def_l1 >= 0):
        irange = _find_irange(data.strain, def_l0, def_l1,
                              'wrong large deformation range')
        data.strain_regions_iranges.append(irange)

    data.strain_regions = [(data.strain[ii.start], data.strain[ii.stop])
                           for ii in data.strain_regions_iranges]

    return data

def set_strain_regions_list(data, ranges=[0.0, 1.0]):
    """
    Set a list of n strain regions.

    The `ranges` argument is a list of 2n floats - beginning and end strain for
    each region.

    Notes
    -----
    Sets `strain_regions` and `strain_regions_iranges` attributes of `data`.
    """
    data.strain_regions_iranges = _find_iranges(data.strain, ranges,
                                                msg='wrong strain range')
    data.strain_regions = [(data.strain[ii.start], data.strain[ii.stop])
                           for ii in data.strain_regions_iranges]

    return data

def set_stress_regions_list(data, ranges=[0.0, 1.0]):
    """
    Set a list of n stress regions.

    The `ranges` argument is a list of 2n floats - beginning and end stress for
    each region.

    Notes
    -----
    Sets `stress_regions` and `stress_regions_iranges` attributes of `data`.
    """
    data.stress_regions_iranges = _find_iranges(data.stress, ranges,
                                                msg='wrong stress range')
    data.stress_regions = [(data.stress[ii.start], data.stress[ii.stop])
                           for ii in data.stress_regions_iranges]

    return data

def set_ring_test_strain(data, diameter=1.0, thickness=0.0, relative=True):
    """
    Set strain attribute to a strain corresponding to a ring test with the
    given pin diameter and the displacements. If a non-zero thickness is given,
    the strain in the specimen mid-surface is computed, instead of the default
    strain of the inner surface.

    If `relative` is True, the displacements are considered to be relative.
    """
    data._strain = None

    dl = data.raw_displ if relative else (data.raw_displ - data.length0)

    data._raw_strain = (2.0 * dl
                        / (2.0 * (data.length0 + diameter)
                           + np.pi * (diameter + thickness)))

    return data

def find_strain_of_stress(data, stresses=[0.0]):
    """
    For every given stress value, find the smallest strain on the stress-strain
    curve that it (approximately) corresponds to.
    """
    stress = data.stress
    data.strains_of_stresses = []
    for ic, val in enumerate(stresses):
        iw = np.where((stress[:-1] < val) & (val < stress[1:]))[0]
        if len(iw):
            iw = iw[0]
            item = (data.strain[iw], stress[iw])

        else:
            item = (np.nan, np.nan)

        data.strains_of_stresses.append(item)

    return data

def _fit_stress_strain(stress, strain):
    return np.polyfit(strain, stress, 1)

def fit_stress_strain(data, region_kind='strain', which=[-999]):
    """
    Determine Young's modulus of elasticity in the selected regions.

    Special value of `which` equal to [-999] means all regions.

    Notes
    -----
    Sets `strain_regions_lin_fits` or `stress_regions_lin_fits` attribute of
    `data`, according to `region_kind`.
    """
    if region_kind == 'strain':
        iranges = data.strain_regions_iranges
        lin_fits = data.strain_regions_lin_fits = []

    elif region_kind == 'stress':
        iranges = data.stress_regions_iranges
        lin_fits = data.stress_regions_lin_fits = []

    else:
        raise ValueError('unknown region kind! (%s)' % region_kind)

    if which == [-999]:
        which = range(len(iranges))

    for ii in which:
        try:
            indx = iranges[ii]

        except IndexError:
            raise IndexError('%s region %d does not exist!' % (region_kind, ii))

        output('%s index range: (%d, %d)'
               % (region_kind, indx.start, indx.stop))
        out = _fit_stress_strain(data.stress[indx], data.strain[indx])
        lin_fits.append((ii, out))

    return data

def _fit_stress_strain_cycles(data, ics):
    data.cycles_lin_fits = []
    nc = len(data.cycles)
    for ii, ic in enumerate(ics):
        ic = ic if ic >= 0 else nc + ic
        irange = data.cycles[ic]
        out = _fit_stress_strain(data.stress[irange], data.strain[irange])
        data.cycles_lin_fits.append((ic, out))

    return data

def fit_stress_strain_cycles(data, odd=1, even=1, cut_last=1):
    """
    Determine overall Young's modulus of elasticity in the selected cycles.

    Notes
    -----
    Sets `cycles_lin_fits`` attribute of `data`.
    """
    if not len(data.cycles):
        data = detect_strain_cycles(data)

    ics = data.get_cycle_indices(odd, even, cut_last)
    return _fit_stress_strain_cycles(data, ics)

def fit_stress_strain_cycles_list(data, ics=[0]):
    """
    Determine overall Young's modulus of elasticity in the listed cycles.

    Notes
    -----
    Sets `cycles_lin_fits`` attribute of `data`.
    """
    if not len(data.cycles):
        data = detect_strain_cycles(data)

    return _fit_stress_strain_cycles(data, ics)

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    This function was taken from SciPy Cookbook.

    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only
        smoothing)

    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    from math import factorial

    try:
        window_size = np.abs(np.int32(window_size))
        order = np.abs(np.int32(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.array([[k**i for i in order_range]
                  for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b)[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')
