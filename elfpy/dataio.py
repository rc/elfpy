import os.path as op
import numpy as np

from elfpy.base import Object
from elfpy.filters import savitzky_golay

class Data(Object):
    """
    Measured data.
    """

    @classmethod
    def from_file(cls, filename, sep=' '):
        raw_data = read_data(filename, sep=sep)

        name = op.splitext(op.basename(filename))[0]
        obj = cls(name, raw_data)
        return obj

    def __init__(self, name, raw_data):
        raw_force = raw_data[:, 0]
        raw_displ = raw_data[:, 1]
        time = raw_data[:, 2]
        filtered = [False, False]

        Object.__init__(self, name=name, raw_data=raw_data,
                        raw_force=raw_force, raw_displ=raw_displ,
                        full_time=time, filtered=filtered,
                        _raw_stress=None, _raw_strain=None,
                        _stress=None, _strain=None,
                        iult=None, ultimate_stress=None, ultimate_strain=None,
                        icycle=None, cycles=[], irange=slice(None),
                        linear_fits=None,
                        strain_regions=None,
                        irange_small=None, irange_large=None,
                        linear_fit_small=None, linear_fit_large=None,
                        strains_of_stresses=None)

    def set_initial_values(self, length0=None, area0=None,
                           lengths=None, areas=None):
        if length0 is None:
            length0 = lengths[self.name]

        if area0 is None:
            area0 = areas[self.name]

        self.length0 = length0
        self.area0 = area0

        self._raw_strain = self.raw_displ / self.length0
        self._raw_stress = self.raw_force / self.area0

        self._strain = None
        self._stress = None

    @property
    def time(self):
        return self.full_time[self.irange]

    @property
    def raw_strain(self):
        return self._raw_strain[self.irange]

    @property
    def raw_stress(self):
        return self._raw_stress[self.irange]

    @property
    def strain(self):
        if self._strain is None:
            self._strain = self._raw_strain
            self.filtered[0] = False
        return self._strain[self.irange]

    @property
    def stress(self):
        if self._stress is None:
            self._stress = self._raw_stress
            self.filtered[1] = False
        return self._stress[self.irange]

    def filter_strain(self, window_size, order):
        self.filtered[0] = True
        self._strain = savitzky_golay(self.raw_strain, window_size, order)

    def filter_stress(self, window_size, order):
        self.filtered[1] = True
        self._stress = savitzky_golay(self.raw_stress, window_size, order)

    def get_cycle_indices(self, odd=True, even=True, cut_last=False):
        if not (odd or even): return []

        iis = np.arange(len(self.cycles), dtype=np.int)

        iis = iis[:-1] if cut_last else iis

        if not odd:
            iis = iis[1::2]

        elif not even:
            iis = iis[0::2]

        return iis

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

def read_data(filename, sep=' '):
    """
    Read a data file.
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
        split_row = row.replace(',', '.').split(sep)

        new = [float( ii ) for ii in split_row]
        data.append(new)


    data = np.array(data, dtype=np.float64)
    print 'shape:', data.shape
    return data

def _get_filename(datas, filename, default, suffix):
    if not filename:
        filename = default + '.' + suffix

    else:
        if r'%n' in filename:
            prefix = '_'.join([data.name for data in datas])
            filename = filename.replace('%n', prefix)

        if '.' not in filename:
            filename = filename + '.' + suffix

    return filename

def save_ultimate_values(datas, filename='', mode='w'):
    filename = _get_filename(datas, filename, 'ultimate_values', 'txt')

    fd = open(filename, mode)
    fd.write('# index, data name, cycle, ultimate strain, ultimate stress\n')
    for ii, data in enumerate(datas):
        if data.ultimate_strain is None:
            raise ValueError('use "get_ultimate_values" filter!')

        ics = 'na' if data.icycle is None else '%d' % data.icycle
        fd.write('%d, %s, %s, ' % (ii, data.name, ics))
        fd.write('%.5e, %.5e\n' % (data.ultimate_strain, data.ultimate_stress))

    fd.close()

def save_fits(datas, filename='', mode='w'):
    filename = _get_filename(datas, filename, 'linear_fits', 'txt')

    fd = open(filename, mode)
    fd.write('# index, data name, cycle, strain region1 start, stop,'
             ' stiffness1, ...\n')
    for ii, data in enumerate(datas):
        ok = 2
        if data.linear_fit_small is None:
            ok -= 1

        if data.linear_fit_large is None:
            ok -= 1

        if not ok:
            raise ValueError('use "fit_stress_strain" filter!')

        ics = 'na' if data.icycle is None else '%d' % data.icycle
        fd.write('%d, %s, %s, ' % (ii, data.name, ics))
        if data.linear_fit_small is not None:
            strain = data.strain[data.irange_small]
            fd.write('%.5e, %.5e, %.5e'
                     % (strain[0], strain[-1], data.linear_fit_small[0]))

            if ok == 1:
                fd.write('\n')

        if ok == 2:
            fd.write(', ')

        if data.linear_fit_large is not None:
            strain = data.strain[data.irange_large]
            fd.write('%.5e, %.5e, %.5e\n'
                     % (strain[0], strain[-1], data.linear_fit_large[0]))

    fd.close()

def save_cycles_fits(datas, filename='', mode='w'):
    filename = _get_filename(datas, filename, 'linear_cycles_fits', 'txt')

    fd = open(filename, mode)
    fd.write('# index, data name, cycle1, strain region1 start, stop,'
             ' stiffness1, cycle2, ...\n')
    for ii, data in enumerate(datas):
        if data.linear_fits is None:
            raise ValueError('use "fit_stress_strain_cycles" filter!')

        fd.write('%d, %s, ' % (ii, data.name))

        for iii, (ic, fit) in enumerate(data.linear_fits):
            strain = data.strain[data.cycles[ic]]
            fd.write('%d, %.5e, %.5e, %.5e'
                     % (ic, strain[0], strain[-1], fit[0]))

            cc = '\n' if (iii + 1) == len(data.linear_fits) else ', '
            fd.write(cc)

    fd.close()

def save_strain_of_stress(datas, filename='', mode='w'):
    filename = _get_filename(datas, filename, 'strain_of_stress', 'txt')

    fd = open(filename, mode)
    fd.write('# index, data name, cycle, index1, strain1, stress1, ...\n')
    for ii, data in enumerate(datas):
        if data.strains_of_stresses is None:
            raise ValueError('use "find_strain_of_stress" filter!')
        ics = 'na' if data.icycle is None else '%d' % data.icycle
        fd.write('%d, %s, %s, ' % (ii, data.name, ics))

        aux = []
        for ic, (strain, stress) in enumerate(data.strains_of_stresses):
            aux.append('%d, %.5e, %.5e' % (ic, strain, stress))
        fd.write((', '.join(aux)) + '\n')

    fd.close()
