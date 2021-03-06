import sys
import os.path as op
import numpy as np

from elfpy.base import output, Object
from elfpy.filters import savitzky_golay

class Data(Object):
    """
    Measured data.
    """

    @classmethod
    def from_file(cls, filename, sep=' ', header_rows=2, icycles=None,
                  itime=2, idispl=1, iforce=0):
        raw_data = read_data(filename, sep=sep, header_rows=header_rows)

        name = op.splitext(op.basename(filename))[0]
        obj = cls(name, raw_data, icycles=icycles,
                  itime=itime, idispl=idispl, iforce=iforce)
        return obj

    def __init__(self, name, raw_data, icycles, itime, idispl, iforce):
        raw_force = raw_data[:, iforce]
        raw_displ = raw_data[:, idispl]
        time = raw_data[:, itime]
        filtered = [False, False]

        Object.__init__(self, name=name, icycles=icycles,
                        itime=itime, idispl=idispl, iforce=iforce,
                        raw_data=raw_data,
                        raw_force=raw_force, raw_displ=raw_displ,
                        full_time=time, filtered=filtered,
                        _raw_stress=None, _raw_strain=None,
                        _stress=None, _strain=None,
                        iult=None, ultimate_stress=None, ultimate_strain=None,
                        icycle=None, cycles=[], cycles_lengths=None,
                        irange=slice(None),
                        cycles_lin_fits=None,
                        strain_regions=None,
                        strain_regions_iranges=None,
                        strain_regions_lin_fits=None,
                        stress_regions=None,
                        stress_regions_iranges=None,
                        stress_regions_lin_fits=None,
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
    Read an info file.

    Parameters
    ----------
    filename : string
        The name of file with cross-sectional area and length information.

    Returns
    -------
    info : dict
        The values of cross-sectional area and length of the specimens,
    """
    fd = open(filename, 'r')
    info = {}
    for line in fd:
        if line and (not line.isspace()) and (line[0] != '#'):
            key, val = line.split()
            info[key] = float(val)
    fd.close()
    return info

def read_data(filename, sep=' ', header_rows=2):
    """
    Read a data file.
    """
    if sys.version_info > (3, 0):
        fd = open(filename, 'r', errors='replace')

    else:
        fd = open(filename, 'r')

    tdata = fd.readlines()
    fd.close()
    header = '\n'.join(tdata[:header_rows])
    output(header)

    tdata = tdata[header_rows:]

    output('length:', len(tdata))

    data = []
    for row in tdata:
        split_row = row.split(sep)

        new = [float(ii.replace(',', '.')) for ii in split_row]
        data.append(new)


    data = np.array(data, dtype=np.float64)
    output('shape:', data.shape)
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

def _save_regions_fits(datas, filename, mode, fit_mode):
    fd = open(filename, mode)
    fd.write('# index, data name, cycle, %s region1 start, stop,'
             ' stiffness1, ...\n' % fit_mode)
    for ii, data in enumerate(datas):
        lin_fits = getattr(data, '%s_regions_lin_fits' % fit_mode)

        if lin_fits is None:
            raise ValueError('use "fit_stress_strain" filter in %s mode!'
                             % fit_mode)

        ics = 'na' if data.icycle is None else '%d' % data.icycle
        fd.write('%d, %s, %s, ' % (ii, data.name, ics))

        iranges = getattr(data, '%s_regions_iranges' % fit_mode)
        for ii, (ik, fit) in enumerate(lin_fits):
            irange = iranges[ik]
            values = getattr(data, fit_mode)[irange]
            fd.write('%.5e, %.5e, %.5e' % (values[0], values[-1], fit[0]))

            cc = '\n' if (ii + 1) == len(lin_fits) else ', '
            fd.write(cc)

    fd.close()

def save_strain_regions_fits(datas, filename='', mode='w'):
    filename = _get_filename(datas, filename,
                             'strain_regions_linear_fits', 'txt')
    _save_regions_fits(datas, filename, mode, 'strain')

def save_stress_regions_fits(datas, filename='', mode='w'):
    filename = _get_filename(datas, filename,
                             'stress_regions_linear_fits', 'txt')
    _save_regions_fits(datas, filename, mode, 'stress')

def save_cycles_fits(datas, filename='', mode='w'):
    filename = _get_filename(datas, filename, 'linear_cycles_fits', 'txt')

    fd = open(filename, mode)
    fd.write('# index, data name, cycle1, strain region1 start, stop,'
             ' stiffness1, cycle2, ...\n')
    for ii, data in enumerate(datas):
        if data.cycles_lin_fits is None:
            raise ValueError('use "fit_stress_strain_cycles" filter!')

        fd.write('%d, %s, ' % (ii, data.name))

        for iii, (ic, fit) in enumerate(data.cycles_lin_fits):
            strain = data.strain[data.cycles[ic]]
            fd.write('%d, %.5e, %.5e, %.5e'
                     % (ic, strain[0], strain[-1], fit[0]))

            cc = '\n' if (iii + 1) == len(data.cycles_lin_fits) else ', '
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
