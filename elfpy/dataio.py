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
                        icycle=None, cycles=[], irange=slice(None))

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
            self._strain = self.raw_strain
            self.filtered[0] = False
        return self._strain[self.irange]

    @property
    def stress(self):
        if self._stress is None:
            self._stress = self.raw_stress
            self.filtered[1] = False
        return self._stress[self.irange]

    def filter_strain(self, window_size, order):
        self.filtered[0] = True
        self._strain = savitzky_golay(self.raw_strain, window_size, order)

    def filter_stress(self, window_size, order):
        self.filtered[1] = True
        self._stress = savitzky_golay(self.raw_stress, window_size, order)

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

def save_ultimate_values(datas, filename='', mode='w'):
    if not filename:
        filename = 'ultimate_values.txt'

    fd = open(filename, mode)
    fd.write('# index, data name, cycle, ultimate strain, ultimate stress\n')
    for ii, data in enumerate(datas):
        if data.ultimate_strain is None:
            raise ValueError('use "get_ultimate_values" filter!')

        ics = 'na' if data.icycle is None else '%d' % data.icycle
        fd.write('%d, %s, %s, ' % (ii, data.name, ics))
        fd.write('%.5e, %.5e\n' % (data.ultimate_strain, data.ultimate_stress))

    fd.close()
