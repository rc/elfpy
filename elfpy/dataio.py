import numpy as np

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
