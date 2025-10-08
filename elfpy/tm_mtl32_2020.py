import numpy as np
import pandas as pd

from .devices import TestingMachine

class MTL32_2020(TestingMachine):
    """
    TestResources Inc. MTL32_2020 controller.
    """
    name = 'mtl32_2020'

    separator = ','
    # Workaround for malfunctioning Force1 - use Force2.
    converted_columns = dict(time=1, displ=7, force=4, cycle=2)

    def read_data(self, filename):
        mdf = pd.read_csv(filename, skiprows=5)
        mdf = mdf.rename(columns=lambda x: x.strip())
        return mdf

    def convert(self, mdf):
        """
        Convert to the format suitable for`elfpy-process`.
        """
        df = pd.DataFrame()
        df['Time [s]'] = mdf['Time sec']
        df['Cycle'] = mdf['CY-X1']
        df['Force1 [N]'] = mdf['X1L N']
        df['Force2 [N]'] = mdf['X2L N']
        df['Force [N]'] = np.minimum(df['Force1 [N]'], df['Force2 [N]'])
        df['Displacement [mm]'] = mdf['X1Disp mm'] + mdf['X2Disp mm']
        df['Elongation [mm]'] = (df['Displacement [mm]']
                                 - df['Displacement [mm]'][0])
        return df

    def get_init_length(self, mdf):
        length = mdf.loc[0, 'X1Disp mm'] + mdf.loc[0, 'X2Disp mm']
        return length
