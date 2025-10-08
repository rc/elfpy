import os.path as op

from soops import locate_files, load_classes, Struct

class TestingMachine(Struct):
    converted_columns = {}

    @staticmethod
    def any_by_name(name, **kwargs):
        """
        Create an instance of a Machine subclass according to its name.
        """
        return devices_table[name](**kwargs)

    def get_column(self, key):
        return self.converted_columns.get(key, None)

filedir = op.dirname(__file__)
devices_table = load_classes(locate_files(op.join(filedir, 'tm_*.py')),
                             [TestingMachine],
                             package_name='elfpy')
del filedir
