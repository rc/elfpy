from soops import locate_files, load_classes, Struct

class TestingMachine(Struct):
    converted_columns = {}

    @staticmethod
    def any_by_name(name, **kwargs):
        """
        Create an instance of a Machine subclass according to its name.
        """
        files = locate_files('tm_*.py')
        table = load_classes(files, [TestingMachine], package_name='elfpy')
        return table[name](**kwargs)

    def get_column(self, key):
        return self.converted_columns.get(key, None)
