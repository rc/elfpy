import os, sys
import numpy as np
import copy
import types


_dashes = '-'*50

class Object(object):
    traits = {}

    @staticmethod
    def objects_from_dict(*args, **kwargs):

        kws = copy.copy(kwargs)

        level = kws.pop('level', 0)
        flag = kws.pop('flag')

        # For level 0 only...
        constructor = kws.pop('constructor', Object)

        out = []
        for arg in args:
            if type(arg) == dict:
                if flag[level]:
                    arg.update(kws)
                    aux = constructor(**arg)
                    iterator = aux.__dict__
                else:
                    aux = iterator = arg

                for key, val in iterator.items():
                    if (type(val) == dict) and (key != 'traits'):
                        try:
                            flag[level+1]
                        except:
                            flag = flag + (0,)
                        val2 = Object.objects_from_dict(val, level=level+1,
                                                        flag=flag)

                        if flag[level]:
                            aux.__dict__[key] = val2
                        else:
                            aux[key] = val2

                out.append(aux)
            else:
                out.append(arg)

        if len(out) == 1:
            out = out[0]

        return out

    def __init__(self, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)
            self.traits = copy.copy(self.__class__.traits)
            self.set_default_traits(kwargs.keys())

    def set_default_traits(self, keys):
        self.traits.update({}.fromkeys(keys))

    def init_trait(self, name, value, format=None):
        setattr(self, name, value)
        self.traits[name] = format

    def get_dict(self):
        aux = copy.copy(self.__dict__)
        aux.pop('traits')
        return aux

    def get(self, key, default=None, msg_if_none=None):
        """
        A dict-like get().
        """
        out = getattr(self, key, default)

        if (out is None) and (msg_if_none is not None):
            raise ValueError(msg_if_none)

        return out

    def get_not_none(self, key, default):
        val = self.get(key, default)
        if val is None:
            if default is None:
                msg = 'both attribute "%s" and default value are None!' % key
                raise ValueError(msg)

            else:
                val = default

        return val

    def __str__(self):
        return self._format()

    def __repr__(self):
        if hasattr(self, 'name'):
            return "%s:%s" % (self.__class__.__name__, self.name)
        else:
            return object.__repr__(self)

    def copy(self, deep=False):
        if deep:
            return copy.deepcopy(self)
        else:
            return copy(self)

    def _format(self, mode='print', header=True):
        if header:
            if mode != 'report':
                msg = ['%s' % object.__str__(self)]
            elif mode == 'report':
                msg = [_dashes, self.__repr__(), _dashes]
        else:
            msg = []

        keys = list(self.traits.keys())
        order = np.argsort(keys)
        for ii in order:
            key = keys[ii]
            if (key == 'name') and (mode == 'report'): continue

            val = self.traits[key]

            not_set = False
            if isinstance(val, tuple):
                tr = val[1]
                try:
                    attr = tr(getattr(self, key))
                except:
                    attr = '<not set>'
                    not_set = True
                    val = '%s'
                else:
                    val = val[0]
            elif isinstance(val, types.FunctionType):
                attr = val(self)
                val = None
            else:
                try:
                    attr = getattr(self, key)
                except:
                    attr = '<not set>'
                    not_set = True

            if not_set and (mode == 'set_only'): continue

            if issubclass(attr.__class__, Object):
                sattr = repr(attr)
                attr = '%s: %s' % (key, sattr)
            else:
                if val is None:
                    attr = '%s: %s' % (key, attr)
                else:
                    attr = ('%s: ' % key) + (val % attr)

            msg.append(attr)

        return '\n'.join(msg)

    def fd_open(self, filename):
        if isinstance(filename, str):
            fd = open(filename, 'w')

        else:
            fd = filename

        self._fd = fd
        self._filename = filename

        return fd

    def fd_close(self):
        if self._fd is not self._filename:
            self._fd.close()

    def report(self, filename, header=True):
        fd = self.fd_open(filename)
        fd.write(self._format(mode='report', header=header))
        fd.write('\n')
        self.fd_close()

class Config(Object):

    @staticmethod
    def from_file(filename, required=None, optional=None, defaults=None):
        conf_mod = import_file(filename)

        if 'define' in conf_mod.__dict__:
            define_dict = conf_mod.__dict__['define']()
        else:
            define_dict = conf_mod.__dict__

        return Config.from_conf(define_dict, required, optional, defaults)

    @staticmethod
    def from_conf(conf, required=None, optional=None, defaults=None):
        if required is None:
            required = []

        if optional is None:
            optional = [key for key in conf.keys() if key[:7] == 'options']

        if defaults is None:
            defaults = {}

        valid = {}
        for kw in required:
            try:
                val = conf[kw]
            except KeyError:
                raise ValueError('missing keyword "%s"!' % kw)
            valid[kw] = val

        for kw in optional:
            valid[kw] = conf.get(kw, None)

        for group_key, group in valid.items():
            for key, val in defaults.items():
                if key not in group:
                    group[key] = val

        return Config(**valid)

    def override(self, options, can_override):
        for group_key, group in self.__dict__.items():
            for key, val in group.items():
                if key in can_override:
                    group[key] = getattr(options, key)

class Output(Object):
    """Factory class providing output (print) functions.

    Example:

    >>> output = Output('sfepy:')
    >>> output(1, 2, 3, 'hello')
    >>> output.prefix = 'my_cool_app:'
    >>> output(1, 2, 3, 'hello')
    """
    traits = {
        'prefix' : None,
        'output_function' : None,
        'level' : None,
    }

    def __init__(self, prefix, filename=None, combined=False, **kwargs):
        Object.__init__(self, **kwargs)

        self.prefix = prefix

        self.set_output(filename, combined)

    def __call__(self, *argc, **argv):
        self.output_function(*argc, **argv)

    def set_output(self, filename=None, combined=False, append=False):
        """Set the output function - all SfePy printing is accomplished by
        it. If filename is None, output is to screen only, otherwise it is to
        the specified file, moreover, if combined is True, both the ways are
        used.

        Arguments:
                filename - print into this file
                combined - print both on screen and into a file
                append - append to an existing file instead of overwriting it
        """
        self.level = 0
        def output_screen(*argc, **argv):
            format = '%s' + ' %s' * (len(argc) - 1)
            msg =  format % argc

            if msg.startswith('...'):
                self.level -= 1

            print(self._prefix + ('  ' * self.level) + msg)

            if msg.endswith('...'):
                self.level += 1

        def output_file(*argc, **argv):
            format = '%s' + ' %s' * (len(argc) - 1)
            msg =  format % argc

            if msg.startswith('...'):
                self.level -= 1

            fd = open(filename, 'a')
            print(self._prefix + ('  ' * self.level) + msg, file=fd)
            fd.close()

            if msg.endswith('...'):
                self.level += 1

        def output_combined(*argc, **argv):
            output_screen(*argc, **argv)
            output_file(*argc, **argv)

        if filename is None:
            self.output_function = output_screen

        else:
            if not append:
                fd = open(filename, 'w')
                fd.close()

            if combined:
                self.output_function = output_combined
            else:
                self.output_function = output_file

    def get_output_function(self):
        return self.output_function

    def set_output_prefix(self, prefix):
        assert_(isinstance(prefix, str))
        if len(prefix) > 0:
            prefix += ' '
        self._prefix = prefix

    def get_output_prefix(self):
        return self._prefix[:-1]
    prefix = property(get_output_prefix, set_output_prefix)

output = Output('elfpy:')
