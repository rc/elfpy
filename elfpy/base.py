import inspect
from functools import wraps

from soops import Output

def type_as(arg, type_arg):
    try:
        out = type(type_arg)(arg)

    except ValueError:
        msg = 'argument "%s" cannot be converted to type %s!'
        raise ValueError(msg % (arg, type(type_arg)))

    return out

def requires(*attr_names, deps=None):

    def decorator(fun):

        @wraps(fun)
        def wrapper(data, *args, **kwargs):
            names = [name for name in attr_names if getattr(data, name) is None]
            if names:
                msgs = []
                for name in names:
                    msg = f'attribute "{name}" is not defined'
                    if deps is not None:
                        msg += f', run one of: {deps[name]}'

                    msgs.append(msg)

                raise AttributeError('; '.join(msgs))

            return fun(data, *args, **kwargs)

        wrapper.__signature__ = inspect.signature(fun)
        return wrapper

    return decorator

output = Output('elfpy:')
