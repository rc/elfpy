from soops import Output

def type_as(arg, type_arg):
    try:
        out = type(type_arg)(arg)

    except ValueError:
        msg = 'argument "%s" cannot be converted to type %s!'
        raise ValueError(msg % (arg, type(type_arg)))

    return out

output = Output('elfpy:')
