import os
import os.path
import time
import numpy as np

from mpi4py import MPI


from utils import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def general_max(*args):
    if len(args)==1:
        if isinstance(args[0],Raster):
            return np.max(args[0].data)
    else:
        return max(*args)

class File:
    def __init__(self, name, data=None):
        if not os.path.isfile(name):
            if name == 'result':
                open(self.name, 'w+')
            else:
                raise IOError("No such file as %s"%name)
        self.name = name
        self.data = data

class Raster(File):
    
    def __init__(self, name, data=None, nodata=None, driver=None, georef=None, proj=None):
        "constructor for raster"
        if name is None:
            self.name = None
            self.data = data
            self.nodata = nodata
            self.driver = driver
            self.georef = georef
            self.proj = proj
            self.x_size = data.shape[1]
            self.y_size = data.shape[0]
        else:
            if not os.path.isfile(name):
                self.name = name
            else:
                self.name = name
                (self.data, self.x_size, self.y_size,
                 self.nodata, self.driver, self.georef, self.proj) = read_raster(name)

    def __repr__(self):
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

    def __str__(self):
        print self.__repr__()
        if self.data is not None:
            return  str(self.data)
        elif self.name is not None:
            return self.name
        else:
            return ' '
    def __add__(self, other):
        if isinstance(other, Raster):
            result = np.add(self.data, other.data)
        else:
            result = np.add(self.data, other)
        
        return Raster(None, result, self.nodata, self.driver, self.georef, self.proj)
              
    def __radd__(self, other):
        if isinstance(other, Raster):
            result = np.add(self.data, other.data)
        else:
            result = np.add(self.data, other)
        return Raster(None, result, self.nodata, self.driver, self.georef, self.proj)
    
    def __abs__(self):
        result = np.absolute(self.data)
        return Raster(None, result, self.nodata, self.driver, self.georef, self.proj)
        
    # def __sub__(self, other):
    #     if isinstance(other, Raster):
    #         result = self.data - other.data
    #     elif isinstance(other[0], np.ndarray):
    #         result = np.subtract(self.data, other)
    #     else:
    #         raise TypeError("only raster file can be subtract from raster file\n")
    #     return ( result, self.x_size, self.y_size,
    #              self.nodata, self.driver, self.georef, self.proj)
        
    # def __rsub__(self, other):
    #     if isinstance(other, Raster):
    #         result =  other.data - self.data
    #     elif isinstance(other[0], np.ndarray):
    #         result = np.subtract(other, self.data)
    #     else:
    #         raise TypeError("only raster file can be subtract from raster file\n")
    #     return ( result, self.x_size, self.y_size,
    #              self.nodata, self.driver, self.georef, self.proj)
    # def __mul__(self, other):
    #     if isinstance(other, Raster):
    #         result = np.multiply(self.data, other.data)
    #     elif isinstance(other[0], np.ndarray):
    #         result = np.multiple(other, self.data)
    #     else:
    #         raise TypeError("only raster file can be multiply to raster file\n")
    #     return ( result, self.x_size, self.y_size,
    #              self.nodata, self.driver, self.georef, self.proj)
    # def __rmul__(self, other):
    #     if isinstance(other, Raster):
    #         result = np.multiply(self.data, other.data)
    #     elif isinstance(other, np.ndarray):
    #         result = np.multiple(self.data, other)
    #     else:
    #         raise TypeError("only raster file can be multiply to raster file\n")
    #     return ( result, self.x_size, self.y_size,
    #              self.nodata, self.driver, self.georef, self.proj)


    def write_data(self, data, x_offset, y_offset, nodata,
              x_size, y_size, driver, georef, proj ):
        self.x_size = x_size;
        self.y_size = y_size;
        self.driver = driver
        self.nodata = nodata
        self.data = data
        self.georef = georef
        self.proj = proj
        dataset = create_raster(x_size, y_size, self.name, 
                                driver, georef, proj)
        write_raster(dataset, data, x_offset, y_offset, nodata)
        
Symbol = str
List = list
Number = (int, float)



class Procedure(object):
    "A user defined Scheme procedure."
    def __init__(self, parms, body, env):
        self.parms, self.body, self.env = parms, body, env
    def __call__(self, *args):
        return eval(self.body, Env(self.parms, args, self.env))

class Env(dict):
    "An environment: a dict of {'var' : val} pairs, with an outer Env."
    def __init__(self, parms=(), args=(), outer=None):
        self.update(zip(parms, args))
        self.outer = outer
    def find(self, var):
        "Find the innermost Env where var appears."
        return self if (var in self) else self.outer.find(var)

def tokenize(chars):
    return chars.replace('(', ' ( ').replace(')', ' ) ').split()

def parse(program):
    return read_from_tokens(tokenize(program))

def read_from_tokens(tokens):
    print tokens
    if len(tokens) == 0:
        raise SyntaxError('unexppected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        print L
        return L
    elif ')' == token:
        raise SyntaxError('unexpected )')
    else:
        return atom(token)

def atom(token):
    "Number become numbers, every other token is a symbol."
    try: return int(token)
    except ValueError:
        try: return float(token)
        except ValueError:
            if token[0] == '"' and token[-1] == '"':
                name = token[1:-1]
                try: return global_env.find(name)[name]
                except: return Raster(name)
            else:
                return Symbol(token)
            

def standard_env():
    "An environment with some Scheme standard procedures."
    import math, operator as op
    env = Env()
    env.update(vars(math)) # sin, cos, sqrt, pi, ...
    env.update({
        '+':op.add, '-':op.sub, '*':op.mul, '/':op.div, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
        'append':  op.add,  
        'apply':   apply,
        'begin':   lambda *x: x[-1],
        'car':     lambda x: x[0],
        'cdr':     lambda x: x[1:], 
        'cons':    lambda x,y: [x] + y,
        'eq?':     op.is_, 
        'equal?':  op.eq, 
        'length':  len, 
        'list':    lambda *x: list(x), 
        'list?':   lambda x: isinstance(x,list), 
        'map':     map,
        'max':     general_max,
        'min':     min,
        'not':     op.not_,
        'null?':   lambda x: x == [], 
        'number?': lambda x: isinstance(x, Number),   
        'procedure?': callable,
        'round':   round,
        'symbol?': lambda x: isinstance(x, Symbol),
    })
    return env

global_env = standard_env()

def eval(x, env=global_env): 
    if isinstance(x, Symbol):
        return env.find(x)[x]
    elif not isinstance(x, List):
        return x 
    elif x[0] == 'quote':
        (_, exp) = x
        return exp
    elif x[0] == 'if':
        (_, test, conseq, alt) = x
        exp = (conseq if eval(test, env) else alt)
        return eval(exp, env)
    elif x[0] == 'define':
        (_, var, exp) = x
        result = eval(exp, env)
        if isinstance(var, Raster):            
            var.write_data(result.data, 0, 0, result.nodata, result.x_size, result.y_size, 
                      result.driver, result.georef, result.proj)
            env[var.name] = var
            env[var] = var
        else:
            env[var] = result
    elif x[0] == 'set!':
        (_, var, exp) = x
        result = eval(exp, env)
        if isinstance(var, Raster):            
            var.write_data(result.data, 0, 0, result.nodata, result.x_size, result.y_size, 
                      result.driver, result.georef, result.proj)
        else:
            env.find(var)[var] = result

    elif x[0] == 'lambda':
        (_, parms, body) = x
        return Procedure(parms, body, env)
    else:
        proc = eval(x[0], env)
        args = [eval(arg, env) for arg in x[1:]]
        return proc(*args)


def schemestr(exp):
    "Convert a Python object back into a Schem-readable string."
    if isinstance(exp, list):
        return '(' + ' ' .join(map(schemestr, exp)) + ')'
    else:
        return str(exp)

def repl(prompt='lis.py>'):
    "A prompt-read-eval-print loop."
    while True:
        parsed_str = parse(raw_input(prompt))
        print("start to eval")
        val = eval(parsed_str)
        if val is not None:
            print(schemestr(val))


if __name__ =='__main__':
    if rank == 0:
        repl()

