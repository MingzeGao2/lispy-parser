import os
import os.path
import time
import numpy as np

from mpi4py import MPI


from utils import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

Symbol = str
List = list
Number = (int, float)

def get_proc_dimensions(x_size, y_size):
    displacement = (0, )
    proc_data_size = ()
    
    normal_y_size = y_size/size
    last_y_size = y_size - normal_y_size * (size-1)
    y_size_list = [normal_y_size]*(size-1)
    y_size_list.append(last_y_size)
    proc_x_size = x_size

    for i in range(size):
        if i == size - 1:
            proc_data_size = proc_data_size + (proc_x_size*last_y_size, )
        else:
            displacement = displacement +  (displacement[-1] + normal_y_size*proc_x_size, )
            proc_data_size = proc_data_size + (proc_x_size*normal_y_size, )

    return (proc_x_size, y_size_list, proc_data_size, displacement)

def transfer_data(raster):
    # print "scatter data"
    displacement = (0,)
    proc_data_size = ()
    proc_x_size = 0
    proc_y_size = 0
    y_size_list = []
    data = None

    if rank == 0:
        (proc_x_size, y_size_list, proc_data_size, displacement) = get_proc_dimensions(raster.x_size, raster.y_size)
        data = raster.data
    proc_y_size = comm.scatter(y_size_list, root=0)
    proc_x_size = comm.bcast(proc_x_size, root=0)
    comm.Barrier()
    # print rank, proc_y_size, proc_x_size
    proc_data = np.zeros((proc_x_size, proc_y_size), dtype=np.float32)
    
    comm.Scatterv([data, proc_data_size, displacement, MPI.FLOAT], proc_data)
    comm.Barrier()
    return proc_data 

def gather_data(proc_data, x_size, y_size):
    # print "gather data"
    displacement = (0, )
    proc_data_size = ()
    if rank == 0:
        (_, _, proc_data_size, displacement) = get_proc_dimensions(x_size, y_size)
        data = np.zeros((x_size, y_size), dtype=np.float32)
    else:
        data = None
    comm.Gatherv(proc_data, [data, proc_data_size, displacement, MPI.FLOAT])
    comm.Barrier()
    if rank == 0:
        return data
    else:
        return None

def general_max(*args):
    if rank == 0:
        if len(args)==1:
            if isinstance(args[0],Raster) and args[0].data is not None:
                return np.max(args[0].data)
        else:
            return max(*args)
    else:
        return 0
def general_min(*args):
    if rank == 0:
        if len(args)==1:
            if isinstance(args[0],Raster) and args[0].data is not None:
                return np.min(args[0].data)
        else:
            return min(*args)
    else:
        return 0


def binary_op(x1, x2, op):
    x_size = 0
    y_size = 0
    nodata = None
    driver = None
    georef = None
    proj = None
    if isinstance(x1, Number) and isinstance(x2, Number):
        return op(x1, x2);
    else:
        if isinstance(x1, Raster):
            if rank == 0:
                (x_size, y_size) = x1.x_size, x1.y_size
                (nodata, driver, georef, proj) = x1.get_geo_info()
            x1 = transfer_data(x1)
        if isinstance(x2, Raster):
            if rank == 0:
                (x_size, y_size) = x2.x_size, x2.y_size
                (nodata, driver, georef, proj) = x2.get_geo_info()
            x2 = transfer_data(x2)
        proc_result = op(x1, x2)
        proc_data = gather_data(proc_result, x_size, y_size)
        comm.Barrier()
    return Raster(None, proc_data, nodata, driver, georef, proj)

    
def add(x1, x2):
    return binary_op(x1, x2, np.add)
def sub(x1, x2):
    return binary_op(x1, x2, np.subtract)
def mul(x1, x2):
    return binary_op(x1, x2, np.multiply)
def div(x1, x2):
    return binary_op(x1, x2, np.divide)


class Raster:
    
    def __init__(self, name, data=None, nodata=None, driver=None, georef=None, proj=None):
        "constructor for raster"
        if rank == 0:
            if name is None:        # intermidiate raster
                self.name = None
                self.data = data
                self.nodata = nodata
                self.driver = driver
                self.georef = georef
                self.proj = proj
                self.x_size = None if data is None  else data.shape[0]
                self.y_size = None if data is None else data.shape[1]
            else:                   # empty raster 
                if not os.path.isfile(name):
                    self.name = name
                else:               # raster already in file system
                    self.name = name
                    (self.data, self.x_size, self.y_size,
                     self.nodata, self.driver, self.georef, self.proj) = read_raster(name)
        else:
            self.name = name

    def __repr__(self):
        return '<%s.%s object at %s>' % (
            self.__class__.__module__,
            self.__class__.__name__,
            hex(id(self))
        )

        
    def get_geo_info(self):
        return (self.nodata, self.driver, self.georef, self.proj)


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
        if rank == 0:
            result = np.absolute(self.data)
        else:
            result = None
        return Raster(None, result, self.nodata, self.driver, self.georef, self.proj)        

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
    if len(tokens) == 0:
        raise SyntaxError('unexppected EOF while reading')
    token = tokens.pop(0)
    if '(' == token:
        L = []
        while tokens[0] != ')':
            L.append(read_from_tokens(tokens))
        tokens.pop(0)
        # print L
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
        '+':add, '-':sub, '*':mul, '/':div, 
        '>':op.gt, '<':op.lt, '>=':op.ge, '<=':op.le, '=':op.eq, 
        'abs':     abs,
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
        'min':     general_min,
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
        if  isinstance(var, Raster):            
            if rank == 0:
                var.write_data(result.data, 0, 0, result.nodata, result.x_size, result.y_size, 
                               result.driver, result.georef, result.proj)
            env[var.name] = var
            env[var] = var
        else:
            env[var] = result
    elif x[0] == 'set!':
        (_, var, exp) = x
        result = eval(exp, env)
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
    parsed_str = None
    while True:
        if (rank == 0):        
            parsed_str = parse(raw_input(prompt))
        comm.Barrier()
        parsed_str = comm.bcast(parsed_str, root = 0)
        comm.Barrier()
        val = eval(parsed_str)
        comm.Barrier()
        if rank == 0 and val is not None:
            print(schemestr(val))


if __name__ =='__main__':
    repl()

