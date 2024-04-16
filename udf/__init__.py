"""
UDF Library
===========================
Library of user defined functions.

Functions
---------
--> Mat functions
    sortrows
    unique
    py2mat
    mat2py
    NumPy functions
--> Data functions
    dialog
    save
    load
--> Other functions
    gpu_init
    create_gif
    tb_create
    tb_show
--> Classes
    sequence
    opt
"""
__version__ = "1.1.0"
from . import imports as ip
from . import data_ops as d
from . import mat_ops as m
from . import other_ops as o
from . import class_ops as c

# --> Mat functions
sortrows = m.sortrows
unique = m.unique
py2mat = m.py2mat
mat2py = m.mat2py
# --> NumPy functions
nf = m.numpy_func()

# --> Data functions
dialog = d.dialog
save = d.save
load = d.load

# --> Other functions
gpu_init = o.gpu_init
create_gif = o.create_gif
tb_create = o.tb_create
tb_show = o.tb_show
TbWriter = o.TbWriter
tf = None


# --> Classes
def sequence():
    """
    Sequence class initiate function
    :return: Sequence
    """
    import_tf()
    from . import spec_class_ops as s
    return s.sequence()


def opt():
    """
    Options class initiate function
    :return: Opt
    """
    return c.opt(dialog('-o', '.xlsx'))


def memory(**kwargs):
    """
    Memory class initiate function
    :param kwargs: Type and samples selection
    :return: Memory
    """
    return c.memory(**kwargs)


def import_tf():
    """
    Function that imports Tensorflow only if neccessary to do so.
    :return: Initialized Tensorflow
    """
    global tf
    tf = ip.import_tf()
    return

def BO_df(**kwargs):
    """
    BO_df class initiate function
    :return: BO_dataframe
    """
    return c.BO_df(**kwargs)