from . import imports as ip
from .data_ops import dialog


def numpy_func():
    nt, np = ip.nt, ip.np
    # funcs = nt('functions', ['arr', 'l_and', 'l_or', 'where', 'move', 'reshape', 'lin', 'cat'])
    funcs = nt('functions', 'arr l_and l_or where move reshape lin cat')
    functions = funcs(arr=np.array,
                      l_and=np.logical_and,
                      l_or=np.logical_or,
                      where=np.where,
                      move=np.moveaxis,
                      reshape=np.reshape,
                      lin=np.linspace,
                      cat=np.concatenate)
    return functions


def unique(var=ip.np.ndarray, axis=0, **kwargs):
    """
    Function that creates unique elements of given ndarray and creates the unique
    indices, reverted unique indices based on given axis.
    :param var: Input ndarray
    :param axis: Axis along which to find uniques
    :return: List of unique ndarrays in form:
        [unique elements, reduced input ndarray, unique ids, reversed unique ids]
    """
    if not isinstance(var, ip.np.ndarray):
        raise TypeError("Defined input is not of numpy.ndarray type!")
    elif len(var.shape) > 2:
        raise AttributeError("Function is defined only for 2D ndarrays!")
    kg = kwargs.get
    ids = ip.np.linspace(0, var.shape[axis] - 1, var.shape[axis], dtype=int)
    uni_ids = ip.np.unique(var, return_index=True, axis=axis)[1]
    r_uni_ids = ids[~ip.np.isin(ids, uni_ids)]
    if axis == 0:
        var_uni = var[uni_ids, :]
        var_red = var[~ip.np.isin(ids, uni_ids), :]
    else:
        var_uni = var[:, uni_ids]
        var_red = var[:, ~ip.np.isin(ids, uni_ids)]
    if kg("uni_ids") is None:
        if kg("r_uni_ids") is None:
            return list([var_uni, var_red])
        elif kg("r_uni_ids"):
            return list([var_uni, var_red, r_uni_ids])
        else:
            return list([var_uni, var_red])
    elif kg("uni_ids"):
        if kg("r_uni_ids") is None:
            return list([var_uni, var_red, uni_ids])
        elif kg("r_uni_ids"):
            return list([var_uni, var_red, uni_ids, r_uni_ids])
        else:
            return list([var_uni, var_red, uni_ids])
    else:
        if kg("r_uni_ids") is None:
            return list([var_uni, var_red])
        elif kg("r_uni_ids"):
            return list([var_uni, var_red, r_uni_ids])
        else:
            return list([var_uni, var_red])


def sortrows(arr=ip.np.ndarray, axis=1):
    """
    User defined function for sorting rows.
    :param arr: Input numpy.ndarray to be sorted.
    :param axis: Axis along which the sort should be done.
    :return:
    """
    if not isinstance(arr, ip.np.ndarray):
        raise TypeError("Defined input is not of numpy.ndarray type!")
    else:
        for i in range(arr.shape[axis]):
            if i == 0:
                arr = arr[arr[:, - i - 1].argsort()]
            else:
                arr = arr[arr[:, - i - 1].argsort(kind='mergesort')]
        return arr


def py2mat(var=None, **kwargs):
    """
    Function that stores the python variable as matlab variable
    :param var: Given Python variable to be stored as Matlab variable
    :param kwargs: Keyword arguments
    :return: Saves the variable as Matlab .mat file
    """
    if var is None:
        raise AttributeError("No variable specified!")
    kg = kwargs.get  # kwargs get function
    file_path = dialog('-s', '.mat') if kg('path') is None else kg('path')
    ip.io.savemat(file_path, {"var": var}, appendmat=kg("appendmat"), format='5',
               long_field_names=kg("long_field_names"), do_compression=kg("do_compression"),
               oned_as=kg("oned_as"))
    return


def mat2py(**kwargs):
    """
    Function that extracts the matlab variable as python variable
    :param kwargs: Keyword arguments
    :return: Loads the variable from Matlab .mat file
    """
    kg = kwargs.get  # kwargs get function
    file_path = dialog('-o', '.mat')
    var = io.loadmat(file_path, mdict=kg("mdict"), appendmat=kg("appendmat"))
    return var['var']
