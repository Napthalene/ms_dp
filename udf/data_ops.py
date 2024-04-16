from . import imports as ip


def dialog(dtype='-o', ftype='.pckl', name=None):
    # type: (str, str) -> str
    """
    Function opening the dialog window to save or to load a file.
    :param dtype: Type of dialog: -o represent open file, -s represents save file.
    :param ftype: Type of file to save or load, default .pckl
    :return: File path
    """
    if not any([dtype == t for t in ['-o', '-s', '-d']]):
        raise TypeError("Unknown dialog window type!")
    root = ip.Tk()
    root.withdraw()
    if name is None:
        name = 'open_or_save'

    if dtype == '-o':
        wrapper = ip.filedialog.askopenfile(
            initialfile=name + ftype, filetypes=(("Files", "*{}".format(ftype)), ("All files", "*.*")))
    elif dtype == '-s':
        wrapper = ip.filedialog.asksaveasfile(
            initialfile=name + ftype, filetypes=(("Files", "*{}".format(ftype)), ("All files", "*.*")))
    elif dtype == '-d':
        file_path = ip.filedialog.askdirectory()
        if file_path is None:
            raise AttributeError("No file specified!")
        else:
            return file_path
    if wrapper is None:
        raise AttributeError("No file specified!")
    else:
        file_path = wrapper.name
        del wrapper
    return file_path


def save(var=None, path=None):
    # type: (float) -> None
    """
    Function that saves given variable.
    :param var: Variable, default None
    :return: Saves the python object as pickle file
    """
    if var is None:
        raise AttributeError("No variable specified!")
    if path is None:
        file_path = dialog('-s')
    else:
        file_path = path
    with open(file_path, 'wb') as output:  # Overwrites any existing file.
        ip.pickle.dump(var, output)
    output.close()
    return


def load(path=None):
    # type: () -> var
    """
    Function that loads given pickle file.
    :return: Loads the pickle file as python object
    """
    if path is None:
        file_path = dialog()
    else:
        file_path = path
    with open(file_path, 'rb') as input_file:  # Overwrites any existing file.
        var = ip.pickle.load(input_file)
    input_file.close()
    return var
