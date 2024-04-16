import os
import subprocess
import webbrowser
import ctypes
import logging
from collections import namedtuple as nt
from collections import deque as dq
from collections import Counter as ct
import numpy as np
from scipy import io
from random import random as r
import imageio
from tkinter import filedialog, Tk
import pickle
import time
import pandas as p


def import_tf():
    # type: () -> Tensorflow
    """
    Function that imports Tensorflow only if neccessary to do so.
    :return: Initialized Tensorflow
    """
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.autograph.set_verbosity(0)
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    return tf
