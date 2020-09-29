"""
Reference:
https://github.com/perslev/MultiPlanarUNet/blob/master/mpunet/utils/utils.py

"""
import re
import os
import numpy as np
import glob
import contextlib


def create_folders(folders, create_deep=False):
    """ Make a new directory or directory.

    Args:
        folders (str or [str]): path
        create_deep (bool): (Default: False)
    """
    def safe_make(path, make_func):
        try:
            make_func(path)
        except FileExistsError:
            # If running many jobs in parallel this may occur
            pass
    make_func = os.mkdir if not create_deep else os.makedirs
    if isinstance(folders, str):
        if not os.path.exists(folders):
            safe_make(folders, make_func)
    else:
        folders = list(folders)
        for f in folders:
            if f is None:
                continue
            if not os.path.exists(f):
                safe_make(f, make_func)
