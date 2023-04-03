"""
This module manipulate folders



Notes
-----

"""

import os
import shutil


def check_path(path_name):
    """check if a path exist and if not create the folders

    Parameters
    ----------
    path_name : str
        path to check

    Returns
    -------
    str
        same path
    """
    os.makedirs(path_name,exist_ok=True)
    return path_name


def del_create_path(path_name):
    """delete a path with all its folders/files
    and recreat it

    Parameters
    ----------
    path_name : str
        path to re-create

    Returns
    -------
    str
        same path
    """
    if os.path.exists(path_name):
        shutil.rmtree(path_name)
    os.makedirs(path_name)
    return path_name
