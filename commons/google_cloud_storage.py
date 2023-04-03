import subprocess

"""
This module define helpers to upload and dowload files from google cloud storage using gsutil tool

Notes
-----

"""

def build_gs_path(folders):
    """build a path string for google cloud storage path that is understand by gsutil

    Parameters
    ----------
    folders : list[str]
        list of folders

    Returns
    -------
    str
        path for gsutil tool
    """
    return "/".join(folders)


def download_blob(bucket_name, gs_path, destination_path):
    """Download a dataset from google cloud storage using gsutil

    Parameters
    ----------
    bucket_name : str
        name of the bucket
    gs_path : str
        name of the path where the files are
    destination_path : str
        path to the destination

    Notes
    -----
    destination_path can be a string build with os.path.join

    gs_path is a string where folders are separated with '/'
    ex: "fo1/fo2" copy the fo2 folder itself
    "fo1/fo2/*" copy the content of the fo2 folder
    """
    gsutil_str = f"gsutil  -m cp -r gs://{bucket_name}/{gs_path} {destination_path}"
    subprocess.call(gsutil_str, shell=True)


def upload_blob(source_path, bucket_name, gs_path):
    """upload the content of a folder (not the folder itself) to google cloud storage

    Parameters
    ----------
    source_path : str
        path to the folder that contains the data
    bucket_name : str
        name of the bucket
    gs_path : str
        path where to put the files

    Notes
    -----
    source_path can be a string build with os.path.join
    ex: "fo1/fo2" copy the fo2 folder itself
        "fo1/fo2/*" copy the content of the fo2 folder

    gs_path is a string where folders are separated with '/'
    """
    gsutil_str = f"gsutil  -m cp -r {source_path} gs://{bucket_name}/{gs_path}"
    subprocess.call(gsutil_str, shell=True)



