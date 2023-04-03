import json
import os


def get_data_from_json(json_filepath):
    """get the json structure in a json file

    Parameters
    ----------
    json_filepath : str
        filepath of the json file

    Returns
    -------
    dict
        json structure store in the file transform as a dictionnary
        or dict() if the file do not exist
    """
    if os.path.isfile(json_filepath):
        with open(json_filepath) as json_file:
            data = json.load(json_file)
        return data
    else:
        return dict()
