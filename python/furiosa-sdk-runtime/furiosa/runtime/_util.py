"""Utility module"""


def list_to_dict(items: list) -> dict:
    """Transform a list to a dict with keys derived from indexes"""
    dict_values = {}
    for idx, value in enumerate(items):
        dict_values[idx] = value

    return dict_values
