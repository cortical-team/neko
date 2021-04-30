import os

import requests


def download_file(url, folder, filename):
    """Downloads a file from url.

    Args:
        url: The url of the file
        folder: The folder to save
        filename: The filename to save

    Returns:
        None
    """
    os.makedirs(folder, exist_ok=True)
    dest_file = os.path.join(folder, filename)
    if not os.path.exists(dest_file):
        # TODO: error handling
        resp = requests.get(url)
        with open(dest_file, 'wb+') as f:
            f.write(resp.content)
