"""
Download model automatically.
"""
import os
import tarfile
from os.path import exists, join
from typing import Optional

import requests


#  HUMAN_WGS_MODEL_URL = 'https://github.com/relastle/halcyon/releases/download/0.0.1/human-wgs.tar.gz'  # noqa
HUMAN_WGS_MODEL_URL = 'https://dl.dropboxusercontent.com/s/iohn6mfe31ppcfo/human-wgs.tar.gz?dl=0'  # noqa
MODEL_NAME = 'human-wgs'


def download() -> Optional[str]:
    """
    Download and extract it and return config path.
    """
    model_dir = os.getenv('HALCYON_MODEL_BASE_DIR', '/tmp/halcyon')
    config_path = join(model_dir, MODEL_NAME, 'config.json')
    if exists(config_path):
        return config_path
    resp = requests.get(HUMAN_WGS_MODEL_URL, stream=True)
    if resp.status_code != 200:
        return None
    os.makedirs(model_dir, exist_ok=True)
    tar_path = f'{model_dir}/{MODEL_NAME}.tar.gz'
    with open(tar_path, 'wb') as f:
        f.write(resp.raw.read())

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    return config_path


def main() -> None:
    download()
    return


if __name__ == '__main__':
    main()
