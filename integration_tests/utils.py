from os.path import abspath, dirname


def get_project_root() -> str:
    return dirname(dirname(abspath(__file__)))
