import os


def get_writer():
    _, writer = os.pipe()
    return os.fdopen(writer, 'w')