import configparser
from distutils.util import strtobool


class Parser:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.loaded = False

    def load(self, path):
        self.config.read(path)
        self.loaded = True

    def get_raw(self):
        return {s:dict(self.config.items(s)) for s in self.config.sections()}

    def get(self, section, key, dtype):
        if self.loaded:
            if dtype != bool or key == 'no_cuda': # TODO Workaround, permanent fix required
                return dtype(self.config[section][key])
            elif dtype == bool:
                return bool(strtobool(self.config[section][key]))

        else:
            raise Exception('No config loaded')


