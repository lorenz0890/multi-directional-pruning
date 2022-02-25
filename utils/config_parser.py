import configparser
from distutils.util import strtobool


class Parser:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.config_global = configparser.ConfigParser()
        self.loaded = False

    def __config_global_override(self):
        print(self.config_global, flush=True)
        for section in self.config_global:
            if section in self.config:
                for key in self.config_global[section]:
                    if key in self.config[section]:
                        self.config[section][key] = self.config_global[section][key]
                        print(self.config[section][key])
                        print('Override: config[{}][{}] = {}'.format(
                            section, key, self.config_global[section][key]), flush=True)

    def load(self, path):
        self.config_global.read('configs/global_override.ini')
        self.config.read(path)
        self.__config_global_override()
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


