import configparser

class Parser:
    def __init__(self):
        self.config = configparser.ConfigParser()
        self.loaded = False

    def load(self, path):
        self.config.read(path)
        self.loaded = True

    def get(self, section, key, dtype):
        if self.loaded:
            return dtype(self.config[section][key])
        else:
            raise Exception('No config loaded')


