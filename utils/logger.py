import json
import os
from datetime import datetime

import torch


class Logger:
    def __init__(self, config, path = None):
        self.path = path
        self.logdict = {}
        self.config = config
        self.logdict['LOGDATA'] = {}
        self.active = True

        if self.path is None or self.path == 'Default':
            self.path = os.getcwd()+'/logfiles/'

    def toggle(self): #no logging for debugging
        self.active = not self.active

    def log(self, key, value):
        if self.active:
            if key not in self.logdict['LOGDATA']:
                self.logdict['LOGDATA'][key] = []
            self.logdict['LOGDATA'][key].append(value)

    def __make_name(self):
        name = ''
        #for section in self.config:
        for element in self.config['EXPERIMENT']:
            name = name + '_' + str(self.config['EXPERIMENT'][element])
        name = name + '_' + datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        name += '.json'
        return name
    
    def store(self):
        self.logdict['METADATA'] = self.config
        with open(self.path + self.__make_name(), 'w') as outfile:
            json.dump(self.logdict, outfile)