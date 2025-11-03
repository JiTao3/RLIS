import json
import logging

class ConfigurationParser(object):
    def __init__(self, configuration_file):
        self.configuration_file = configuration_file
        with open(configuration_file, "r") as f:
            self.config = json.load(f)
        self.pretty_print()
    
    def pretty_print(self):
        logging.info(json.dumps(self.config, indent=4))

    def get_config(self):
        return self.config