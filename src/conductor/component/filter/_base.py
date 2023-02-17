from conductor._base import Configurable

import logging
logger = logging.getLogger("conductor.component.filter")

class Filter(Configurable):
    _name = "filter"
    
    def __repr__(self):
        return Filter._name

    def __init__(self, name, configs=None, child_default_configs={}):
        Configurable.__init__(
            self,
            "filter",
            [name],
            configs,
            Configurable.merge_configs({}, child_default_configs, override_first=True)
        )
        self.name = name