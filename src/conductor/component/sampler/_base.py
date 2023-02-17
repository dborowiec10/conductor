from conductor._base import Configurable

import logging
logger = logging.getLogger("conductor.component.sampler")


class Sampler(Configurable):
    _name = "sampler"
    
    def __repr__(self):
        return Sampler._name

    def __init__(self, name, configs=None, child_default_configs={}):
        Configurable.__init__(
            self,
            "sampler",
            [name],
            configs,
            Configurable.merge_configs({}, child_default_configs, override_first=True)
        )
        self.name = name
