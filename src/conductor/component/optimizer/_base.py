from conductor._base import Configurable

import logging
logger = logging.getLogger("conductor.component.method.optimizer")

class Optimizer(Configurable):
    _name = "optimizer"
    
    def __repr__(self):
        return Optimizer._name

    def __init__(self, name, configs=None, child_default_configs={}):
        Configurable.__init__(
            self,
            "optimizer",
            [name],
            configs,
            Configurable.merge_configs({}, child_default_configs, override_first=True)
        )
        self.name = name