from conductor._base import Configurable

import logging
logger = logging.getLogger("conductor.component.cost_model")


class CostModel(Configurable):
    _name = "cost_model"
    
    def __repr__(self):
        return CostModel._name

    def __init__(self, scheduling, name, configs=None, child_default_configs={}):
        Configurable.__init__(
            self,
            "cost_model",
            [name],
            configs,
            Configurable.merge_configs({}, child_default_configs, override_first=True)
        )
        self.scheduling = scheduling
        self.name = name
        