from conductor._base import Configurable
from conductor.component.cost_model._base import CostModel
import logging
logger = logging.getLogger("conductor.component.cost_model.sketch")


class SketchCostModel(CostModel):
    _name = "sketch"
    
    def __repr__(self):
        return CostModel.__repr__(self) + ":" + SketchCostModel._name

    def __init__(self, name, configs=None, child_default_configs={}):
        CostModel.__init__(self, "sketch", name, configs=configs, child_default_configs=Configurable.merge_configs(
            {}, child_default_configs, override_first=True
        ))
