from conductor._base import Configurable
from conductor.component.cost_model._base import CostModel
import logging
logger = logging.getLogger("conductor.component.cost_model.template")

class TemplateCostModel(CostModel):
    _name = "template"
    
    def __repr__(self):
        return CostModel.__repr__(self) + ":" + TemplateCostModel._name

    def __init__(self, name, configs=None, child_default_configs={}):
        CostModel.__init__(self, "template", name, configs=configs, child_default_configs=Configurable.merge_configs(
            {}, child_default_configs, override_first=True
        ))
