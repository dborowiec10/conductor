from conductor._base import Configurable
from conductor.component.filter._base import Filter

class TemplateFilter(Filter):
    _name = "template"
    
    def __repr__(self):
        return Filter.__repr__(self) + ":" + TemplateFilter._name

    def __init__(self, name, configs=None, child_default_configs={}):
        Filter.__init__(
            self, 
            name, 
            configs=configs, 
            child_default_configs=Configurable.merge_configs({}, child_default_configs, override_first=True)
        )

    def filter(self, cost_model, optimizer, plan_size, visited, dimensions, space, best_flops):
        raise NotImplementedError()
