
from conductor.component.filter.template import TemplateFilter

import logging
logger = logging.getLogger("conductor.component.filter.default")

class DefaultFilter(TemplateFilter):
    _name = "default"
    
    def __repr__(self):
        return TemplateFilter.__repr__(self) + ":" + DefaultFilter._name

    def __init__(self, configs=None):
        TemplateFilter.__init__(self, "default", configs=configs, child_default_configs={
            "balanced_epsilon": 0.05
        })
        self.balanced_epsilon = self.config["balanced_epsilon"]

    def filter(self, cost_model, optimizer, plan_size, visited, dimensions, space, best_flops):
        return optimizer.find_maximums(cost_model, plan_size, visited), self.balanced_epsilon
