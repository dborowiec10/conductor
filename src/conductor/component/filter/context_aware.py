import numpy as np

from conductor.utils import sample_ints
from conductor.component.filter.template import TemplateFilter

import logging
logger = logging.getLogger("conductor.component.filter.context_aware")

class ContextAwareFilter(TemplateFilter):
    _name = "context_aware"

    def __repr__(self):
        return TemplateFilter.__repr__(self) + ":" + ContextAwareFilter._name

    def __init__(self, configs=None):
        TemplateFilter.__init__(self, "context_aware", configs=configs, child_default_configs={
            "balanced_epsilon": 0.05
        })
        self.balanced_epsilon = self.config["balanced_epsilon"]

    def filter(self, cost_model, optimizer, plan_size, visited, dimensions, space, best_flops):
        maximums = optimizer.find_maximums(cost_model, plan_size, visited)
        samples = np.array(sample_ints(0, len(space), 20))
        _, mean_of_variance = cost_model._expected_imporvement(samples)
        self.balanced_epsilon = mean_of_variance/best_flops
        return maximums, self.balanced_epsilon