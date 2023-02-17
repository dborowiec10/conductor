import numpy as np

from conductor.component.method.template import TemplateMethod
import logging
logger = logging.getLogger("conductor.component.method.random_index")

# Simple, random index choice method for tuning - AutoTVM
class RandomIndexMethod(TemplateMethod):
    _name = "random_index"
    
    def __repr__(self):
        return TemplateMethod.__repr__(self) + ":" + RandomIndexMethod._name

    def __init__(self, orch_scheduler, configs=None, profilers_specs=[]):
        TemplateMethod.__init__(
            self, 
            ["standalone", "random_index"], 
            orch_scheduler,
            configs=configs, 
            child_default_configs={}, 
            profilers_specs=profilers_specs
        )
        self.range_len = None
        self.rand_state = {}
        self.visited = []
        self.rand_max = None
        self.index_offset = 0
        self.counter = 0

    def update(self, inputs, results):
        pass

    def has_next(self):
        return self.counter < self.range_len

    def next_batch(self, count):
        ret = []
        for _ in range(count):
            if self.rand_max == 0:
                break
            index_ = np.random.randint(self.rand_max)
            self.rand_max -= 1
            index = self.rand_state.get(index_, index_) + self.index_offset
            ret.append(self.task.config_space.get(index))
            self.visited.append(index)
            self.rand_state[index_] = self.rand_state.get(self.rand_max, self.rand_max)
            self.rand_state.pop(self.rand_max, None)
            self.counter += 1
        return ret

    def load(self, task, configurer):
        TemplateMethod.load(self, task, configurer)
        self.range_len = len(self.task.config_space)
        self.rand_max = self.range_len

    def unload(self):
        TemplateMethod.unload(self)