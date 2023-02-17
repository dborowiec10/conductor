
from conductor.component.method.template import TemplateMethod
import logging

logger = logging.getLogger("conductor.component.method.grid_index")

class GridIndexMethod(TemplateMethod):
    _name = "grid_index"
    
    def __repr__(self):
        return TemplateMethod.__repr__(self) + ":" + GridIndexMethod._name

    def __init__(self, orch_scheduler, configs=None, profilers_specs=[]):
        TemplateMethod.__init__(
            self, 
            ["standalone", "grid_index"],
            orch_scheduler,
            configs=configs, 
            child_default_configs={}, 
            profilers_specs=profilers_specs
        )
        self.range_len = None
        self.index_offset = 0
        self.counter = 0

    def update(self, inputs, results):
        pass

    def has_next(self):
        return self.counter < self.range_len

    def next_batch(self, count):
        ret = []
        for _ in range(count):
            if self.counter >= self.range_len:
                break
            index = self.counter + self.index_offset
            ret.append(self.task.config_space.get(index))
            self.counter = self.counter + 1
        return ret

    def load(self, task, configurer):
        TemplateMethod.load(self, task, configurer)
        self.range_len = len(self.task.config_space)

    def unload(self):
        TemplateMethod.unload(self)