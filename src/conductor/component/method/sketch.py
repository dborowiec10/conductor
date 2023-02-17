from conductor._base import Configurable
from conductor.component.method._base import Method

import logging
logger = logging.getLogger("conductor.component.method.sketch")

class SketchMethod(Method):
    _name = "sketch"

    def __repr__(self):
        return Method.__repr__(self) + ":" + SketchMethod._name

    def __init__(self, subtypes, orch_scheduler, configs=None, child_default_configs={}, profilers_specs=[]):
        Method.__init__(
            self, 
            ["sketch"] + subtypes,
            "sketch", 
            orch_scheduler,
            configs=configs,
            child_default_configs=Configurable.merge_configs({}, child_default_configs, override_first=True),
            profilers_specs=profilers_specs
        )

    def load(self, task, configurer):
        Method.load(self, task, configurer)

    def unload(self):
        Method.unload(self)