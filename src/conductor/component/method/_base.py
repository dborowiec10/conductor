from conductor._base import Configurable
from conductor.component.method.flex.utils import Config
from conductor.profiler._base import Profilable

import logging
logger = logging.getLogger("conductor.component.method")

class Method(Configurable, Profilable):
    _name = "method"
    
    def __repr__(self):
        return Method._name

    def __init__(self, subtypes, scheduling_type, orch_scheduler, configs=None, child_default_configs={}, profilers_specs=[]):
        Configurable.__init__(
            self, 
            "method", 
            subtypes, 
            configs, 
            Configurable.merge_configs({}, child_default_configs, override_first=True)
        )
        Profilable.__init__(self, "method", ["time_profiler"], specs=profilers_specs)
        self.task = None
        self.task_type = scheduling_type
        self.orch_scheduler_cls = orch_scheduler
        self.orch_scheduler = None
        self.measurer = None

    # implements method checkpointing for profilable
    def profiling_checkpoint(self, checkpoint, ctx=None):
        if checkpoint == "load":
            self.begin_profilers()
            
        elif checkpoint == "unload":
            self.persist_profilers()
            self.end_profilers()

        elif checkpoint == "execute:start":
            self.checkpoint_data["execute:start"] = self.start_profilers()

        elif checkpoint == "execute:stop":
            self.stop_profilers(self.checkpoint_data["execute:start"])
            
        else:
            pass

    def set_measurer(self, measurer):
        self.measurer = measurer

    def load(self, task, configurer):
        self.profiling_checkpoint("load")

        self.task = task

        if self.orch_scheduler is None:
            self.orch_scheduler = self.orch_scheduler_cls()

        if self.measurer is not None:
            self.measurer.set_orch_scheduler(self.orch_scheduler)

        self.measurer.set_configurer(configurer)

    def unload(self):
        pass

    def execute(self, num_measurements):
        raise NotImplementedError()