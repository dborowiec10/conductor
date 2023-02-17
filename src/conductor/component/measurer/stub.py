from tvm.auto_scheduler.measure import ProgramMeasurer, LocalRPCMeasureContext, LocalBuilder
import tvm
import logging
logger = logging.getLogger("conductor.component.measurer.stub")

class StubMeasurer():
    _name = "stub"
    
    # def __repr__(self):
    #     return Measurer.__repr__(self) + ":" + StubMeasurer._name

    def __init__(self):
        # Measurer.__init__(self, "stub", builder, runner, configs=configs, child_default_configs=Configurable.merge_configs({
        #     "target": "cuda",
        #     "target_host": "llvm",
        #     "devices": [],
        #     "nvidia_mps_devices": []
        # }, {}, override_first=True))
        measure_ctx = LocalRPCMeasureContext(min_repeat_ms=100)
        self.program_measurer = ProgramMeasurer(LocalBuilder(), measure_ctx.runner, [], 0)
        tvm._ffi.register_func("auto_scheduler.measurer.measure", f=self.measure_wrapper, override=True)
        self.proposed = []


    def measure_wrapper(self, task, configs):
        _, mrs, _ = self.measure(task, configs, options=None)
        return mrs

    def measure(self, task, configs, options=None):
        self.proposed = configs

        return ([], [], len(configs))