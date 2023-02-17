from tvm.autotvm.tuner.sa_model_optimizer import SimulatedAnnealingOptimizer as tvm_sa
from conductor.component.optimizer._base import Optimizer

import logging
logger = logging.getLogger("conductor.component.optimizer.simulated_annealing")

class SimulatedAnnealingOptimizer(Optimizer, tvm_sa):
    _name = "simulated_annealing"
    
    def __repr__(self):
        return Optimizer.__repr__(self) + ":" + SimulatedAnnealingOptimizer._name

    def __init__(self, task, configs=None):
        Optimizer.__init__(self, "simulated_annealing", configs=configs, child_default_configs={
            "temp": (1, 0),
            "early_stop": 50,
            "persistent": True,
            "log_interval": 50
        })
        tvm_sa.__init__(
            self,
            task,
            temp=self.config["temp"],
            early_stop=self.config["early_stop"],
            persistent=self.config["persistent"],
            log_interval=self.config["log_interval"]
        )
