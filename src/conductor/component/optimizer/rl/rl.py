
from conductor.component.optimizer.rl.rl_model_optimizer import ReinforcementLearningOptimizer as cham_rl
from conductor.component.optimizer._base import Optimizer

import logging
logger = logging.getLogger("conductor.component.optimizer.reinforcement_learning")

class ReinforcementLearningOptimizer(Optimizer, cham_rl):
    _name = "reinforcement_learning"
    
    def __repr__(self):
        return Optimizer.__repr__(self) + ":" + ReinforcementLearningOptimizer._name

    def __init__(self, task, configs=None):
        Optimizer.__init__(self, "reinforcement_learning", configs=configs, child_default_configs={
            "temp": (1, 0),
            "early_stop": 50,
            "persistent": True,
            "log_interval": 50
        })
        cham_rl.__init__(
            self,
            task,
            temp=self.config["temp"],
            early_stop=self.config["early_stop"],
            persistent=self.config["persistent"],
            log_interval=self.config["log_interval"]
        )
