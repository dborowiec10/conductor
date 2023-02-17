from conductor.component.optimizer.rl.rl import ReinforcementLearningOptimizer
from conductor.component.optimizer.sa.sa import SimulatedAnnealingOptimizer

import logging
logger = logging.getLogger("conductor.component.optimizer")


config_map = {
    "SimulatedAnnealingOptimizer": ["template", "default"],
    "ReinforcementLearningOptimizer": ["template", "default"]
}

optimizers = {
    "template": {
        "simulated_annealing": SimulatedAnnealingOptimizer,
        "reinforcement_learning": ReinforcementLearningOptimizer
    }
}