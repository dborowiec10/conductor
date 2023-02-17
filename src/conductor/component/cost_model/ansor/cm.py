import os

from tvm.auto_scheduler.cost_model import XGBModel as ansor_xgb
from tvm.auto_scheduler.cost_model import RandomModel as ansor_rnd
from conductor.component.cost_model.sketch import SketchCostModel
import logging
logger = logging.getLogger("conductor.component.cost_model.sketch.ansor")

class AnsorXGBCostModel(SketchCostModel, ansor_xgb):
    _name = "ansor_xgb"
    
    def __repr__(self):
        return SketchCostModel.__repr__(self) + ":" + AnsorXGBCostModel._name

    def __init__(self, num_warmup_samples, configs=None):
        SketchCostModel.__init__(self, "ansor_xgb", configs=configs, child_default_configs={
            "model_file": None,
            "log_file": None,
            "adaptive_training": False,
            "seed": None,
            "verbose_eval": 25
        })
        ansor_xgb.__init__(
            self,
            verbose_eval=self.config["verbose_eval"],
            num_warmup_sample=num_warmup_samples, 
            seed=self.config["seed"],
            model_file=self.config["model_file"],
            adapative_training=self.config["adaptive_training"],
        )
        if self.config["model_file"] is not None and os.path.isfile(self.config["model_file"]):
            logger.info("TaskScheduler: Load pretrained model...")
            self.load(self.config["model_file"])
        elif self.config["log_file"] is not None and os.path.isfile(self.config["log_file"]):
            logger.info("TaskScheduler: Reload measured states and train the model...")
            self.update_from_file(self.config["log_file"])

class AnsorRandomCostMOdel(SketchCostModel, ansor_rnd):
    _name = "ansor_rnd"

    def __repr__(self):
        return SketchCostModel.__repr__(self) + ":" + AnsorRandomCostMOdel._name

    def __init__(self, num_warmup_samples, configs=None):
        SketchCostModel.__init__(self, "ansor_rnd", configs=configs, child_default_configs={})
        ansor_rnd.__init__(self)