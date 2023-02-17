
from conductor.component.cost_model.adatune.adatune import RFModel as adatune_rf
from conductor.component.cost_model.template import TemplateCostModel
import logging
logger = logging.getLogger("conductor.component.cost_model.adatune")


class AdatuneRFItervarCostModel(TemplateCostModel, adatune_rf):
    _name = "adatune_rf_itervar"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + AdatuneRFItervarCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "adatune_rf_itervar", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "itervar"
        adatune_rf.__init__(
            self, 
            task, 
            feature_type=feature_type,
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )


class AdatuneRFKnobCostModel(TemplateCostModel, adatune_rf):
    _name = "adatune_rf_knob"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + AdatuneRFKnobCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "adatune_rf_knob", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "knob"
        adatune_rf.__init__(
            self, 
            task, 
            feature_type=feature_type,
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )

class AdatuneRFSimpleknobCostModel(TemplateCostModel, adatune_rf):
    _name = "adatune_rf_simpleknob"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + AdatuneRFSimpleknobCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "adatune_rf_simpleknob", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "simpleknob"
        adatune_rf.__init__(
            self, 
            task, 
            feature_type=feature_type,
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )

class AdatuneRFCurveCostModel(TemplateCostModel, adatune_rf):
    _name = "adatune_rf_curve"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + AdatuneRFCurveCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "adatune_rf_curve", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "curve"
        adatune_rf.__init__(
            self, 
            task, 
            feature_type=feature_type,
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )