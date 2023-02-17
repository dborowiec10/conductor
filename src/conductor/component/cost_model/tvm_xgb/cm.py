from conductor.component.cost_model.template import TemplateCostModel
from tvm.autotvm.tuner.xgboost_cost_model import XGBoostCostModel as xgb_cm

import logging
logger = logging.getLogger("conductor.component.cost_model.template.tvm_xgb")


class TVMXGBItervarRankCostModel(TemplateCostModel, xgb_cm):
    _name = "tvm_xgb_itervar_rank"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + TVMXGBItervarRankCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "tvm_xgb_itervar_rank", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "itervar"
        loss_type = "rank"
        xgb_cm.__init__(
            self, 
            task, 
            feature_type=feature_type, 
            loss_type=loss_type, 
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )

class TVMXGBItervarRegCostModel(TemplateCostModel, xgb_cm):
    _name = "tvm_xgb_itervar_reg"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + TVMXGBItervarRegCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "tvm_xgb_itervar_reg", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "itervar"
        loss_type = "reg"
        xgb_cm.__init__(
            self, 
            task, 
            feature_type=feature_type, 
            loss_type=loss_type, 
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )

class TVMXGBKnobRankCostModel(TemplateCostModel, xgb_cm):
    _name = "tvm_xgb_knob_rank"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + TVMXGBKnobRankCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "tvm_xgb_knob_rank", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "knob"
        loss_type = "rank"
        xgb_cm.__init__(
            self, 
            task, 
            feature_type=feature_type, 
            loss_type=loss_type, 
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )

class TVMXGBKnobRegCostModel(TemplateCostModel, xgb_cm):
    _name = "tvm_xgb_knob_reg"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + TVMXGBKnobRegCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "tvm_xgb_knob_reg", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "knob"
        loss_type = "reg"
        xgb_cm.__init__(
            self, 
            task, 
            feature_type=feature_type, 
            loss_type=loss_type, 
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )

class TVMXGBCurveRankCostModel(TemplateCostModel, xgb_cm):
    _name = "tvm_xgb_curve_rank"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + TVMXGBCurveRankCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "tvm_xgb_curve_rank", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "curve"
        loss_type = "rank"
        xgb_cm.__init__(
            self, 
            task, 
            feature_type=feature_type, 
            loss_type=loss_type, 
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )

class TVMXGBCurveRegCostModel(TemplateCostModel, xgb_cm):
    _name = "tvm_xgb_curve_reg"

    def __repr__(self):
        return TemplateCostModel.__repr__(self) + ":" + TVMXGBCurveRegCostModel._name

    def __init__(self, task, configs=None):
        TemplateCostModel.__init__(self, "tvm_xgb_curve_reg", configs=configs, child_default_configs={
            "num_threads": None,
            "log_interval": 25
        })
        feature_type = "curve"
        loss_type = "reg"
        xgb_cm.__init__(
            self, 
            task, 
            feature_type=feature_type, 
            loss_type=loss_type, 
            num_threads=self.config["num_threads"], 
            log_interval=self.config["log_interval"]
        )