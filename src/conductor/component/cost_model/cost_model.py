from conductor.component.cost_model.tvm_xgb.cm import TVMXGBCurveRankCostModel, \
    TVMXGBCurveRegCostModel, \
    TVMXGBItervarRankCostModel, \
    TVMXGBKnobRankCostModel, \
    TVMXGBItervarRegCostModel, \
    TVMXGBKnobRegCostModel

from conductor.component.cost_model.adatune.cm import AdatuneRFCurveCostModel, \
    AdatuneRFItervarCostModel, \
    AdatuneRFKnobCostModel, \
    AdatuneRFSimpleknobCostModel

from conductor.component.cost_model.ansor.cm import AnsorRandomCostMOdel, AnsorXGBCostModel

cost_models = {
    "template": {
        "tvm_xgb_itervar_rank": TVMXGBItervarRankCostModel,
        "tvm_xgb_itervar_reg": TVMXGBItervarRegCostModel,
        "tvm_xgb_knob_rank": TVMXGBKnobRankCostModel,
        "tvm_xgb_knob_reg": TVMXGBKnobRegCostModel,
        "tvm_xgb_curve_rank": TVMXGBCurveRankCostModel,
        "tvm_xgb_curve_reg": TVMXGBCurveRegCostModel,
        "adatune_rf_itervar": AdatuneRFItervarCostModel,
        "adatune_rf_knob": AdatuneRFKnobCostModel,
        "adatune_rf_simpleknob": AdatuneRFSimpleknobCostModel,
        "adatune_rf_curve": AdatuneRFCurveCostModel
    },
    "sketch": {
        "ansor_xgb": AnsorXGBCostModel,
        "ansor_rnd": AnsorRandomCostMOdel
    }
}
    
