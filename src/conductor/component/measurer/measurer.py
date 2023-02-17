from conductor.component.measurer.default import DefaultMeasurer
from conductor.component.measurer.doppler_multi_estimator import DopplerMultiEstimatorMeasurer
from conductor.component.measurer.doppler_multi_estimator_no_update import DopplerMultiEstimatorNoUpdateMeasurer
from conductor.component.measurer.doppler_estimator import DopplerEstimatorMeasurer
from conductor.component.measurer.doppler_estimator_no_update import DopplerEstimatorNoUpdateMeasurer
from conductor.component.measurer.doppler_rl import DopplerRLMeasurer
from conductor.component.measurer.doppler_bic import DopplerBicMeasurer
from conductor.component.measurer.doppler_learn import DopplerLearn
from conductor.component.measurer.stub import StubMeasurer
measurers = {
    "default": DefaultMeasurer,
    "doppler_multi_estimator": DopplerMultiEstimatorMeasurer,
    "doppler_multi_estimator_no_update": DopplerMultiEstimatorNoUpdateMeasurer,
    "doppler_estimator": DopplerEstimatorMeasurer,
    "doppler_estimator_no_update": DopplerEstimatorNoUpdateMeasurer,
    "doppler_rl": DopplerRLMeasurer,
    "doppler_bic": DopplerBicMeasurer,
    "doppler_learn": DopplerLearn,
    "stub": StubMeasurer
}