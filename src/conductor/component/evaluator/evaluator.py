from conductor.component.evaluator.adaptive import AdaptiveEvaluator
from conductor.component.evaluator.default import DefaultEvaluator
from conductor.component.evaluator.nvtx import NVTXEvaluator
from conductor.component.evaluator.activity import ActivityEvaluator
from conductor.component.evaluator.cupti import CUPTIEvaluator

evaluators = {
    "default": DefaultEvaluator,
    "adaptive": AdaptiveEvaluator,
    "activity": ActivityEvaluator,
    "nvtx": NVTXEvaluator,
    "cupti": CUPTIEvaluator
}