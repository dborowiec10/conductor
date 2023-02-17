from conductor.component.method.random_index import RandomIndexMethod
from conductor.component.method.grid_index import GridIndexMethod
from conductor.component.method.genetic import GeneticAlgorithmMethod
from conductor.component.method.composite_sketch import CompositeSketchMethod
from conductor.component.method.composite_template import CompositeTemplateMethod
from conductor.component.method.flex.flex_random import FlexRandomMethod
from conductor.component.method.flex.flex_q import FlexQMethod
from conductor.component.method.flex.flex_p import FlexPMethod

methods = {
    "template": {
        "standalone": {
            "genetic": GeneticAlgorithmMethod,
            "random_index": RandomIndexMethod,
            "grid_index": GridIndexMethod
        },
        "composite": CompositeTemplateMethod
    },
    "sketch": {
        "standalone": {},
        "composite": CompositeSketchMethod
    },
    "flex": {
        "composite": None,
        "standalone": {
            "qflex": FlexQMethod,
            "pflex": FlexPMethod,
            "rndflex": FlexRandomMethod
        }
    }
}
