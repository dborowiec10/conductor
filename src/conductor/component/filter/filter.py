from conductor.component.filter.context_aware import ContextAwareFilter
from conductor.component.filter.diversity_aware import DiversityAwareFilter
from conductor.component.filter.default import DefaultFilter

filters = {
    "template": {
        "diversity_aware": DiversityAwareFilter,
        "context_aware": ContextAwareFilter,
        "default": DefaultFilter
    }
}


