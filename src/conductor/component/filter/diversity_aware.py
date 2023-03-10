import numpy as np

from conductor.utils import point2knob
from conductor.component.filter.template import TemplateFilter

import logging
logger = logging.getLogger("conductor.component.filter.diversity_aware")

class DiversityAwareFilter(TemplateFilter):
    _name = "diversity_aware"
    
    def __repr__(self):
        return TemplateFilter.__repr__(self) + ":" + DiversityAwareFilter._name

    def __init__(self, configs=None):
        TemplateFilter.__init__(self, "diversity_aware", configs=configs, child_default_configs={
            "ratio": None,
            "balanced_epsilon": 0.05
        })
        assert self.config["ratio"] is not None
        self.ratio = self.config["ratio"]
        self.balanced_epsilon = self.config["balanced_epsilon"]

    def filter(self, cost_model, optimizer, plan_size, visited, dimensions, space, best_flops):
        candidates = optimizer.find_maximums(
            cost_model, plan_size * self.ratio, visited)
        scores = cost_model.predict(candidates)
        knobs = [point2knob(x, dimensions) for x in candidates]
        index = self.submodular_pick(
            0 * scores, knobs, plan_size, knob_weight=1)
        return np.array(candidates)[index], self.balanced_epsilon

    def submodular_pick(self, scores, knobs, n_pick, knob_weight=1.0):
        """Run greedy optimization to pick points with regard to both score and diversity.
        DiversityScore = knob_weight * number of unique knobs in the selected set
        Obj = sum(scores[i] for i in pick) + DiversityScore
        Note that this objective function is a monotone submodular function.

        Parameters
        ----------
        scores: Array of float
            score of every points
        knobs: Array of Array of int
            feature vector (tunable knobs) of every points
        n_pick: int
            number of points to pick
        knob_weight: float
            weight of an unique knob feature
        """
        n = len(scores)
        assert n == len(knobs)
        n_knobs = len(knobs[0])

        knobs_set = [set() for _ in range(n_knobs)]

        ret = []
        remain = list(range(len(scores)))

        for _ in range(n_pick):
            max_x = -1
            max_delta = -1e9

            for x in remain:
                tmp_delta = scores[x]
                for i in range(n_knobs):
                    if knobs[x][i] not in knobs_set[i]:
                        tmp_delta += knob_weight

                if tmp_delta > max_delta:
                    max_delta, max_x = tmp_delta, x

            ret.append(max_x)
            remain.remove(max_x)
            for i in range(n_knobs):
                knobs_set[i].add(knobs[max_x][i])

        return ret
