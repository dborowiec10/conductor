import numpy as np
from conductor.component.method.template import TemplateMethod
from conductor.utils import array_mean, knob2point, point2knob
import logging
logger = logging.getLogger("conductor.component.method.genetic")

class GeneticAlgorithmMethod(TemplateMethod):
    _name = "genetic"
    
    def __repr__(self):
        return TemplateMethod.__repr__(self) + ":" + GeneticAlgorithmMethod._name

    def __init__(self, orch_scheduler, configs=None, profilers_specs=[]):
        TemplateMethod.__init__(
            self, 
            ["standalone", "genetic"], 
            orch_scheduler, 
            configs=configs, 
            child_default_configs={
                "pop_size": 100,
                "elite_num": 3,
                "mutation_prob": 0.1  
            }, 
            profilers_specs=profilers_specs
        )

        assert self.config["elite_num"] <= self.config["pop_size"], "The number of elites must be less than population size"
        self.pop_size = self.config["pop_size"]
        self.elite_num = self.config["elite_num"]
        self.mutation_prob = self.config["mutation_prob"]
        self.visited = set([])
        self.genes = []
        self.scores = []
        self.elites = []
        self.elite_scores = []
        self.trial_pt = 0
        self.dim_keys = []
        self.dims = []
        self.space = None

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            if res.error_no == 0:
                y = inp.task.flop / array_mean(res.costs)
                self.scores.append(y)
            else:
                self.scores.append(0.0)

        if len(self.scores) >= len(self.genes) and len(self.visited) < len(self.space):
            genes = self.genes + self.elites
            scores = np.array(self.scores[: len(self.genes)] + self.elite_scores)

            # reserve elite
            self.elites, self.elite_scores = [], []
            elite_indexes = np.argpartition(
                scores, -self.elite_num)[-self.elite_num:]
            for ind in elite_indexes:
                self.elites.append(genes[ind])
                self.elite_scores.append(scores[ind])

            # cross over
            indices = np.arange(len(genes))
            scores += 1e-8
            scores /= np.max(scores)
            probs = scores / np.sum(scores)
            tmp_genes = []
            for _ in range(self.pop_size):
                p1, p2 = np.random.choice(indices, size=2, replace=False, p=probs)
                p1, p2 = genes[p1], genes[p2]
                point = np.random.randint(len(self.dims))
                tmp_gene = p1[:point] + p2[point:]
                tmp_genes.append(tmp_gene)

            # mutation
            next_genes = []
            for tmp_gene in tmp_genes:
                for j, dim in enumerate(self.dims):
                    if np.random.random() < self.mutation_prob:
                        tmp_gene[j] = np.random.randint(dim)

                if len(self.visited) < len(self.space):
                    while knob2point(tmp_gene, self.dims) in self.visited:
                        j = np.random.randint(len(self.dims))
                        tmp_gene[j] = np.random.randint(
                            self.dims[j]  # pylint: disable=invalid-sequence-index
                        )
                    next_genes.append(tmp_gene)
                    self.visited.add(knob2point(tmp_gene, self.dims))
                else:
                    break

            self.genes = next_genes
            self.trial_pt = 0
            self.scores = []

    def has_next(self):
        return len(self.visited) - (len(self.genes) - self.trial_pt) < len(self.space)

    def next_batch(self, count):
        ret = []
        for _ in range(count):
            gene = self.genes[self.trial_pt % self.pop_size]
            self.trial_pt += 1
            ret.append(self.space.get(knob2point(gene, self.dims)))
        return ret

    def load(self, task, configurer):
        TemplateMethod.load(self, task, configurer)
        self.space = self.task.config_space
        for k, v in self.space.space_map.items():
            self.dim_keys.append(k)
            self.dims.append(len(v))
        self.pop_size = min(self.pop_size, len(self.space))
        self.elite_num = min(self.pop_size, self.elite_num)
        for _ in range(self.pop_size):
            tmp_gene = point2knob(
                np.random.randint(len(self.space)), self.dims)
            while knob2point(tmp_gene, self.dims) in self.visited:
                tmp_gene = point2knob(
                    np.random.randint(len(self.space)), self.dims)

            self.genes.append(tmp_gene)
            self.visited.add(knob2point(tmp_gene, self.dims))

    def unload(self):
        TemplateMethod.unload(self)
