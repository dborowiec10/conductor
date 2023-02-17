import numpy as np

from conductor.component.method.template import TemplateMethod
from conductor.component.filter.default import DefaultFilter
from conductor.utils import array_mean, point2knob, knob2point
import logging

logger = logging.getLogger("conductor.component.method.composite_template")


class CompositeTemplateMethod(TemplateMethod):
    _name = "composite"
    
    def __repr__(self):
        return TemplateMethod.__repr__(self) + ":" + CompositeTemplateMethod._name

    def __init__(self, orch_scheduler, cost_model, optimizer, sampler, _filter, configs=None, profilers_specs=[]):
        TemplateMethod.__init__(
            self, 
            ["composite"], 
            orch_scheduler,
            configs=configs, 
            child_default_configs={
                "plan_size": 64
            }, 
            profilers_specs=profilers_specs
        )
        self.plan_size = self.config["plan_size"]
        self.cost_model_cl = cost_model
        self.optimizer_cl = optimizer
        self.sampler_cl = sampler
        self.filter_cl = _filter
        self.cost_model = None
        self.optimizer = None
        self.sampler = None
        self._filter = None
        self.trials = []
        self.trial_pt = 0
        self.visited = set()
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0
        self.balanced_epsilon = 0.05
        self.target = None
        self.space = None
        self.space_len = None
        self.dims = None
        self.configs = configs

    def load(self, task, configurer):
        TemplateMethod.load(self, task, configurer)
        self.target = self.task.target
        self.space = self.task.config_space
        self.space_len = len(self.space)
        self.dims = [len(x) for x in self.space.space_map.values()]

        if self.cost_model_cl:
            self.cost_model = self.cost_model_cl(self.task, configs=self.configs)
        if self.optimizer_cl:
            self.optimizer = self.optimizer_cl(self.task, configs=self.configs)
        if self.sampler_cl:
            self.sampler = self.sampler_cl(self.plan_size, configs=self.configs)
        if self.filter_cl:
            self._filter = self.filter_cl(configs=self.configs)
        if not self._filter:
            self._filter = DefaultFilter(configs=self.configs)
        self.balanced_epsilon = self._filter.balanced_epsilon

    def unload(self):
        self.cost_model._close_pool()
        TemplateMethod.unload(self)

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                self.xs.append(index)
                cost = array_mean(res.costs)
                flops = inp.task.flop / cost
                self.flops_max = max(self.flops_max, flops)
                self.ys.append(flops)
            else:
                self.xs.append(index)
                self.ys.append(0.0)

        if self.sampler:
            # if len(visited) >= sampler.next_update a.k.a. plan size
            condition = len(self.visited) >= self.sampler.next_update
        else:
            # if len(xs) >= self.plan_size * (self.train_ct + 1)
            # i.e. we have to have at least self.plan_size * 1 to start model fitting and sampling
            condition = len(self.xs) >= self.plan_size * (self.train_ct + 1)

        if condition and self.flops_max > 1e-6:
            self.cost_model.fit(self.xs, self.ys, self.plan_size)
            maximums, eps = self._filter.filter(self.cost_model, self.optimizer, self.plan_size, self.visited, self.dims, self.space, self.best_flops)
            if self.sampler:
                samples = [point2knob(config, self.dims) for config in maximums]
                reduced_samples = self.sampler.sample(samples, self.dims)
                maximums = [knob2point(sample, self.dims) for sample in reduced_samples]
                self.sampler.next_update += len(maximums)

            self.balanced_epsilon = eps
            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1

    def has_next(self):
        return len(self.visited) < len(self.space)

    def next_batch(self, count):
        ret = []
        counter = 0
        while counter < count:
            if len(self.visited) >= self.space_len:
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.trials) - int(self.balanced_epsilon * self.plan_size):
                index = np.random.randint(self.space_len)
                while index in self.visited:
                    index = np.random.randint(self.space_len)
            ret.append(self.space.get(index))
            self.visited.add(index)
            counter += 1
        return ret

    def reset(self):
        self.best_config = None
        self.best_flops = 0
        self.best_measure_pair = None

    def load_history(self, data_set):
        raise NotImplementedError()