
from conductor.profiler.profiler import Profiler
from conductor._base import Configurable
from conductor.experiment import Experiment
from collections import OrderedDict

import time

import logging
logger = logging.getLogger("conductor.profiler.time_profiler")

class TimeProfiler(Profiler):
    _name = "time_profiler"

    def __repr__(self):
        return Profiler.__repr__(self) + ":" + TimeProfiler._name

    def __init__(self, _scope, _collection, configs=None, child_default_configs={}):
        Profiler.__init__(self, "time_profiler", "", _scope, _collection, configs=configs, child_default_configs=Configurable.merge_configs({}, child_default_configs, override_first=True))

    def begin(self):
        Profiler.begin(self)

    def end(self):
        self.results.clear()

    def append_context(self, key, ctx):
        self.results[key].update(ctx)

    def start(self, context={}):
        key = Profiler.start(self)
        self.results[key] = OrderedDict()
        self.results[key].update(context)
        self.results[key]["start"] = time.time()
        self.results[key]["end"] = None
        self.results[key]["duration"] = None
        return key

    def stop(self, key):
        self.results[key]["end"] = time.time()
        self.results[key]["duration"] = self.results[key]["end"] - self.results[key]["start"]

    def persist(self, clear_all=True):
        Experiment.current.insert_many(self.collection, list(self.results.values()))
        if clear_all:        
            self.results.clear()