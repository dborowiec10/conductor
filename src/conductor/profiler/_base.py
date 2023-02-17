
from collections import OrderedDict
from conductor.profiler.profilers import profilers

class ProfilerSpecification(object):
    _name = "profiler_specification"

    def __repr__(self):
        return ProfilerSpecification._name

    def _validate(self, _dict, configurations):
        assert isinstance(_dict, dict)
        assert "type" in _dict
        assert _dict["type"] in profilers.keys()
        assert "scope" in _dict
        assert "configuration" in _dict
        if configurations is not None:
            assert _dict["configuration"] in configurations
        assert "collection" in _dict

    def __init__(self, _dict, _configurations, override_config=None):
        self._validate(_dict, _configurations)
        self._type = _dict["type"]
        self.scope = _dict["scope"]
        if override_config is not None:
            self.configuration = override_config
        else:
            self.configuration = _configurations[_dict["configuration"]]
        self.collection = _dict["collection"]

    def from_spec(self):
        klass = profilers[self._type]
        return klass(self.scope, self.collection, configs=self.configuration)

class Profilable(object):
    _name = "profilable"
    
    def __repr__(self):
        return Profilable._name

    def __init__(self, scope, acceptable, specs=[]):
        self.scope = scope
        self.acceptable = acceptable
        self.specs = specs
        self.profilers = OrderedDict()
        for s in specs:
            if s.scope == self.scope and s._type in self.acceptable:
                self.profilers[s._type] = s.from_spec()
        self.checkpoint_data = {}

    def profiling_checkpoint(self, checkpoint):
        raise NotImplementedError()

    def begin_profilers(self, exclude=[]):
        for p in list(self.profilers.values()):
            if p._type not in exclude:
                p.begin()

    def end_profilers(self, exclude=[]):
        for p in list(self.profilers.values()):
            if p._type not in exclude:
                p.end()

    def start_profilers(self, context={}, exclude=[]):
        keymap = {}
        for p in list(self.profilers.values()):
            if p._type not in exclude:
                k = p.start(context=context)
                keymap[p._type] = k
        return keymap

    def stop_profilers(self, keymap, exclude=[]):
        for p in list(self.profilers.values()):
            if p._type not in exclude:
                p.stop(keymap[p._type])

    def persist_profilers(self, exclude=[], clear_all=True):
        for k, p in enumerate(list(self.profilers.values())):
            if p._type not in exclude:
                p.persist(clear_all=clear_all)

    def get_profiler(self, _type):
        return self.profilers[_type]

    def profiler_exists(self, _type):
        return _type in self.profilers
    
    def is_profiler_acceptable(self, _type):
        return True if _type in self.acceptable else False
    
    def append_context_profilers(self, keymap, ctx, exclude=[]):
        for p in self.profilers.values():
            if p._type not in exclude:
                p.append_context(keymap[p._type], ctx)