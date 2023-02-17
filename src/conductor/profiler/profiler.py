
from conductor._base import Configurable
import uuid
from collections import OrderedDict

class Profiler(Configurable):
    _name = "profiler"
    
    def __repr__(self):
        return Profiler._name

    def __init__(self, _type, _subtype, _scope, _collection, configs=None, child_default_configs={}):
        self._type = _type
        self._scope = _scope
        self.collection = _collection
        self.results = None
        Configurable.__init__(self, _type, [_subtype], configs, Configurable.merge_configs({}, child_default_configs, override_first=True))

    def append_context(self, key, ctx):
        raise NotImplementedError("abstract method, needs to be implemented")

    def begin(self):
        self.results = OrderedDict()

    def end(self):
        raise NotImplementedError("abstract method, needs to be implemented")

    def start(self):
        return str(uuid.uuid4())

    def stop(self, key):
        raise NotImplementedError("abstract method, needs to be implemented")
        
    def persist(self, clear_all=True, clear_checkpoint=True):
        # needs to handle clearing automatically
        raise NotImplementedError("abstract method, needs to be implemented")



    