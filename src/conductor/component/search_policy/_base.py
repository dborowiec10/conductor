from conductor._base import Configurable

import logging
logger = logging.getLogger("conductor.component.search_policy")

class SearchPolicy(Configurable):
    _name = "search_policy"
    
    def __repr__(self):
        return SearchPolicy._name

    def __init__(self, name, configs=None, child_default_configs={}):
        Configurable.__init__(
            self,
            "search_policy",
            [name],
            configs,
            Configurable.merge_configs({}, child_default_configs, override_first=True)
        )
        self.name = name