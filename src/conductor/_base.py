import os

def get_conductor_path():
    return os.environ["CONDUCTOR_PATH"]


class Configuration(object):
    _name = "configuration"

    def __repr__(self):
        return Configuration._name

    def __init__(self, _dict):
        self.name = _dict["name"]
        self.c = _dict["configuration"]

        if ":" in _dict["entity"]:
            self.entity = _dict["entity"].split(":")
        else:
            self.entity = [_dict["entity"]]

    def to_dict(self):
        return {
            "name": self.name,
            "configuration": self.c,
            "entity": ":".join(self.entity) if len(self.entity) > 1 else self.entity[0]
        }
        
    
class Specification(object):
    _name = "specification"

    def __repr__(self):
        return Specification._name

    def __init__(self, name, _type):
        self.name = name
        self._type = _type

class InputOutputSpecification(Specification):
    _name = "input_output_specification"

    def __repr__(self):
        return Specification.__repr__(self) + ":" + InputOutputSpecification._name

    def __init__(self, _dict):
        Specification.__init__(self, _dict["name"], _dict["type"])

class Configurable(object):
    _name = "configurable"

    def __repr__(self):
        return Configurable._name

    def __init__(self, _type, _subtypes, configurations, default_config):
        self._type = _type
        self._subtypes = _subtypes

        if configurations is not None:
            is_simple = False
            if isinstance(configurations, list):
                confs = configurations
            elif isinstance(configurations, dict):
                confs = list(configurations.values())
                if len(confs) > 0:
                    if not isinstance(confs[0], Configuration):
                        is_simple = True
            elif isinstance(configurations, Configuration):
                confs = [configurations]

            tmp_conf = None
            if not is_simple:
                if "" in _subtypes or None in _subtypes:
                    comb = [_type]
                else:
                    comb = [_type] + _subtypes
                for c in confs:
                    if len(c.entity) > 1:
                        good = True
                        for k, ce in enumerate(c.entity):
                            if comb[k] != ce:
                                good = False
                                break
                        if good:
                            tmp_conf = c.c
                            break
                    else:
                        if c.entity[0] == self._type:
                            tmp_conf = c.c
                            break
            else:
                tmp_conf = configurations
                
            assert isinstance(default_config, dict)
            self.config = {}
            for cdk, cdv in default_config.items():
                if cdk in tmp_conf:
                    self.config[cdk] = tmp_conf[cdk]
                else:
                    self.config[cdk] = cdv
        else:
            assert isinstance(default_config, dict)
            self.config = default_config

    @classmethod
    def merge_configs(cls, first, second, override_first=False):
        common = {}
        for kf, vf in first.items():
            if kf in second and override_first:
                common[kf] = second[kf]
            else:
                common[kf] = vf
        for ks, vs in second.items():
            if ks not in common:
                common[ks] = vs
        return common