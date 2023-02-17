from abc import ABCMeta, abstractmethod
import os
import logging

logger_task = logging.getLogger("conductor.task")

class Task(metaclass=ABCMeta):
    _name = "task"

    def __repr__(self):
        return Task._name

    def __init__(
        self, 
        subtype, 
        spec,
        input_types, 
        output_types, 
        specification_types,
        task_results_path, 
        models_path, 
        tensor_programs_path
    ):
        self.name = spec.name
        self.spec = spec
        self.input_types = input_types
        self.output_types = output_types
        self.specification_types = specification_types
        self.subtype = subtype
        self.results_path = task_results_path
        self.models_path = models_path
        self.tensor_programs_path = tensor_programs_path
    
    def load(self):
        os.makedirs(self.results_path, exist_ok=True)

    def prepare_inputs(self, inputs):
        dct = {}
        for i in inputs:
            dct[i[0]] = i[1]
        return dct

    def prepapre_outputs(self, outputs):
        # return out...
        pass

    def find_specifications_by_type(self, _type):
        specs = []
        for s in list(self.spec.specifications.values()):
            if s._type == _type:
                specs.append(s)
        return specs

    def find_configuration_by_entity(self, entity):
        split_entity = entity.split(":")
        for c in list(self.spec.configurations.values()):
            if isinstance(c.entity, list):
                if len(c.entity) == len(split_entity):
                    bad = False
                    for i in range(len(c.entity)):
                        if c.entity[i] != split_entity[i]:
                            bad = True
                            break
                    if not bad:
                        return c
            else:
                if c.entity == entity:
                    return c
        return None

    @abstractmethod
    def run(self, inputs, idx):
        raise NotImplementedError("Must override run")





