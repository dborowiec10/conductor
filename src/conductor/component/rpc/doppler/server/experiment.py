from conductor.store import Store
from conductor.mediation import ERROR_TYPES
import uuid
import time
import copy
import json

class Experiment(object):
    current = None
    
    def __repr__(self):
        return Experiment._name
    
    def __init__(self, store_config, exp_id):
        self.s = Store(
            store_config["host"],
            store_config["port"], 
            store_config["user"],
            store_config["pass"],
            store_config["db"]
        )
        self.exp = {
            "id": exp_id,
            "data": {
                "tasks": {},
                "methods": {}
            }
        }
        self.stage = None
        self.current_task = None
        self.current_method = None
        Experiment.current = self
    
    def insert_many(self, coll, data):
        if data != None and type(data) == list and len(data) > 0:
            dcopy = data.copy()
            base = self.get_base()
            for k, _ in enumerate(dcopy):
                dcopy[k].update(base)
            self.s.insert_many(coll, dcopy)
            
            
    def get_task(self, t):
        exclude_keys = ['status']
        return {k: t[k] for k in set(list(t.keys())) - set(exclude_keys)}

    def get_base(self):
        return {
            "experiment_id": self.exp["id"],
            "task": self.get_task(self.current_task),
            "method": self.current_method
        }

    def add_task(self, task_config, task_idx):
        t = {
            "idx": task_idx,
        }
        t.update(task_config)
        s = json.dumps(t)
        # use dict as ordered set
        self.exp["data"]["tasks"][s] = None
        
    def add_method(self, method_spec, method_idx):
        m = {
            "idx": method_idx,
            "kind": method_spec["kind"],
            "schedulable": method_spec["scheduling"],
            "name": method_spec["name"],
            "cost_model": method_spec["cost_model"],
            "optimizer": method_spec["optimizer"],
            "search_policy": method_spec["search_policy"],
            "sampler": method_spec["sampler"],
            "filter": method_spec["filter"]
        }
        s = json.dumps(m)
        # use dict as ordered set
        self.exp["data"]["methods"][s] = None
    
    def set_task(self, idx):
        self.current_task = json.loads(list(self.exp["data"]["tasks"].keys())[idx])

    def set_method(self, idx):
        self.current_method = json.loads(list(self.exp["data"]["methods"].keys())[idx])
        
    def set_experiment_stage(self, stage):
        self.current_stage = stage

