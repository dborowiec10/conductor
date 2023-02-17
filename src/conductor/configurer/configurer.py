from conductor.mediation import ERROR_TYPES, FallbackContext, HistoryBestContext, SingleConfigContext, decode_measure_input_result, encode_measure_input_result
from conductor.experiment import Experiment

from random import randrange
import os
import json
import numpy as np

class Configurer(object):
    _name = "configurer"
    
    def __repr__(self):
        return Configurer._name

    def __init__(self, base_path):
        self.base_path = base_path
        self.records = []
        self.ranked_records = []

    def load_records(self, fname):
        with open(fname, "r") as _fil:
            for row in _fil:
                ret, _ = decode_measure_input_result(row)
                if ret is None:
                    continue
                else:
                    self.records.append(ret)

    def load_all_records(self, filename=None):
        fname = os.path.join(self.base_path, "all.log") if filename is None else filename
        self.load_records(fname)

    def load_ranked_records(self, filename=None):
        fname = os.path.join(self.base_path, "rank.log") if filename is None else filename
        self.load_records(fname)

    def load_best_records(self, filename=None):
        fname = os.path.join(self.base_path, "best.log") if filename is None else filename
        self.load_records(fname)

    def load_best_ranked_records(self, filename=None):
        fname = os.path.join(self.base_path, "best.rank.log") if filename is None else filename
        self.load_records(fname)

    def save_all_records(self, filename=None):
        fname = os.path.join(self.base_path, "all.log") if filename is None else filename
        if not os.path.exists(fname):
            open(fname, 'a').close()
        outexp = []
        with open(fname, "w") as _fil:
            for k, (inp, res) in enumerate(self.records):
                _str, _dict = encode_measure_input_result(inp, res, k=k)
                outexp.append(_dict)
                if _str[-1] != "\n":
                    _str += "\n"
                _fil.write(_str)
        Experiment.current.insert_logs("all", outexp, fname)
        

    def save_best_records(self, filename=None):
        fname = os.path.join(self.base_path, "best.log") if filename is None else filename
        if not os.path.exists(fname):
            open(fname, 'a').close()
        outexp = []
        with open(fname, "w") as _fil:
            best_records = self.pick_best_records()
            for k, (inp, res) in enumerate(best_records):
                _str, _dict = encode_measure_input_result(inp, res, k=k)
                outexp.append(_dict)
                if _str[-1] != "\n":
                    _str += "\n"
                _fil.write(_str)
        Experiment.current.insert_logs("all", outexp, fname)


    def save_ranked_records(self, filename=None):
        fname = os.path.join(self.base_path, "rank.log") if filename is None else filename
        if not os.path.exists(fname):
            open(fname, 'a').close()
        outexp = []
        with open(fname, "w") as _fil:
            for k, (inp, res) in enumerate(self.ranked_records):
                _str, _dict = encode_measure_input_result(inp, res, k=k)
                outexp.append(_dict)
                if _str[-1] != "\n":
                    _str += "\n"
                _fil.write(_str)
        Experiment.current.insert_logs("all", outexp, fname)


    def save_best_ranked_records(self, filename=None):
        fname = os.path.join(self.base_path, "best.rank.log") if filename is None else filename
        if not os.path.exists(fname):
            open(fname, 'a').close()
        outexp = []
        with open(fname, "w") as _fil:
            best_records = self.pick_best_records(records=self.ranked_records)
            for k, (inp, res) in enumerate(best_records):
                _str, _dict = encode_measure_input_result(inp, res, k=k)
                outexp.append(_dict)
                if _str[-1] != "\n":
                    _str += "\n"
                _fil.write(_str)
        Experiment.current.insert_logs("all", outexp, fname)
                
    def config_from_record(self, record):
        if record is not None and len(record) > 0 and record[0] is not None:
            return record[0].config
        else:
            return None

    def pick_index_record(self, idx):
        return self.records[idx]

    def pick_index_config(self, idx):
        return self.config_from_record(self.pick_index_record(idx))

    def pick_index_config(self, idx):
        return self.config_from_record(self.pick_index_record(idx))

    def context_fallback(self):
        return FallbackContext()

    def context_history_best(self, records=None):
        if records is None:
            return HistoryBestContext(self.records)
        else:
            return HistoryBestContext(records)

    def context_single_random(self, target, key, t_args):
        return SingleConfigContext(self.pick_random_config(target, key, t_args))

    def context_single_history_best(self, target, key, t_args):
        return SingleConfigContext(self.pick_best_config(target, key, t_args))

    def context_single_index(self, idx):
        return SingleConfigContext(self.pick_index_config(idx))

    def equal_by_model(self, inp, target, key, t_args):
        by_model = target.model == inp.target.model and key == inp.workload_key and t_args == inp.task.args
        return by_model

    def equal_by_keys(self, inp, target, key, t_args):
        by_keys = target.keys == inp.target.keys and key == inp.workload_key and t_args == inp.task.args
        return by_keys

    def add_records(self, inputs, results, optional=None):
        for inp, res in zip(inputs, results):
            self.records.append((inp, res))

    def add_ranked_records(self, inputs, results, optional=None):
        for inp, res in zip(inputs, results):
            self.ranked_records.append((inp, res))

    def pick_best_records(self, records=None):
        best_context = self.context_history_best(records=records)
        best_set = set()
        for v in list(best_context.best_by_model.values()):
            best_set.add(v[0].unique_key())
        for v in list(best_context.best_by_key.values()):
            best_set.add(v[0].unique_key())
        ret_records = []
        recs = self.records if records is None else records
        for inp, res in recs:
            if inp.unique_key() in best_set:
                ret_records.append((inp, res))
                best_set.remove(inp.unique_key())
        return ret_records

    def pick_best_record(self, target, key, t_args):
        all_records = self.pick_best_records()
        for inp, res in all_records:
            by_model = self.equal_by_model(inp, target, key, t_args)
            by_keys = self.equal_by_keys(inp, target, key, t_args)
            if by_model or by_keys:
                return (inp, res)
        return (None, None)

    def pick_best_configs(self):
        return [self.config_from_record(r) for r in self.pick_best_records()]

    def pick_best_config(self, target, key, t_args):
        return self.config_from_record(self.pick_best_record(target, key, t_args))

    def pick_random_records(self):
        groups = {}
        for inp, res in self.records:
            k = inp.unique_key()
            if k not in groups:
                groups[k] = []
            groups[k].append((inp, res))
        ret_records = []
        for k, v in groups.items():
            rec = v[randrange(len(v))]
            ret_records.append(rec)
        return ret_records
        
    def pick_random_configs(self):
        return [self.config_from_record(r) for r in self.pick_random_records()]

    def pick_random_record(self, target, key, t_args):
        matching_records = []
        for inp, res in self.records:
            by_model = self.equal_by_model(inp, target, key, t_args)
            by_keys = self.equal_by_keys(inp, target, key, t_args)
            if by_model or by_keys:
                matching_records.append((inp, res))
        return matching_records[randrange(len(matching_records))]

    def pick_random_config(self, target, key, t_args):
        return self.config_from_record(self.pick_random_record(target, key, t_args))