from conductor._base import Configurable
from conductor.component.method._base import Method
from conductor.component.method.flex.utils import flatten_graph, Config
from conductor.component.method.flex.space import generate_space_inter_op, generate_space_intra_op
from conductor.component.method.flex.scheduler import GraphScheduler, OpScheduler
from conductor.experiment import Experiment

import tvm

import logging
logger = logging.getLogger("conductor.component.method.flex")

class OpState(object):
    def __init__(self):
        self.inline = False
        self.loop_lst = []
        self.loop_idx = []
        self.compute_at = False
        self.consumer_lst = []

def get_configs(old_configs, new_configs, mode):
    ret_configs = []
    for conf in new_configs:
        if mode == "op":
            c = Config(old_configs.op_config_lst + [conf], old_configs.graph_config)
        elif mode == "graph":
            c = Config(old_configs.op_config_lst, conf)
        ret_configs.append(c)
    return ret_configs

def propose_configs(scheduler, trials, type_keys):
    warm_up_ret = scheduler.walker_group.forward(trials, policy="random")
    warm_up_configs = [{} for i in range(trials)]   # empty configs
    warm_up_indices = [{} for i in range(trials)]   # the indices
    for count in range(trials):
        for type_key in type_keys:
            warm_up_configs[count][type_key] = []
            for name in scheduler.space.types[type_key]:
                entity = warm_up_ret[name][0][count]
                warm_up_indices[count][name] = warm_up_ret[name][1][count]
                warm_up_configs[count][type_key].append(entity)
    return warm_up_configs, warm_up_indices

class FlexMethod(Method):
    _name = "flex"
    
    def __repr__(self):
        return Method.__repr__(self) + ":" + FlexMethod._name

    def __init__(self, subtypes, orch_scheduler, configs=None, child_default_configs={}, profilers_specs=[]):
        Method.__init__(
            self,
            ["flex"] + subtypes,
            "flex",
            orch_scheduler,
            configs=configs,
            child_default_configs=Configurable.merge_configs({
                "force_inline": True,
                "rewrite": False,
                "slevel": 4,
                "rlevel": 3,
                "op_space_groups": 3,
                "op_unroll_policy": "off",
                "op_fuse_policy": "off",
                "op_reorder_policy": "off",
                "op_perf_model_path_lst": [],
                "graph_perf_model_path": None,
                "op_graph_trial_split": [0.8, 0.2],
                "op_graph_early_stop": [15, 5],
                "op_trials": None,
                "graph_trials": None
            }, child_default_configs, override_first=True),
            profilers_specs=profilers_specs
        )
        self.force_inline = self.config["force_inline"]
        self.rewrite = self.config["rewrite"]
        if self.rewrite:
            self.force_inline = True
        self.slevel = self.config["slevel"]
        self.rlevel = self.config["rlevel"]
        self.op_space_groups = self.config["op_space_groups"]
        self.op_unroll_policy = self.config["op_unroll_policy"]
        self.op_fuse_policy = self.config["op_fuse_policy"]
        self.op_reorder_policy = self.config["op_reorder_policy"]
        self.op_perf_model_path_lst = self.config["op_perf_model_path_lst"]
        self.graph_perf_model_path = self.config["graph_perf_model_path"]
        self.op_graph_trial_split = self.config["op_graph_trial_split"]
        self.op_graph_early_stop = self.config["op_graph_early_stop"]
        self.op_trials = self.config["op_trials"]
        self.graph_trials = self.config["graph_trials"]

    def load(self, task, configurer):
        Method.load(self, task, configurer)

    def unload(self):
        Method.unload(self)

    def measure_configs(self, scheduler, configs, new_configs, config_mode):
        confs = get_configs(configs, new_configs, config_mode)
        results = []
        measure_inputs, measure_results, total_error_count = self.measurer.measure(
            scheduler.task,
            confs,
            options=(scheduler.op_pos if config_mode == "op" else None, scheduler.rewrite)
        )
        Experiment.current.set_experiment_stage("method:execute")
        for measure_result in measure_results:
            results.append((measure_result.mean * 1e3) if measure_result.mean < 1e20 else float("inf"))
        return results, {"measure_inputs": measure_inputs, "measure_results": measure_results, "total_error_count": total_error_count}

    # warm up a single op
    def warm_up(self, scheduler, epochs, trials, configs, type_keys, config_mode, max_repeat=20, use_model=False):
        warm_up_enough = False
        count_repeat = 0
        count_measurements = 0
        res_dicts = []
        while not warm_up_enough:
            for _ in range(epochs):
                warm_up_configs, warm_up_indices = propose_configs(scheduler, trials, type_keys)
                if use_model:
                    warm_up_results = scheduler.walker_group.query_performance(warm_up_indices)
                else:
                    warm_up_results, res_dict = self.measure_configs(scheduler, configs, warm_up_configs, config_mode)
                    res_dicts.append(res_dict)
                    scheduler.walker_group.add_perf_data(warm_up_indices, warm_up_results)
                for count in range(trials):
                    if warm_up_results[count] < 1e20:
                        scheduler.walker_group.record(warm_up_indices[count], warm_up_results[count])
                count_measurements += trials

            # this will check if we can get at least 1 config
            if not scheduler.walker_group.top1():
                epochs = 1
                count_repeat += 1
                if count_repeat >= max_repeat or count_measurements >= (epochs * trials):
                    warm_up_enough = True
            else:
                warm_up_enough = True
        return res_dicts

    # we can expect that num_measurements == num_trials at strategy level (i.e. there is only a single batch)
    # this is doen because flextensor
    def execute(self, num_measurements):
        Experiment.current.set_experiment_stage("method:execute")
        self.profiling_checkpoint("execute:start")

        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = True
        tvm.autotvm.env.GLOBAL_SCOPE.silent = True

        op_type_keys = ["fuse", "reorder", "spatial", "reduce", "unroll"]
        graph_type_keys = ["inline", "merge"]

        # flextensor is odd about their trials, can try to figure out how many measurements to do based on the num_measurements
        # can also just use flextensor's method of doing trials and assign them from config
        if self.op_trials is not None and self.graph_trials is not None:
            ops_trials = int(self.op_trials)
            graph_trials = int(self.graph_trials)
            mode = "flex"
        else:
            # we need to manually figure this out
            ops_trials = int(int(num_measurements) * self.op_graph_trial_split[0])  # ops
            ops_trials = ops_trials // int(len(self.op_schedulers))
            graph_trials = int(int(num_measurements) * self.op_graph_trial_split[1])  # graph
            mode = "manual"

        ops_early_stop = int(self.op_graph_early_stop[0])
        graph_early_stop = int(self.op_graph_early_stop[1])

        err_cnt = 0
        best_flop = 0
        best_flop_config = None
        best_flop_idx = None
        best_flop_pair = None

        best_cost = 1e10
        best_cost_config = None
        best_cost_idx = None
        best_cost_pair = None

        res_dict_list = []

        if self.rewrite:
            self.force_inline = True

        ops, _ = self.task.func(*self.task.args)
        
        op_lst, down_graph = flatten_graph(ops)

        # state of ops
        op_states = [OpState() for op in op_lst]
        for count_op, op in enumerate(op_lst):
            consumer_lst = []
            for count_output in range(op.num_outputs):
                if op.output(count_output) in down_graph:
                    consumer_lst.extend(down_graph[op.output(count_output)])
            op_states[count_op].consumer_lst = list(set(consumer_lst))

        s = tvm.te.create_schedule(op_lst[0])

        _op_perf_model_path_lst = [None for i in range(len(op_lst))]
        if len(self.op_perf_model_path_lst) > 0:
            for (op_pos, path) in self.op_perf_model_path_lst:
                _op_perf_model_path_lst[op_pos] = path

        graph_space = generate_space_inter_op(op_lst, down_graph, force_inline=self.force_inline, special_space=self.task.special_space)
        if self.force_inline:
            configs = Config([], {"inline": [graph_space.subspaces["inline"].static_entities[0]]})
        else:
            configs = Config([], None)

        for pos, op in enumerate(op_lst):
            if "cuda" in self.task.target.keys:
                space = generate_space_intra_op(op, down_graph, slevel=self.slevel, rlevel=self.rlevel, groups=self.op_space_groups)

            elif "cpu" in self.task.target.keys:
                rslevel = max(self.slevel, self.rlevel)
                space = generate_space_intra_op(
                    op, down_graph, 
                    slevel=rslevel, rlevel=rslevel, 
                    unroll_policy=self.op_unroll_policy, 
                    fuse_policy=self.op_fuse_policy, 
                    reorder_policy=self.op_reorder_policy
                )
            
            op_sched = OpScheduler(
                self.task, space, pos,
                perf_path=_op_perf_model_path_lst[pos],
                use_model=False if _op_perf_model_path_lst[pos] is None else True,
                rewrite=self.rewrite
            )

            if self.force_inline and graph_space.subspaces["inline"].able_inline(pos):
                op_config = {}
            else:
                logger.info(f"op_scheduler:{pos}, tuning")
                if op_sched.perf_path is not None:
                    op_sched.walker_group.model_path = op_sched.perf_path
                op_config, result_dictionaries = self.procedure(op_sched, configs, op_type_keys, mode, "op", ops_trials, ops_early_stop, use_model=op_sched.use_model)
                res_dict_list += result_dictionaries
            configs.op_config_lst.append(op_config)

        logger.info(f"graph_scheduler, tuning")

        graph_scheduler = GraphScheduler(
            self.task,
            graph_space,
            perf_path=self.graph_perf_model_path,
            use_model=False if self.graph_perf_model_path is None else True,
            rewrite=self.rewrite
        )
        if graph_scheduler.perf_path is not None:
            graph_scheduler.walker_group.model_path = graph_scheduler.perf_path
        graph_config, result_dictionaries = self.procedure(graph_scheduler, configs, graph_type_keys, mode, "graph", graph_trials, graph_early_stop, use_model=graph_scheduler.use_model)
        res_dict_list += result_dictionaries

        measure_inputs = []
        measure_results = []
        for res_dict in res_dict_list:
            measure_inputs += res_dict["measure_inputs"]
            measure_results += res_dict["measure_results"]
            err_cnt += res_dict["total_error_count"]

        for idx, (measure_input, measure_result) in enumerate(zip(measure_inputs, measure_results)):
            flops = measure_result.achieved_flop
            mean_cost = measure_result.mean   
            if flops > best_flop:
                best_flop = flops
                self.best_flops = best_flop
                best_flop_config = measure_input.config
                best_flop_pair = (measure_input, measure_result)
                best_flop_idx = idx                            
            if mean_cost < best_cost:
                best_cost = mean_cost
                best_cost_config = measure_input.config
                best_cost_pair = (measure_input, measure_result)
                best_cost_idx = idx

        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = False
        tvm.autotvm.env.GLOBAL_SCOPE.silent = False

        self.profiling_checkpoint("execute:stop")

        return ({
            "flop": best_flop, "config": best_flop_config,
            "pair": best_flop_pair, "idx": best_flop_idx
        }, {
            "cost": best_cost, "config": best_cost_config,
            "pair": best_cost_pair, "idx": best_cost_idx
        }, err_cnt, measure_inputs, measure_results)
