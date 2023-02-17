from tvm.auto_scheduler.search_policy import SketchPolicy as AnsorSketchPolicy
from conductor.component.search_policy._base import SearchPolicy

import logging
logger = logging.getLogger("conductor.component.method.search_policy.sketch")

class SketchPolicy(SearchPolicy, AnsorSketchPolicy):
    _name = "ansor_sketch"
    
    def __repr__(self):
        return SearchPolicy.__repr__(self) + ":" + SketchPolicy._name

    def __init__(self, task, cost_model, configs=None):
        SearchPolicy.__init__(self, "ansor_sketch", configs=configs, child_default_configs={
            "eps_greedy": 0.05,
            "retry_search_one_round_on_empty": 1,
            "sample_init_min_population": 50,
            "sample_init_use_measured_ratio": 0.2,
            "evolutionary_search_population": 2048,
            "evolutionary_search_num_iters": 4,
            "evolutionary_search_mutation_prob": 0.85,
            "cpu_multi_level_tiling_structure": "SSRSRS",
            "gpu_multi_level_tiling_structure": "SSSRRSRS",
            "max_innermost_split_factor": 64,
            "max_vectorize_size": 16,
            "disable_change_compute_location": 0,
            "seed": None,
            "verbose": 0
        })
        
        params = {
            "eps_greedy": self.config["eps_greedy"],
            "retry_search_one_round_on_empty": self.config["retry_search_one_round_on_empty"],
            "sample_init_min_population": self.config["sample_init_min_population"],
            "sample_init_use_measured_ratio": self.config["sample_init_use_measured_ratio"],
            "evolutionary_search_population": self.config["evolutionary_search_population"],
            "evolutionary_search_num_iters": self.config["evolutionary_search_num_iters"],
            "evolutionary_search_mutation_prob": self.config["evolutionary_search_mutation_prob"],
            "cpu_multi_level_tiling_structure": self.config["cpu_multi_level_tiling_structure"],
            "gpu_multi_level_tiling_structure": self.config["gpu_multi_level_tiling_structure"],
            "max_innermost_split_factor": self.config["max_innermost_split_factor"],
            "max_vectorize_size": self.config["max_vectorize_size"],
            "disable_change_compute_location": self.config["disable_change_compute_location"]
        }

        AnsorSketchPolicy.__init__(
            self, 
            task, 
            program_cost_model=cost_model, 
            params=params, 
            seed=self.config["seed"],
            verbose=self.config["verbose"]
        )