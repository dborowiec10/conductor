from conductor.core.orchestrator.strategy._base import Strategy
import traceback
from collections import OrderedDict
from scipy.special import logsumexp
import numpy as np
import math
import logging
import time
import os
from conductor.core.orchestrator.utils import ios
from tvm.auto_scheduler import measure
from multiprocessing.managers import BaseManager, NamespaceProxy
logger = logging.getLogger("conductor.orchestrator.strategy.p3bo")

class Proposal:
    def __init__(self, key, proposers=set([]), cost=0.0):
        self.proposers = proposers
        self.cost = cost
        self.key = key

class Progress:
    def __init__(self, task_idx):
        self.id = task_idx
        # build and run is separated because we need to coordinate 
        # the separation between builder and runner
        # we keep the idxes of the builder for which build is not needed
        # then when runner is looking to run, we skip the build that is not needed.
        self.build_iteration = 0
        self.run_iteration = 0
        # key: iteration_idx, value: List of List of idxes to remove
        self.to_skip = {}

class Packet(object):
    def __init__(self, proposals, task_idx):
        self.task_idx = task_idx
        self.proposals = proposals
        self.local_hashes = {}

    def hash_callback(self, config, hash_value):
        # logger.info(self.proposals.keys())
        proposal = self.proposals.get(hash_value, None)
        should_skip = False
        if proposal is None:
            proposal = Proposal(hash_value, set([self.task_idx]))
            self.proposals[hash_value] = proposal
        else:
            proposal.proposers.add(self.task_idx)
            self.proposals[hash_value] = proposal
            # logger.info("Proposal with key %s -- proposers: %s", hash_value, proposal.proposers)
            should_skip = True
        self.local_hashes[str(config)] = hash_value
        return should_skip



class Manager(BaseManager): pass

class TestProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__', 'hash_callback')

    def hash_callback(self, config, hash_value):
        callmethod = object.__getattribute__(self, '_callmethod')
        return callmethod('hash_callback', (config, hash_value))

Manager.register('packet', Packet, TestProxy)


class P3BOStrategy(Strategy):
  def __init__(self, 
               self_spec, 
               task_spec, 
               measurer,
               builder, 
               runner, 
               evaluator, 
               setting, 
               config, 
               general_config, 
               res_path):
      Strategy.__init__(self, self_spec, task_spec, measurer, builder, runner, evaluator, setting, config, general_config, res_path)
      self.decay_rate = 0.99
      self.history_window = 5
      self.num_tasks = len(task_spec)
      self.num_measure_trials = 0
      self.rewards = OrderedDict()
      self.task_best_costs = OrderedDict()
      self.stopped = {}
      self.unreasonable_num = 1e32
      self.best_score = {}
      self.best_score_all = self.unreasonable_num
      for t_spec in task_spec:
          t_idx = t_spec["task_idx"]
          self.rewards[t_idx] = [0.0] * self.history_window
          self.stopped[t_idx] = False
          self.best_score[t_idx] = self.unreasonable_num
          

      # Keep track of what configuration each optimizer has proposed for.
      # key: "hashedkey", value: Proposal
      self.proposals = OrderedDict()
      self.tasks_idx_to_remove_iteration = {}
      self.total_batch_size = self.setting.get("batch_size", 64) * self.num_tasks
      self.trials = self.setting.get("total_trials", 512*1e9)
      self.iteration = 0
      self.manager = Manager()
      self.manager.start()

  def calculate_reward_for_idx(self, task_idx):
      #eq-1
      # (Best Execution time - Optimizer Best) / Best Execution time
      
      # this task has errors. No proposals.
      if self.best_score[task_idx] == self.unreasonable_num:
          logger.info("No reward for task %d yet.", task_idx)
          return 0.0

      # everyone has errors. No proposals.
      if self.best_score_all == self.unreasonable_num:
          logger.info("No rewards for everyone yet.")
          return 0.0

      return np.exp((self.best_score_all - self.best_score[task_idx]) / self.best_score_all)

  def calculate_rewards(self):
      for task_idx, reward_histories in self.rewards.items():
          reward_t = self.calculate_reward_for_idx(task_idx)
          reward_histories.pop(0)
          reward_histories.append(reward_t)
          self.rewards[task_idx] = reward_histories
          logger.info("Task %d -- %s ", task_idx, self.rewards[task_idx])

  def propose_task(self):
      """Currently only works for 1 task , multiple optimizers.
         
         NOTE: in order to work for multiple tasks (a model with many operators), 
         will need to re-think how that is done.
      """
      task_idx = []
      rewards = []
      for k, reward_histories in self.rewards.items():
          credit = 0
          # eq-2
          for idx, history in enumerate(reward_histories):
              credit += history * math.pow(self.decay_rate, self.history_window-idx)

          rewards.append(credit)
          task_idx.append(k)
      rewards= np.array(rewards)
      # numerical stability 
      # eq-3 with tau = 1
      proba = np.exp(rewards - logsumexp(rewards))
      return np.random.choice(task_idx, p=proba)




#   def filter(self, task_idx, task_progress, hashes):
#     #   logger.info(self.proposals.keys())
#     #   logger.info(hashes)

#       for idx, hash_val in enumerate(hashes):
#           proposal = self.proposals.get(hash_val, None)
#           if proposal is None:
#               proposal = Proposal(hash_val, set([task_idx]))
#               self.proposals[hash_val] = proposal
#               continue
              
#           proposal.proposers.add(task_idx)
#           self.proposals[hash_val] = proposal
#           logger.info("Proposal with key %s -- proposers: %s", hash_val, proposal.proposers)

#           skips = task_progress.to_skip.get(task_progress.build_iteration, [])
#           skips.append(idx)
#           task_progress.to_skip[task_progress.build_iteration] = skips
      
#       if task_progress.to_skip.get(task_progress.build_iteration, None) is None:
#           task_progress.to_skip[task_progress.build_iteration] = []

          

#   def call_back(self, task_idx, hashes, local_hashes):
#       # compare configuration with the past configuration
#     #   task_progress = self.tasks_idx_to_remove_iteration.get(task_idx, None)
#     #   if task_progress is None:
#     #       task_progress = Progress(task_idx)

#       self.filter(task_idx, task_progress, hashes)
#       task_progress.build_iteration += 1
#       # TODO: looks like there is race condition or context problems
#       # hence has to append the hash here instead of returning the hashes.
#       # FIXME.
#       for lh in hashes:
#           local_hashes.append(lh)
#       logger.info("Local Hashes Length: %d", len(local_hashes))
#       self.tasks_idx_to_remove_iteration[task_idx] = task_progress

  
#   def decide(self, task_idx, config, build_res):
#       task_progress = self.tasks_idx_to_remove_iteration.get(task_idx, None)
#       for idx, build_r in enumerate(build_res):
#           skip_idxes = task_progress.to_skip[task_progress.run_iteration]
#           if idx in skip_idxes:
#               build_res[idx] = build_r._replace(error_no=ios.MeasureErrorNo.SKIP)
#               logger.info("Build result %d -- %s", idx, build_res[idx])
#               continue

#       task_progress.run_iteration += 1
#       self.tasks_idx_to_remove_iteration[task_idx] = task_progress        




  def save_proposals_and_update(self, task_idx, measure_inputs, 
                                measure_results, local_hashes):
      for measure_input, measure_result in zip(measure_inputs, measure_results):
          hash_str = local_hashes[str(measure_input.config)]
          proposal = self.proposals.get(hash_str, None)
          if proposal is None:
              raise RuntimeError("Proposal should not be none.")

          if measure_result.error_no == ios.MeasureErrorNo.NO_ERROR:
              if measure_result.mean == 0.0:
                  continue

              proposal.cost = measure_result.mean
              # TODO: At the moment this has an assumption 
              # where each task is optimizing for the **SAME** operator
              self.best_score_all = min(self.best_score_all, proposal.cost)
              logger.info("Proposal %.26f , Best %.26f", proposal.cost, self.best_score_all)
              for proposer in proposal.proposers:
                  self.best_score[proposer] = min(self.best_score[proposer], proposal.cost)
                  logger.info("Proposer %d -- best score %.26f", proposer, self.best_score[proposer])
              self.proposals[hash_str] = proposal
          elif measure_result.error_no == ios.MeasureErrorNo.SKIP:
              for proposer in proposal.proposers:
                  self.best_score[proposer] = min(self.best_score[proposer], proposal.cost)
                  logger.info("Proposer %d -- best score %.26f", proposer, self.best_score[proposer])
      
      logger.info(self.best_score)
      logger.info(self.best_score_all)


          
  def run(self):
      model, task_specs = self.prepare_tasks()
      task_implement_pairs = []

      
      while self.iteration < 10 and all([not v for k,v in self.stopped.items()]):
          sample_size = 0

          counter = {}
          for task_spec in task_specs:
              counter[task_spec["details"]["task_idx"]] = 0

          while sample_size < self.total_batch_size:
              chosen_idx = self.propose_task()
              counter[chosen_idx] += 1
              sample_size+=1

          logger.info("Current sample size assignments: %s", str(counter))

          for task_spec in task_specs:
              task_idx = task_spec["details"]["task_idx"]

              num_measure = counter[task_idx]
              if num_measure <= 0:
                  continue
              
              should_stop = self.stopped[task_idx]

              if should_stop:
                  continue

              task_opt = task_spec["details"]["method_spec"]["optimizer"]
              task_cm = task_spec["details"]["method_spec"]["cost_model"]
              
              logger.info("Trial for task %s - with optimizer %s , cost model %s", 
                task_idx, task_opt, task_cm)
              
              _method, task, task_weight, implement, configurer, part_of_model, idetsvals = \
                  self.prepare_task(task_spec, "proper")
              task_implement_pairs.append((part_of_model, implement, configurer))

              k1 = self.start_profiler("strategy", "time_profiler", idetsvals)
              k2 = self.start_profiler("strategy", "python_system_monitor", idetsvals)
              k3 = self.start_profiler("strategy", "python_powercap_profiler", idetsvals)

              idets = {"keys": self.inp_dets["keys"], "vals": idetsvals}
              _method.set_input_details(idets)
              _method.set_measurer(self.measurer)
              _method.load(task, configurer)
              
              packet = self.manager.packet(self.proposals, task_idx)
              self.measurer.set_hash_callback(packet)

              try:
                  flop_dict, cost_dict, error_count, m_inputs, m_results = _method.execute(num_measure)
                  
                  self.proposals = packet.proposals
                  self.save_proposals_and_update(task_idx, m_inputs, m_results, packet.local_hashes)

                  if len(m_inputs) < 1:
                      should_stop = True

              except Exception as e:
                  logger.error("stopping task due to exception!")
                  should_stop = True
                  logger.error("exception" + traceback.format_exc())
              
              self.stopped[task_idx] = should_stop

              _method.persist()
              _method.unload()
              self.stop_profiler("strategy", "time_profiler", k1)
              self.stop_profiler("strategy", "python_system_monitor", k2)
              self.stop_profiler("strategy", "python_powercap_profiler", k3)

              self.persist()

          self.iteration += 1
          self.calculate_rewards()
          logger.info("Iteration: %d, Current Rewards: %s", self.iteration, str(self.rewards))

      logger.info("stop strategy proper")
      for (pom, imp, conf) in task_implement_pairs:
          if not pom:
              conf.save_best_records(imp.best_path)
              conf.save_best_records(os.path.join(self.res_path, "best.log"))
              conf.save_records(conf.records, os.path.join(self.res_path, "all.log"))
              imp.save_details(os.path.join(self.res_path, "details.json"))

      if model["model"] and model["implementation"]:
          model["configurer"].save_best_records(model["implementation"].best_path)
          model["configurer"].save_best_records(os.path.join(self.res_path, "best.log"))
          model["configurer"].save_records(model["configurer"].records, os.path.join(self.res_path, "all.log"))
          model["implementation"].save_details(os.path.join(self.res_path, "details.json"))
          return [model["implementation"]]
      else:
          return [implement for t in task_specs]