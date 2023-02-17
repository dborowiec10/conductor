from conductor._base import Configurable
from conductor.component.measurer.doppler_base import DopplerBaseMeasurer
from conductor.experiment import Experiment
import logging
import tvm
import time
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence

logger = logging.getLogger("conductor.component.measurer.doppler_rlearn")

class DistributionFunctions(object):
    _name = "distribution_functions"

    def __repr__(self):
        return DistributionFunctions._name

    def __init__(self, device):
        self.device = device

    def sample(self, probs):
        dist = Categorical(probs)
        return dist.sample().to(self.device)
  
    def logproba(self, probs, x):
        dist = Categorical(probs)
        logp = dist.log_prob(x)
        return logp.to(self.device)

    def kldiv(self, probs, other_probs):
        dist1 = Categorical(probs)
        dist2 = Categorical(other_probs)
        diff = kl_divergence(probs, other_probs)
        return diff.to(self.device)

    def entropy(self, probs):
        dist = Categorical(probs)
        en = dist.entropy()
        return en.to(self.device)

class PolicyFunctions(object):
    _name = "policy_functions"

    def __repr__(self):
        return PolicyFunctions._name

    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
  
    def generalized_adv_estimation(self, values, rewards, next_values):
        delta = (rewards + self.gamma * next_values) - values
        gae = delta.clone().detach()
        # eq.11 & 12
        for t in reversed(range(len(delta)-1)):
            gae[t] = delta[t] + self.gamma*self.lam*delta[t+1]
        logger.info("Delta: %s, Gae: %s", delta, gae)
        return gae

class PPO(object):
    """
        https://arxiv.org/pdf/1707.06347.pdf
        We didn't use KLLoss.
    """

    _name = "ppo"

    def __repr__(self):
        return PPO._name

    def __init__(self, device, value_clip, vf_loss_coef, entropy_coef, gamma, lam):
        self.value_clip = value_clip
        self.vf_loss_coef = vf_loss_coef
        self.entropy_coef = entropy_coef
        self.dist_fn = DistributionFunctions(device)
        self.policy_fn = PolicyFunctions(gamma=gamma, lam=lam)
        self.gamma = gamma
        self.mse = nn.MSELoss()

    def loss_fn(self, action_proba, old_action_proba, values, old_values, next_values, actions, rewards):
        # not backpropagating to old policies.
        old_action_proba_d = old_action_proba.detach()
        next_values = next_values.squeeze(1)
        values = values.squeeze(1)
        advs = self.policy_fn.generalized_adv_estimation(values, rewards, next_values)
        advs = ((advs-advs.mean()) / (advs.std() + 1e-13)).detach()
        logproba = self.dist_fn.logproba(action_proba, actions)
        old_logproba = self.dist_fn.logproba(old_action_proba_d, actions).detach()
        ratios = (logproba - old_logproba).exp()
        mean_action_entropy = self.dist_fn.entropy(action_proba).mean()
        logger.info("Action Entropy: %s", mean_action_entropy)
        # Clip loss eq.7
        clipped_ratios = torch.clamp(ratios, -self.value_clip, self.value_clip)
        loss_clip = torch.min(advs*ratios, advs*clipped_ratios).mean()
        logger.info("loss_clip: %s, advs: %s", loss_clip, advs)
        # Value Loss a part of eq.9
        loss_value = self.mse((rewards + self.gamma*next_values), values)
        logger.info("lossValue: %s", loss_value)
        # Entropy loss a part of eq.9
        loss_ent = (self.dist_fn.entropy(action_proba) * self.entropy_coef).mean()
        logger.info("loss_ent: %s", loss_ent)
        # eq.9
        loss = loss_clip - (self.vf_loss_coef * loss_value) + loss_ent
        return loss.mean()

class DopNets(nn.Module):
    _name = "dop_nets"

    def __repr__(self):
        return DopNets._name

    def __init__(self, device, obs_dim, num_actions, hidden=64):
        nn.Module.__init__(self)
        self.observations = obs_dim
        self.actions = num_actions
        self.hidden = hidden
        self.criticNet = nn.Sequential(
            nn.Linear(self.observations, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, int(self.hidden/2)),
            nn.GELU(),
            nn.Linear(int(self.hidden/2), 1) 
        ).to(device)
        self.actorNet = nn.Sequential(
            nn.Linear(self.observations, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Linear(self.hidden, int(self.hidden/2)),
            nn.GELU(),
            nn.Linear(int(self.hidden/2), num_actions),
        ).to(device)
        self.softmax = nn.Softmax(dim=-1)
  
    def forward(self, states, target):
        if target == 'actor':
            act_state = self.actorNet(states)
            return self.softmax(act_state)
        crit = self.criticNet(states)
        return crit

class Experience(Dataset):
    _name = "experience"

    def __repr__(self):
        return Experience._name

    def __init__(self, n=1e10):
        self.actions = []
        self.states = []
        self.rewards = []
        self.next_states = []
        self.n = n
  
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        s = np.array(self.states[idx])
        a = np.array(self.actions[idx])
        r = np.array(self.rewards[idx])
        ns = np.array(self.next_states[idx])
        return (s,a,r,ns)

    def push(self, s, a, r, ns):
        if len(self.states) > self.n:
            self.states.pop(0)
            self.actions.pop(0)
            self.rewards.pop(0)
            self.next_states.pop(0)
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_states.append(ns)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.next_states[:]

class Agent(object):
    _name = "Agent"

    def __repr__(self):
        return Agent._name

    def __init__(
        self, obs_dim, action_dim, is_train, value_clip, entropy_coef, vf_loss_coef, 
        batchsize, gamma, lam, epochs, horizons, lr_rate=0.0002, log_intervals=10
    ):
        self.device = torch.device("cpu")
        self.entropy_coef = entropy_coef
        self.vf_loss_coef = vf_loss_coef
        self.horizons = horizons
        self.value_clip = value_clip
        self.num_actions = action_dim
        self.observations = obs_dim
        self.epochs = epochs
        self.batchsize = batchsize
        self.gamma = gamma
        self.lam = lam
        self.lr_rate = lr_rate
        self.is_train = is_train
        self.policy = DopNets(self.device, self.observations, self.num_actions)
        self.old_policy = DopNets(self.device, self.observations, self.num_actions)
        # not training 
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr_rate)
        self.log_intervals = log_intervals
        # components
        self.experience = Experience()
        self.dist_fn = DistributionFunctions()
        self.ppo_fn = PPO(
            self.device,
            self.value_clip, 
            self.vf_loss_coef, 
            self.entropy_coef, 
            self.gamma, 
            self.lam
        )
        if self.is_train:
            self.policy.train()
        else:
            self.policy.eval()
        self.learned_count = 0
  
    def push_experience(self, s, a, r, ns):
        self.experience.push(s,a,r,ns)
  
    def act(self, state):
        s = torch.tensor(state).float().to(self.device).detach()
        logger.info("Get action proba...")
        a_proba = self.policy(s, 'actor')
        logger.info("Get value prediction...")
        v_pred = self.policy(s, 'value')
        if self.is_train:
            action = self.dist_fn.sample(a_proba)
        else:
            action = torch.argmax(a_proba, 1)
        return action.int().cpu().item(), v_pred.float().cpu().item()
  
    def train_step(self, s, a, r, ns):
        a_proba, values = self.policy(s, 'actor'), self.policy(s, 'value')
        old_a_proba, old_values = self.old_policy(s, 'actor'), self.old_policy(s, 'value')
        next_values = self.policy(ns, 'value')
        loss = self.ppo_fn.loss_fn(
            a_proba, old_a_proba, values, 
            old_values, next_values, a, r
        )
        # loss = -loss
        logger.info("Loss: [%.4f]", loss)
        if torch.isnan(loss):
            logger.error("Errorer at --- s: %s, a: %s, r: %s, ns: %s", s, a, r, ns)
            raise ValueError("Loss should not be nan.")
        self.policy.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1e-6)
        self.optimizer.step()
        return loss.cpu().item()

    def update(self):
        dataloader = DataLoader(self.experience, self.batchsize, shuffle=False, drop_last=True)
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.old_policy.eval()

        self.policy.train()
        for _ in range(self.epochs):
            for i, (s, a, r, ns) in enumerate(dataloader):
                l = self.train_step(
                    s.float().to(self.device), 
                    a.float().to(self.device),
                    r.float().to(self.device),
                    ns.float().to(self.device)
                )
                if i % self.log_intervals == 0:
                    logger.info("Train Loss: %.4f", l)
        # self.experience.clear()
        self.learned_count += 1
  
    def save(self):
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'opt_state_dict': self.optimizer.state_dict()
        }, 'model.tar')
  
    def load(self):
        ckpt = torch.load('model.tar')
        self.policy.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['opt_state_dict'])

class DopplerRLMeasurer(DopplerBaseMeasurer):
    _name = "doppler_rlearn"
    
    def __repr__(self):
        return DopplerBaseMeasurer.__repr__(self) + ":" + DopplerRLMeasurer._name

    def __init__(self, builder, runner, configs=None, child_default_configs={}):
        DopplerBaseMeasurer.__init__(self, "doppler_rlearn", builder, runner, configs=configs, child_default_configs=Configurable.merge_configs({
            "max_dop": 128,
            "observations": 4,
            "entropy_coef": 0.3,
            "value_coeff": 1.0,
            "value_clip": 0.2,
            "gamma": 0.9,
            "lambda": 0.99,
            "epochs": 3,
            "batch_size": 8,
            "delta_threshold": 0.05,
            "failed_threshold": 0.05,
            "failed_due_to_dop_threshold": 0.25,
        }, child_default_configs, override_first=True))

        self.max_dop = self.config["max_dop"]
        self.observations = self.config["observations"]
        self.entropy_coef = self.config["entropy_cef"]
        self.value_coeff = self.config["value_coeff"]
        self.value_clip = self.config["value_clip"]
        self.gamma = self.config["gamma"]
        self.lam = self.config["lambda"]
        self.epochs = self.config["epochs"]
        self.batch_size = self.config["batch_size"]

        self.task_state_template = {
            "agent": None,
            "action_space": None,
            "state": None,
            "next_state": None,
            "last_it_failed": 0,
            "prior_max_dop": 0,
            "last_proposed_dop": 0,
            "dop": 1
        }

    def get_reward(self, delta_error, parallelism_error, latency):
        total_error = delta_error + parallelism_error + latency
        if (latency / total_error) > 0.95:
            return latency
        if (parallelism_error / total_error) > 0.95:
            return parallelism_error
        if (delta_error / total_error) < 0.1:
            return ((0.5 * parallelism_error) + (0.5 * latency))
        cost = 0.33 * delta_error + 0.33 * parallelism_error + 0.33 * latency
        logger.info("Delta Err: %.4f , Parallelism Err: %.4f, Latency Err: %.4f --- Reward: [%.4f]", delta_error, parallelism_error, latency, cost) 
        return -cost

    def measure(self, task, configs, options=None):
        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = True
        logger.info("measurer start on #%s inputs", str(len(configs)))

        # at this point, we have all build inputs prepared.
        build_inputs, updated_configs, theoretical_flop = self.builder.orch_scheduler.get_build_inputs(
            configs,
            task,
            self.runner.dev_ctx_details,
            self.runner.device_id,
            self.hash_callback,
            options=options
        )
        
        all_inp_configs = list(zip(self.build_configs(build_inputs), updated_configs))

        all_results = []

        self.create_task_state(self.task_idx)

        if self.stage == "proper":
            curr_max_dop = min(self.max_dop, len(configs))
            if curr_max_dop > self.task_states[str(self.task_idx)]["prior_max_dop"]:

                self.task_states[str(self.task_idx)]["action_space"] = [0] + [i + 1 for i in range(curr_max_dop)]
                logger.info("Action Space len : [%d] for Task %s", len(self.task_states[str(self.task_idx)]["action_space"]), task.workload)
                self.task_states[str(self.task_idx)]["agent"] = Agent(
                    self.observations,
                    len(self.task_states[str(self.task_idx)]["action_space"]),
                    True,
                    self.value_clip,
                    self.entropy_coef,
                    self.value_coeff,
                    self.batch_size,
                    self.gamma,
                    self.lam,
                    self.epochs,
                    -1
                )

            self.task_states[str(self.task_idx)]["prior_max_dop"] = curr_max_dop

        while len(all_inp_configs) > 0:
            if self.stage == "proper":
                _pd = self.optimizers_propose()
            else:
                _pd = 1

            if _pd == 0:
                self.task_states[str(self.task_idx)]["next_state"] = [-1.0, -1.0, -1.0, len(all_inp_configs)/len(configs)]
                r = -30
                a = copy.deepcopy(self.task_states[str(self.task_idx)]["dop"])
                logger.info("Adding to experience.... Now have: [%d]", len(self.task_states[str(self.task_idx)]["agent"].experience))
                self.task_states[str(self.task_idx)]["agent"].push_experience(
                    self.task_states[str(self.task_idx)]["state"], 
                    a, 
                    r, 
                    self.task_states[str(self.task_idx)]["next_state"]
                )
                self.task_states[str(self.task_idx)]["state"] = self.task_states[str(self.task_idx)]["next_state"]
                continue

            self.task_states[str(self.task_idx)]["dop"] = _pd
            prop_dop = _pd

            self.task_states[str(self.task_idx)]["dop"] = int(min(max(1, self.task_states[str(self.task_idx)]["dop"]), len(all_inp_configs)))
            self.task_states[str(self.task_idx)]["last_proposed_dop"] = self.task_states[str(self.task_idx)]["dop"]

            logger.info("measurer proposed_dop[%d], actual self.dop[%d], configs_left[%d]", prop_dop, self.task_states[str(self.task_idx)]["dop"], len(all_inp_configs))

            # pick n configs from the batch
            all_inp_configs, curr_configs, curr_bld_res = self.pick_n_configs(all_inp_configs, self.task_states[str(self.task_idx)]["dop"])
            
            st = time.time()
            # build & run them
            part_results, part_b_results = self.run_configs(curr_configs, curr_bld_res, remove_schedules=False, run_type="regular", set_timeout=self.determine_timeout(self.task_states[str(self.task_idx)]["dop"]))
            et = time.time()

            # initially, error is 0
            avg_delta = 0

            # percentage of bad (not errored candidates is initially 0.0)
            perc_bad = 0
            
            # time taken to measure self.dop candidates (per candidate) / we want to minimize this
            avg_meas_time_per_cand = (et - st) / self.task_states[str(self.task_idx)]["dop"]
            self.task_states[str(self.task_idx)]["times_so_far"].append(avg_meas_time_per_cand)

            if self.task_states[str(self.task_idx)]["dop"] > 1 and self.stage == "proper":
                
                # # done running, lets see if we need to rerun any
                part_results, cnt_fail_due_to_dop, cnt_err_tim_considered, succ_finally = self.remeasure_err_timeout(part_results, part_b_results, curr_configs)

                check = 0
                if succ_finally != 0:
                    check = (succ_finally / self.task_states[str(self.task_idx)]["dop"])

                # if not more than 25% of all failed candidates were due to dop
                if check > self.config["failed_due_to_dop_threshold"]:
                    remeas_indexes = self.get_remeasure_samples(part_results)

                    # re-measure chosen indexes serially:
                    logger.info("measurer re-measuring %d indexes: %s", len(remeas_indexes), str(remeas_indexes))
                    remeas_res = []
                    for idx in remeas_indexes:
                        rem_res, _ = self.run_configs([curr_configs[idx]], [part_b_results[idx]], remove_schedules=True if idx == remeas_indexes[-1] else False, run_type="remeasure",  set_timeout=self.determine_timeout(1))
                        remeas_res += rem_res

                    # calculate error for chosen indexes
                    avg_delta, deltas, avg_cmtp = self.calc_delta(part_results, remeas_res, remeas_indexes)

                    self.task_states[str(self.task_idx)]["deltas_so_far"] += deltas
                    # self.deltas_so_far += deltas

                    retval = self.update_results(remeas_indexes, remeas_res, part_results, avg_delta, avg_cmtp, theoretical_flop)
                    # retval = part_results
                else:
                    retval = part_results
                
                if cnt_fail_due_to_dop > int(self.task_states[str(self.task_idx)]["dop"] * self.config["failed_threshold"]):
                    perc_bad = (cnt_fail_due_to_dop / self.task_states[str(self.task_idx)]["dop"])

                self.task_states[str(self.task_idx)]["perc_bad_so_far"].append(perc_bad)
            else:
                retval = part_results
            
            all_results += retval

            if self.stage == "proper":
                self.task_states[str(self.task_idx)]["next_state"] = [perc_bad, avg_meas_time_per_cand, avg_delta, len(all_inp_configs)/len(configs)]
                r = self.get_reward(avg_delta, perc_bad, avg_meas_time_per_cand)
                a = copy.deepcopy(self.task_states[str(self.task_idx)]["dop"])
                logger.info("Adding to experience.... Now have: [%d]", len(self.task_states[str(self.task_idx)]["agent"].experience))
                self.agent.push_experience(self.task_states[str(self.task_idx)]["state"], a, r, self.task_states[str(self.task_idx)]["next_state"])
                self.task_states[str(self.task_idx)]["state"] = self.task_states[str(self.task_idx)]["next_state"]
            
                if len(self.task_states[str(self.task_idx)]["agent"].experience) > (self.batch_size/2):
                    logger.info("Have at least half of a batchsize experiences of the number of configs in this space... start learning")
                    self.task_states[str(self.task_idx)]["agent"].update()
                    logger.info("Done %d training.", self.task_states[str(self.task_idx)]["agent"].learned_count)

        m_inputs, m_results, conf_m_inputs, conf_m_results, total_error_count, error_counts = self.builder.orch_scheduler.get_inp_res_err(configs, all_results, theoretical_flop, task)
        Experiment.current.update_task_status(self.task_idx, len(configs), error_counts)
        tvm.autotvm.env.GLOBAL_SCOPE.in_tuning = False
        self.configurer.add_records(conf_m_inputs, conf_m_results)
        logger.info("measurer stop on #%s inputs, total errors: %s", str(len(configs)), str(sum(error_counts[1:])))
        return (m_inputs, m_results, total_error_count)