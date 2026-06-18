# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm
import os


from torch.nn.utils.rnn import pad_sequence
from scipy.signal import savgol_filter
from scipy.special import logit, expit
from scipy.stats import genpareto
from kneed import KneeLocator
import lmoments3 as lm
from lmoments3 import distr
import math

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.trainer.ppo.utils import Role

from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from ipdb import set_trace #DM: REMOVE LATER?

class RayBMCTrainer(RayPPOTrainer):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._build_ckpt_handler() # for tracking BMC elements (overrides ._load_checkpoint())
        self._load_checkpoint() 

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        if self.config.algorithm.bmc.ltt.enable:
            bmc_path = os.path.join(self.config.trainer.default_local_dir,'bmc')
            os.makedirs(bmc_path, exist_ok=True)
            
            sampler = self.train_dataloader.sampler

            tree = self.train_dataloader.sampler.tree
            tree_path = os.path.join(bmc_path,'tree.pt')
            torch.save(tree, tree_path)

            leaves = self.train_dataloader.sampler.clusters.numpy()
            leaf_path = os.path.join(bmc_path,'leaves.npz')
            np.savez_compressed(leaf_path, data=leaves)

            if 'wandb' in self.config.trainer.logger and self.config.algorithm.bmc.log_artifacts:
                wandb = logger.logger['wandb']
                self.log_file_to_wandb(tree_path, wandb)
                self.log_file_to_wandb(leaf_path, wandb)

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # profiling
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch = None

        #BMC - additional tracking variables:
        checkpoint_path = os.path.join(self.config.trainer.default_local_dir,'bmc')
        os.makedirs(checkpoint_path, exist_ok=True)
        sampled_history = np.lib.format.open_memmap(os.path.join(checkpoint_path,"sampled_history.npy"),dtype="float32",mode="w+",shape=(self.total_training_steps, len(self.train_dataloader.sampler)),)
        mu_history = np.lib.format.open_memmap(os.path.join(checkpoint_path, "mu_history.npy"),dtype="float32",mode="w+",shape=(self.total_training_steps, len(self.train_dataloader.sampler)),)
        lambda_history = np.lib.format.open_memmap(os.path.join(checkpoint_path,"lambda_history.npy"),dtype="float32",mode="w+",shape=(self.total_training_steps, len(self.train_dataloader.sampler)),)
        sigma2_history = np.lib.format.open_memmap(os.path.join(checkpoint_path,"sigma2_history.npy"),dtype="float32",mode="w+",shape=(self.total_training_steps, len(self.train_dataloader.sampler)),)
        staleness_history = np.lib.format.open_memmap(os.path.join(checkpoint_path,"staleness_history.npy"),dtype="float32",mode="w+",shape=(self.total_training_steps, len(self.train_dataloader.sampler)),)
        selection_history = np.lib.format.open_memmap(os.path.join(checkpoint_path,"selection_history.npy"),dtype="float32",mode="w+",shape=(self.total_training_steps, len(self.train_dataloader.sampler)),)
        reward_history = np.lib.format.open_memmap(os.path.join(checkpoint_path,"reward_history.npy"),dtype="float32",mode="w+",shape=(self.total_training_steps, len(self.train_dataloader.sampler)),)
        
        self.round_robin_completion_step = None
        self.train_dataloader.sampler.batch_size = self.config.data.train_batch_size #DM: leftover artifact from DS approach; not needed?
        dataloader_iter = iter(self.train_dataloader)
        while (self.global_steps-1) <= self.total_training_steps: #BMC: forego epoch structure
            if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
            metrics = {}

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            ##BMC:
            #performing tree sampling:
            indices = self.train_dataloader.sampler.sample_batch()
            #print("[DEBUG - Bayesian Sampling] Sampling complete")
            collate_fn = self.train_dataloader.collate_fn
            examples = [self.train_dataloader.dataset[idx] for idx in indices]
            batch_dict = collate_fn(examples)

            #tracking selections
            indexes = batch_dict['sampler_index'].astype(np.int32)
            selections = np.zeros(len(self.train_dataloader.sampler))  
            np.add.at(selections, indexes, 1)
            #END BMC

            new_batch: DataProto = DataProto.from_single_dict(batch_dict)
            gen_batch = self._get_gen_batch(new_batch)
            gen_batch_output = gen_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
            )

            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, "red"):
                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                    #gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                #BMC: mapping uids to tree sampler ids
                new_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in list(new_batch.non_tensor_batch['sampler_index'])], dtype=object
                )
                uid_to_sampler_map = dict(zip(new_batch.non_tensor_batch['uid'], new_batch.non_tensor_batch['sampler_index']))
                #END BMC

                # repeat to align with repeated responses in rollout
                new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                new_batch = new_batch.union(gen_batch_output)

                with marked_timer("reward", timing_raw, "yellow"):
                    # compute scores. Support both model and function-based.
                    # We first compute the scores using reward model. Then, we call reward_fn to combine
                    # the results from reward model and rule-based results.
                    if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                        # we first compute reward model score
                        reward_tensor = self.rm_wg.compute_rm_score(new_batch)
                        new_batch = new_batch.union(reward_tensor)

                    # we combine with rule-based rm
                    reward_tensor, reward_extra_infos_dict = compute_reward(new_batch, self.reward_fn)

                    new_batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        new_batch.non_tensor_batch.update(
                            {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                        )

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        new_batch, kl_metrics = apply_kl_penalty(
                            new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(
                            kl_metrics
                        )  # TODO: This will be cleared if we use multiple genenration batches
                    else:
                        new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]
                    

                # use below to evaluate effective batch ratios (NOT performing dynamic sampling)
                metric_name = self.config.algorithm.filter_groups.metric
                if metric_name == "seq_final_reward":
                    # Turn to numpy for easier filtering
                    new_batch.non_tensor_batch["seq_final_reward"] = (
                        new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                    )
                elif metric_name == "seq_reward":
                    new_batch.non_tensor_batch["seq_reward"] = (
                        new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                    )

                # Collect the sequence reward for each trajectory
                prompt_uid2metric_vals = defaultdict(list)
                for uid, metric_val in zip(
                    new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                ):
                    prompt_uid2metric_vals[uid].append(metric_val)
                
                prompt_uid2metric_std = {}

                # BMC: getting avg prompt varinace per batch
                batch_var = []
                for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                    prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)
                    batch_var.append(np.var(metric_vals))
                avg_prompt_var = np.mean(np.asarray(batch_var))
                metrics['train/avg_prompt_variance'] = float(avg_prompt_var)

                # BMC: getting zero variance ratio statistic
                kept_prompt_uids = [
                    uid
                    for uid, std in prompt_uid2metric_std.items()
                    if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                ]
                metrics["train/zero_variance_ratio"] = 1-(len(kept_prompt_uids) / self.config.data.train_batch_size)
                
                
                ## BMC: tree updation; some artifacts leftover from DS approach
                #print("[DEBUG] Starting bandit section: tree score population")

                # compute advantages (for tree population), executed on the driver process
                #norm_adv_by_std_in_grpo = False 
                scoring_batch = compute_advantage(
                    new_batch,
                    adv_estimator=self.config.algorithm.adv_estimator,
                    gamma=self.config.algorithm.gamma,
                    lam=self.config.algorithm.lam,
                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                    norm_adv_by_std_in_grpo=False, #avoid difficulty bias for sampling only (via Dr. GRPO)
                )
                scoring_batch.batch['advantages'] = scoring_batch.batch['advantages']

                sampler_scores = {}
                for i,uid in enumerate(scoring_batch.non_tensor_batch['uid']):
                    #print("[DEBUG] In scoring batch loop")

                    # getting per-rollout advantages
                    sampler_index = uid_to_sampler_map[uid]
                    if sampler_index not in sampler_scores:
                        sampler_scores[sampler_index] = []
                    if self.config.algorithm.bmc.algo_type=='bmc': #DM-MOPPS
                        sampler_scores[sampler_index].append(scoring_batch.batch['advantages'][i, 0])
                    elif self.config.algorithm.bmc.algo_type=='mopps': #DM-MOPPS
                        reward = scoring_batch.batch['token_level_scores'][i].sum().item() 
                        binary_reward = 1 if reward >= 1.0 else 0  
                        sampler_scores[sampler_index].append(binary_reward)
                    

                ### BMC: updating posteriors for selected prompts
                if self.config.algorithm.bmc.algo_type=='bmc':

                    # arm-invariant variables
                    coverage = (self.train_dataloader.sampler.batch_size / self.train_dataloader.sampler.num_arms)
                    epsilon = 1/(self.config.actor_rollout_ref.rollout.n**2)
                    logit_eps = torch.tensor(logit(1 - epsilon))
                    r_max = abs(logit_eps)
                    sigma2_max = (r_max / self.config.algorithm.bmc.sigma_rule)**2

                    if self.config.algorithm.bmc.enable_curriculum: #if curriculum is enabled
                        for sampler_index in sampler_scores.keys():
                            # getting "reward" for bandit problem: measure of reward variance
                            # expected-absolute-advantage (EAA) doesn't have to be used; could be anything that's roughly proportional to reward variance
                            # in this case, for GRPO/GSPO, EAA is maximized when group relative reward variance is maximized -> convenient signal to use
                            # if rewards are [-1,1], and GRPO is not divided by std, then EAA is in the [0,1] range (hence why norm_adv_by_std_in_grpo is False for sampling)
                            # we strongly recommend that, whatever the "productivity measure" is (reward variance or otherwise), it's in the [0,1] range
                            eaa = torch.tensor(sampler_scores[sampler_index]).abs().mean()  #expected absolute advantage (EAA): [0,1]
                            v = torch.clamp(eaa, min=epsilon, max=1-epsilon) 
                            r = logit(v)

                            # current priors
                            lambda_ = self.train_dataloader.sampler.lambda_[sampler_index].clone()
                            mu = self.train_dataloader.sampler.mu[sampler_index].clone()
                            sigma2 = self.train_dataloader.sampler.sigma2[sampler_index].clone()

                            # getting normalized surprise
                            local_surprise_max = torch.abs(logit_eps) / torch.sqrt(sigma2)
                            surprise = (r - mu) / torch.sqrt(sigma2)
                            surprise = torch.clamp(surprise, min=-local_surprise_max, max=local_surprise_max)

                            ## posterior updates
                            ROUND_ROBIN_COMPLETED = (not self.config.algorithm.bmc.round_robin_init) or (self.config.algorithm.bmc.round_robin_init and self.train_dataloader.sampler.sampled.all())

                            if ROUND_ROBIN_COMPLETED:
                                # local surprise
                                local_squared_surprise = surprise ** 2

                                # lambda update
                                lambda_decay = torch.exp(-local_squared_surprise) # saturating decay factor
                                lambda_effective = torch.clamp(lambda_ * lambda_decay,
                                min=self.config.algorithm.bmc.lambda_min,
                                max=self.config.algorithm.bmc.lambda_max) # apply decay BEFORE adding new evidence

                                lambda_new = torch.clamp(lambda_effective + 1.0,
                                min=self.config.algorithm.bmc.lambda_min,
                                max=self.config.algorithm.bmc.lambda_max)
                                
                                # mu update
                                mu_new = (lambda_effective * mu + r) / lambda_new

                                # sigma2 update:
                                # confidence contraction/increase
                                sigma2_contraction = sigma2 * (lambda_effective / lambda_new) 

                                # uncertainty injection/increase
                                stale_surprise = self.train_dataloader.sampler.staleness[sampler_index]*coverage # increase uncertainty if arm hasn't been sampled in awhile
                                effective_surprise = local_squared_surprise + stale_surprise # surprise is "synthetic": combining staleness + local surprise (or "error")
                                sigma2_injection = torch.log1p(effective_surprise) / lambda_new # uncertainty injection

                                # combine; keep reasonably bounded
                                sigma2_new = torch.clamp(sigma2_contraction + sigma2_injection,
                                min=self.config.algorithm.bmc.sigma2_min, 
                                max=sigma2_max)
                                
                            else:
                                #print('[DEBUG] Restricting update of priors; round-robin in progress...')
                                # lambda update
                                lambda_new = lambda_ + 1

                                # mu update
                                mu_new = (lambda_ * mu + r) / lambda_new

                                # sigma2 update
                                sigma2_new = sigma2

                            # round robin tracking
                            self.train_dataloader.sampler.sampled[sampler_index] = True

                            # execute update
                            self.train_dataloader.sampler.lambda_[sampler_index] = lambda_new
                            self.train_dataloader.sampler.mu[sampler_index] = mu_new
                            self.train_dataloader.sampler.sigma2[sampler_index] = sigma2_new
                            self.train_dataloader.sampler.staleness[sampler_index] = -1 # reset local staleness

                            #print(f"""[DEBUG] Tree Update:
                            #- Reward Statistics for {sampler_index} (eaa, v, r): ({eaa}, {v}, {r})
                            #- Mu Statistics for {sampler_index} (Old, New, Diff): ({mu}, {self.train_dataloader.sampler.mu[sampler_index]}, {mu-self.train_dataloader.sampler.mu[sampler_index]})
                            #- Lambda Statistics for {sampler_index} (Old, New, Diff): ({lambda_}, {self.train_dataloader.sampler.lambda_[sampler_index]}, {lambda_-self.train_dataloader.sampler.lambda_[sampler_index]})
                            #- Sigma2 Statistics for {sampler_index} (Old, New, Diff): ({sigma2}, {self.train_dataloader.sampler.sigma2[sampler_index]}, {sigma2-self.train_dataloader.sampler.sigma2[sampler_index]})
                            #- Surprise Statistics for {sampler_index} (surprise, sigma2, normalized surprise): ({r-mu}, {sigma2}, {surprise})
                            #""")

                        # globally increment staleness
                        self.train_dataloader.sampler.staleness += 1


                        ## per-batch metrics                
                        # mu (belief)
                        metrics['bmc_aux/mu_mean'] = self.train_dataloader.sampler.mu.mean()
                        metrics['bmc_aux/mu_std'] = self.train_dataloader.sampler.mu.std()
                        metrics['bmc_aux/mu_max'] = self.train_dataloader.sampler.mu.max()
                        metrics['bmc_aux/mu_min'] = self.train_dataloader.sampler.mu.min()

                        # lambda (plasticity)
                        metrics['bmc_aux/lambda_mean'] = self.train_dataloader.sampler.lambda_.mean()
                        metrics['bmc_aux/lambda_median'] = self.train_dataloader.sampler.lambda_.median()
                        metrics['bmc_aux/lambda_std'] = self.train_dataloader.sampler.lambda_.std()
                        metrics['bmc_aux/lambda_max'] = self.train_dataloader.sampler.lambda_.max()
                        metrics['bmc_aux/lambda_min'] = self.train_dataloader.sampler.lambda_.min()

                        # sigma2 (uncertainty)
                        global_sigma2 = self.train_dataloader.sampler.sigma2
                        metrics['bmc_aux/sigma2_mean'] = global_sigma2.mean()
                        metrics['bmc_aux/sigma2_median'] = global_sigma2.median()
                        metrics['bmc_aux/sigma2_max'] = global_sigma2.max()
                        metrics['bmc_aux/sigma2_min'] = global_sigma2.min()
                        metrics['bmc_aux/sigma2_q05'] = global_sigma2.quantile(0.05)
                        metrics['bmc_aux/sigma2_q95'] = global_sigma2.quantile(0.95)
                        metrics['bmc_aux/sigma2_frac_min'] = (global_sigma2 <= self.config.algorithm.bmc.sigma2_min + 1e-6).float().mean()
                        metrics['bmc_aux/sigma2_frac_max'] = (global_sigma2 >= sigma2_max - 1e-6).float().mean()

                        # staleness
                        metrics['bmc_aux/staleness_mean'] = self.train_dataloader.sampler.staleness.mean()
                        metrics['bmc_aux/staleness_median'] = self.train_dataloader.sampler.staleness.median()
                        metrics['bmc_aux/staleness_max'] = self.train_dataloader.sampler.staleness.max()
                        metrics['bmc_aux/staleness_min'] = self.train_dataloader.sampler.staleness.min()

                        # track for round-robin
                        if self.config.algorithm.bmc.round_robin_init:
                            if self.train_dataloader.sampler.sampled.all():
                                if self.round_robin_completion_step == None:
                                    self.round_robin_completion_step = self.global_steps
                                    #DM: ADD CHECKPOINT
                                else:
                                    print(f'[DEBUG] Round-robin completed @ step={self.round_robin_completion_step}')
                        
                            metrics["train/round_robin_progress"] = self.train_dataloader.sampler.sampled.sum() / self.train_dataloader.sampler.sampled.numel()
                    else: #if curriculum is disabled
                        for sampler_index in sampler_scores.keys():
                            self.train_dataloader.sampler.sampled[sampler_index] = True
                        print(f"[DEBUG] Curriculum disabled; skipping arm belief updates and logging")

                elif self.config.algorithm.bmc.algo_type=='mopps':
                    #NOTE: MoPPS differs from BMC in that they appear to NOT decay for all arms; they just decay for those selected
                    for sampler_index in sampler_scores.keys():
                        # debug
                        #print(f'[DEBUG] Alpha/Beta for {sampler_index}  (before): {self.train_dataloader.sampler.alpha[sampler_index]}/{self.train_dataloader.sampler.beta[sampler_index]}')
                        #print(f'[DEBUG] Sampler scores for {sampler_index}: {torch.tensor(sampler_scores[sampler_index]).sum()}; len=={len(sampler_scores[sampler_index])}')

                        # apply decay
                        MOPPS_PRIOR = 1.0 # default prior from paper
                        MOPPS_LAMBDA = 0.5 # default value from paper; their ablations also tested 1 and 0
                        self.train_dataloader.sampler.alpha[sampler_index] = MOPPS_PRIOR + (self.train_dataloader.sampler.alpha[sampler_index] - MOPPS_PRIOR) * MOPPS_LAMBDA
                        self.train_dataloader.sampler.beta[sampler_index]  = MOPPS_PRIOR  + (self.train_dataloader.sampler.beta[sampler_index]  - MOPPS_PRIOR)  * MOPPS_LAMBDA

                        # add evidence
                        self.train_dataloader.sampler.alpha[sampler_index] += torch.tensor(sampler_scores[sampler_index]).sum()
                        self.train_dataloader.sampler.beta[sampler_index] += len(sampler_scores[sampler_index]) - torch.tensor(sampler_scores[sampler_index]).sum()
                        #print(f'[DEBUG] Alpha/Beta for {sampler_index} (after): {self.train_dataloader.sampler.alpha[sampler_index]}/{self.train_dataloader.sampler.beta[sampler_index]}')
                    metrics['mopps/alpha_mean'] = self.train_dataloader.sampler.alpha.mean()
                    metrics['mopps/beta_mean'] = self.train_dataloader.sampler.beta.mean()
                ## END BMC

                batch = new_batch if batch is None else DataProto.concat([batch, new_batch]) #DM: leftover logic from DAPO; could be removed later
                    
                # === Updating ===
                #print(f'[DEBUG] Batch formed; Size = {len(batch)}')
                batch.batch["response_mask"] = compute_response_mask(batch)                        

                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                # TODO: Decouple the DP balancing and mini-batching.
                if self.config.trainer.balance_batch:
                    self._balance_batch(batch, metrics=metrics)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                # recompute old_log_probs
                with marked_timer("old_log_prob", timing_raw, "blue"):
                    old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    entropys = old_log_prob.batch["entropys"]
                    response_masks = batch.batch["response_mask"]
                    loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                    entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                    old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                    metrics.update(old_log_prob_metrics)
                    old_log_prob.batch.pop("entropys")
                    batch = batch.union(old_log_prob)
                
                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer("ref", timing_raw, "olive"):
                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, "cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)


                # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                    batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                    # IS and off-policy metrics already have rollout_corr/ prefix
                    metrics.update(is_metrics)

                with marked_timer("adv", timing_raw, "brown"):
                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                    )
                
                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, "pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # validate
                validate_freq = self.config.trainer.test_freq if self.global_steps < self.config.trainer.dense_cutoff else self.config.trainer.sparse_test_freq
                pprint(f'Validation frequency ={validate_freq}')
                if (  
                    self.val_reward_fn is not None  
                    and self.config.trainer.test_freq > 0  
                    and (is_last_step or self.global_steps % validate_freq == 0)  
                ):  
                    with marked_timer("testing", timing_raw, "green"):
                        if self.global_steps < self.config.trainer.dense_cutoff:
                            pprint(f'Dense validation')
                            
                            # MB Custom logic: odd iterations = test set, even iterations = train pass@1  
                            if self.global_steps % (self.config.trainer.test_freq * 2) == self.config.trainer.test_freq:  
                                # Odd iterations (10, 30, 50...)  
                                val_metrics: dict = self.validate_train(reward_history)  
                            else:  
                                # Even iterations (20, 40, 60...)
                                val_metrics: dict = self._validate()
                        else:
                            pprint(f'Sparse validation')
                            test_val_metrics: dict = self._validate()
                            train_val_metrics: dict = self.validate_train(reward_history) 
                            val_metrics = {**test_val_metrics, **train_val_metrics}
                        
                        if is_last_step:  
                            last_val_metrics = val_metrics  
                    metrics.update(val_metrics)

                #BMC: write priority scores and selection stats to files for checkpointing and analysis
                sampled_history[self.global_steps, :] = self.train_dataloader.sampler.sampled.numpy()
                sampled_history.flush()

                mu_history[self.global_steps, :] = self.train_dataloader.sampler.mu.numpy()
                mu_history.flush()

                lambda_history[self.global_steps, :] = self.train_dataloader.sampler.lambda_.numpy()
                lambda_history.flush()

                sigma2_history[self.global_steps, :] = self.train_dataloader.sampler.sigma2.numpy()
                sigma2_history.flush()

                staleness_history[self.global_steps, :] = self.train_dataloader.sampler.staleness.numpy()
                staleness_history.flush()

                selection_history[self.global_steps, :] = selections
                selection_history.flush()
                #END BMC

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                        #try:
                        #load wandb run
                        wandb = logger.logger['wandb']

                        bmc_path = os.path.join(self.config.trainer.default_local_dir,'bmc')

                        #create compressed data
                        sampled_data = np.load(os.path.join(bmc_path,"sampled_history.npy"), mmap_mode='r')
                        np.savez_compressed(os.path.join(bmc_path,"sampled_history_compressed.npz"), data=sampled_data)

                        mu_data = np.load(os.path.join(bmc_path,"mu_history.npy"), mmap_mode='r') 
                        np.savez_compressed(os.path.join(bmc_path,"mu_history_compressed.npz"), data=mu_data)

                        lambda_data = np.load(os.path.join(bmc_path,"lambda_history.npy"), mmap_mode='r') 
                        np.savez_compressed(os.path.join(bmc_path,"lambda_history_compressed.npz"), data=lambda_data)

                        sigma2_data = np.load(os.path.join(bmc_path,"sigma2_history.npy"), mmap_mode='r') 
                        np.savez_compressed(os.path.join(bmc_path,"sigma2_history_compressed.npz"), data=sigma2_data)

                        staleness_data = np.load(os.path.join(bmc_path,"staleness_history.npy"), mmap_mode='r') 
                        np.savez_compressed(os.path.join(bmc_path,"staleness_history_compressed.npz"), data=staleness_data)

                        selection_data = np.load(os.path.join(bmc_path,"selection_history.npy"), mmap_mode='r') 
                        np.savez_compressed(os.path.join(bmc_path,"selection_history_compressed.npz"), data=selection_data)


                        reward_data = np.load(os.path.join(bmc_path,"reward_history.npy"), mmap_mode='r') 
                        np.savez_compressed(os.path.join(bmc_path,"reward_history_compressed.npz"), data=reward_data)
                        
                        #log artifacts to wandb (old; not used anymore, but keeping just in case)
                        if self.config.algorithm.bmc.log_artifacts:
                            self.log_file_to_wandb(os.path.join(bmc_path,"sampled_history_compressed.npz"), wandb)
                            self.log_file_to_wandb(os.path.join(bmc_path,"mu_history_compressed.npz"), wandb)
                            self.log_file_to_wandb(os.path.join(bmc_path,"lambda_history_compressed.npz"), wandb)
                            self.log_file_to_wandb(os.path.join(bmc_path,"selection_history_compressed.npz"), wandb)
                            self.log_file_to_wandb(os.path.join(bmc_path,"sigma2_history_compressed.npz"), wandb)
                            self.log_file_to_wandb(os.path.join(bmc_path,"staleness_history_compressed.npz"), wandb)

                        #except Exception as e:
                        #    print(f'[WARNING] Failed to log artifacts to wandb; continuing run...; Exception={e}')

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            timing_raw = defaultdict(float)  # clear timing
            batch = None

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return

            # bottom-up update (for parents)
            if self.config.algorithm.bmc.ltt.enable:
                self.train_dataloader.sampler.update_tree()
            #END BMC
            progress_bar.update(1)
            self.global_steps += 1
            #print("[DEBUG] End of optimizer step\n\n")
    
    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid", "sampler_index"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def validate_train(self, reward_history=None):  
        """Evaluate pass@1 on entire training set and store reward history"""  
        # Save original validation settings    
        original_val_dataloader = self.val_dataloader    
        original_n = self.config.actor_rollout_ref.rollout.val_kwargs.n    
    
        # Create dataloader for full training dataset            
        from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn  
        collate_fn = default_collate_fn  
        train_dataloader = StatefulDataLoader(    
            dataset=self.train_dataset,    
            batch_size=len(self.train_dataset),  # Full dataset    
            shuffle=False,    
            collate_fn=collate_fn,    
        )    
    
        # Temporarily replace validation dataloader and set n=1    
        self.val_dataloader = train_dataloader    
        self.config.actor_rollout_ref.rollout.val_kwargs.n = 1    
    
        try:    
            # Use existing _validate logic with n=1    
            metrics = self._validate()    
            
            # Store scores in reward_history if provided  
            if reward_history is not None and self.global_steps < reward_history.shape[0]:  
                # Extract scores from the validation process  
                scores = self._extract_validation_scores()  
                
                # Store at current timestep  
                reward_history[self.global_steps] = scores
                #reward_history[self.global_steps, :len(scores)] = scores

                reward_history.flush()  
                
            # Add prefix to distinguish metrics    
            return {f"train_pass1/{k}": v for k, v in metrics.items()}    
        finally:    
            # Restore original settings    
            self.val_dataloader = original_val_dataloader    
            self.config.actor_rollout_ref.rollout.val_kwargs.n = original_n  
    
    # DM: note that extract_reward() is in verl/trainer/ppo/reward.py in newer verl versions; adding it here for now
    def extract_reward(self,batch: DataProto):
        """
        Extract reward tensor and extra info from batch data.
        """
        reward_tensor = batch.batch["rm_scores"]
        reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
        reward_extra_infos_dict = {key: batch.non_tensor_batch[key] for key in reward_extra_keys}
        return reward_tensor, reward_extra_infos_dict

    def _extract_validation_scores(self):  
        """Extract scores from validation process"""  
        scores = []  
        
        for test_data in self.val_dataloader:  
            test_batch = DataProto.from_single_dict(test_data)  
    
            if "uid" not in test_batch.non_tensor_batch:  
                test_batch.non_tensor_batch["uid"] = np.array(  
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object  
                )  
    
            # repeat test batch  
            test_batch = test_batch.repeat(  
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True  
            )  
    
            test_gen_batch = self._get_gen_batch(test_batch)  
            test_gen_batch.meta_info = {  
                "eos_token_id": self.tokenizer.eos_token_id,  
                "pad_token_id": self.tokenizer.pad_token_id,  
                "recompute_log_prob": False,  
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,  
                "validate": True,  
                "global_steps": self.global_steps,  
            }  
    
            # pad to be divisible by dp_size  
            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers  
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)  
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)  
    
            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():  
                self.checkpoint_manager.sleep_replicas()  
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)  
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)  
                self.checkpoint_manager.update_weights(self.global_steps)  
    
            # unpad  
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)  
    
            test_batch = test_batch.union(test_output_gen_batch)  
            test_batch.meta_info["validate"] = True  
    
            # evaluate using reward_function  
            reward_tensor, reward_extra_info = self.extract_reward(test_batch)  
            batch_scores = reward_tensor.sum(-1).cpu().tolist()  
            scores.extend(batch_scores)  
        
        return scores


    def log_file_to_wandb(self, file_path, wandb):
        run = wandb.run

        filename = os.path.basename(file_path)
        artifact = wandb.Artifact(name=filename, type="dataset")
        artifact.add_file(file_path)
        run.log_artifact(artifact, aliases=["latest"])
        artifact.wait()


    def _build_ckpt_handler(self):  
        from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode  
        from bmc_checkpoint_handler import BMCCheckpointHandler 
        
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")  
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)  
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)  
        default_hdfs_dir = getattr(self.config.trainer, "default_hdfs_dir", None)  
    
        self.ckpt_handler = BMCCheckpointHandler(  
            engine=self.actor_rollout_wg,
            train_dataloader=self.train_dataloader,  
            default_local_dir=self.config.trainer.default_local_dir,  
            max_ckpt_to_keep=max_ckpt_to_keep,  
            default_hdfs_dir=default_hdfs_dir,  
            resume_mode=resume_mode,  
            resume_from_path=resume_from_path,  
            mode=OrchestrationMode.RAY, 
        )

    def _load_checkpoint(self):  
        # overriding PPO's normal checkpoint loader
        resume_step = self.ckpt_handler.load_checkpoint()
        self.global_steps = resume_step
        return resume_step
    
    def _save_checkpoint(self):
        self.ckpt_handler.save_checkpoint(self.global_steps)