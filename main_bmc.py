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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_object
from verl.trainer.ppo.ray_trainer import Role 


#LTT specific imports
from verl.trainer.main_ppo import TaskRunner
from bmc_ray_trainer import RayBMCTrainer
from ipdb import set_trace
import time
import torch
import torch.nn.functional as F
from scipy.stats import normal_inverse_gamma
from pathlib import Path
import pandas as pd
import json

@hydra.main(config_path="dapo_config", config_name="dapo_trainer", version_base=None)
def main(config):
    run_bmc(config)

# Define a function to run the PPO-like training process
def run_bmc(config, task_runner_class=None) -> None:
    print("[DEBUG] In run_bmc()")
    from numpy import savez
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.fs import copy_to_local

    if config.algorithm.bmc.ltt.enable: # if LTT is enabled
        local_path = copy_to_local(config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False))
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        if not config.algorithm.bmc.targeting:
            tree_dataset = create_rl_dataset(config.data.train_files, config.data, tokenizer, processor) #ORIGINAL
        else:
            print("[DEBUG] Targeting enabled; loading both train and test data for tree")
            # Load and add labels (assuming unified file for train and eval)
            train_df = pd.read_parquet(config.data.train_files)
            train_df['targets'] = 0
            train_df["target_weights"] = 0 # doesn't matter; just make fit with target_df
            train_df["extra_info"] = [{"index": 0} for _ in range(len(train_df))]

            target_df = pd.read_parquet(config.data.target_files)
            source_counts = {}
            for source in target_df['data_source']:
                if source not in source_counts:
                    source_counts[source] = 0
                source_counts[source] += 1
            target_df['targets'] = 1
            target_df["target_weights"] = [1/source_counts[target_df['data_source'][x]] for x in range(len(target_df))]
            target_df["extra_info"] = [{"index": 0} for _ in range(len(target_df))]

            tree_df = pd.concat([train_df, target_df], ignore_index=True)

            #DM: for multi-file version? don't worry about for now
            """
            for datafile in config.data.train_files:
                df = pd.read_parquet(datafile)  
                df["target"] = 0
                dfs.append(df)
            for datafile in config.data.val_files:
                df = pd.read_parquet(datafile)
                df["target"] = 1
                dfs.append(df)
            """
            tree_df.to_parquet("tree_data.parquet")           
            tree_dataset = create_rl_dataset(["tree_data.parquet"], config.data, tokenizer, processor)

        if not config.algorithm.bmc.resume_from_checkpoint:
            latents, targets, target_weights = extract_latents_distributed(config.actor_rollout_ref.model.path, tree_dataset, 
            config.algorithm.bmc.ltt.layer_depth, config.data.train_batch_size*config.algorithm.bmc.ltt.latent_batch_factor,
            config.trainer.n_gpus_per_node, config.algorithm.bmc.targeting)

            if targets is not None:
                savez("latents.npz", latents=latents, targets=targets, target_weights=target_weights)
            else:
                savez("latents.npz", latents=latents)
            

        #config.trainer.n_gpus_per_node
        #dataset.dataframe.save_to_disk("dataset.hf") #DM: HARDCODED ARGUMENT (is this even needed? why did i put this here?)
        print("[DEBUG] Pretask setup complete")
        # After this, run the rest of the code "as normal"

    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunnerBMC)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class TaskRunnerBMC(TaskRunner):
    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local
        from numpy import load

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)

        # We should adopt a multi-source reward function here:
        # - for rule-based rm, we directly call a reward score
        # - for model-based rm, we call a model
        # - for code related prompt, we send to a sandbox if there are test cases
        # finally, we combine all the rewards together
        # The reward type depends on the tag of the data
        self.add_reward_model_worker(config)

        # Add a reference policy worker if KL loss or KL reward is used.
        self.add_ref_policy_worker(config, actor_rollout_cls)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(self.role_worker_mapping),
            use_critic=need_critic(config),
        )

        # BMC: add Latent
        self.role_worker_mapping[Role.LatentExtractor] = LatentExtractor
        self.mapping[Role.LatentExtractor] = "global_pool" 


        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation.
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = self.init_resource_pool_mgr(config)

        from verl.utils.dataset.rl_dataset import collate_fn

        # BMC-T-VAL-1: 
        # only the original *train_dataset* is used and passed into the BanditSampler
        # the sampler does NOT see the *tree_dataset*, which contains the train and target data
        # the tree_dataset is only used to get the target indices
        # DO NOT PASS IN THE TREE DATASET BELOW; even if there are downstream bugs, don't risk training on the test set, please...

        # Create training and validation datasets.
        train_dataset = create_rl_dataset( 
            config.data.train_files, 
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )

        if config.algorithm.bmc.treeviz_only==False:
            val_dataset = create_rl_dataset(
                config.data.val_files,
                config.data,
                tokenizer,
                processor,
                is_train=False,
                max_samples=config.data.get("val_max_samples", -1),
            )

        if config.algorithm.bmc.ltt.enable and not config.algorithm.bmc.resume_from_checkpoint:
            embeddings = load("latents.npz")
        else:
            embeddings = None
        train_sampler = create_rl_sampler(config, train_dataset, embeddings, tokenizer) 
        if config.algorithm.bmc.ltt.enable:
            del embeddings
        
        if config.algorithm.bmc.treeviz_only:
            print(f'Tree visualization completed. Analysis files stored at the following path/folder: {config.algorithm.bmc.analysis_file}')
            return
        


        # Initialize the PPO trainer.
        trainer = RayBMCTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        trainer.fit()


#def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset.

    Arguments:
        data_paths: List of paths to data files.
        data_config: The data config.
        tokenizer (Tokenizer): The tokenizer.
        processor (Processor): The processor.

    Returns:
        dataset (Dataset): The dataset.
    """
    from torch.utils.data import Dataset
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from numpy import load

    class RLHFDataset_BMC(RLHFDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # for handling no template by default cases
            qwen_chat_template = """{%- for message in messages %}
            <|im_start|>{{ message['role'] }}
            {{ message['content'] }}<|im_end|>
            {%- endfor %}
            {%- if add_generation_prompt %}
            <|im_start|>assistant
            {%- endif %}
            """

            if getattr(self.tokenizer, "chat_template", None) is None:
                self.tokenizer.chat_template = qwen_chat_template

            if self.processor is not None:
                if not getattr(self.processor, "chat_template", None):
                    self.processor.chat_template = self.tokenizer.chat_template

        def __getitem__(self, item):
            row_dict = super().__getitem__(item)
            row_dict['sampler_index'] = item #need to get original indicies to update sampling tree

            return row_dict

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        from verl.utils.import_utils import load_extern_type

        # Dynamically load the custom dataset class
        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(f"The custom dataset class '{data_config.custom_cls.name}' from '{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset")
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset_BMC
    print(f"Using dataset class: {dataset_cls.__name__}")

    # Instantiate the dataset using the determined dataset class
    dataset = dataset_cls(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
    )
    return dataset    


def create_rl_sampler(config, dataset, latents=None, tokenizer=None):
    """Create a sampler for the dataset.

    Arguments:
        config: The original config.
        dataset (Dataset): The dataset.

    Returns:
        sampler (Sampler): The sampler.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import Sampler, RandomSampler, SequentialSampler, BatchSampler
    import numpy as np
    import pickle
    from scipy.special import logit, expit

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from skdim.id import TwoNN
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors
    import umap.umap_ as umap
    import hdbscan

    import matplotlib.pyplot as plt
    import os
    import warnings
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module="sklearn.utils.deprecation"
    )

    
    class BanditSampler(Sampler):
        def __init__(self, config, data_source, batch_size, latents, drop_last=True, replace=True):
            super().__init__(dataset)

            # Get config, dataset, and LLM
            self.config = config
            self.dataset = dataset
            self.batch_size = batch_size
            self.num_arms = len(dataset)
            self.replace=replace
            self.drop_last = drop_last

            self.use_tree = self.config.algorithm.bmc.ltt.enable
            self.use_curriculum = self.config.algorithm.bmc.enable_curriculum

            ## Calculate latents and statistics
            self.latents = latents

            #DM: self.targets isn't used for anywhere but for print statements and debugging
            # see: self.latents['targets']; line 812 instead
            if config.algorithm.bmc.targeting:
                self.targets = torch.from_numpy(latents['targets'])

                # DM: not taking chances...

                # BMCT-VAL-2: checking if self.num_arms (which determines the valid indexes sampled) is equal to the TRAINING DATA only
                # the concatenated train+target data SHOULD NOT have been passed in to BanditSampler;
                # thus, self.num_arms should be strictly less than len(self.targets)
                if self.num_arms >= len(self.targets):
                    raise ValueError("""
                                    !!!WARNING!!!
                                    BMC-T is being used, but the length of the training dataset is equal or less thanself.targets when it should be strictly greater.
                                    It is highly likely you are training on the test set (somehow). Either that, or you passed in an empty target set?
                                    Please re-validate.
                                    !!!WARNING!!!
                                     """)
            else:
                # DM: i forgot why i put this here (dummy value?); self.targets anywhere isn't used if targeting is disabled
                self.targets = torch.zeros(self.num_arms, dtype=torch.bool) 



            if "VL" in str(self.config.actor_rollout_ref.model.path) or "vision" in str(self.config.actor_rollout_ref.model.path).lower():
                self.model_type = "VLM"
            else:
                self.model_type = "LLM"

            # Setting up analysis file
            self.analysis_file = self.config.algorithm.bmc.analysis_file
            analysis_dir = Path(self.analysis_file)
            analysis_dir.mkdir(parents=True, exist_ok=True)
            
            ## Bayesian Curriculum Parameters
            # Sampling Strategy
            self.sampled = torch.zeros(self.num_arms, dtype=torch.bool) # if arm has been pulled / prompt has been sampled
            self.staleness = torch.full((self.num_arms,), 0.0)

            if self.config.algorithm.bmc.algo_type=='bmc': #DM-MOPPS
                ## Hyperpriors
                self.mu = torch.full((self.num_arms,), 0.0) # all arms map to 0.5 belief initially 
                self.lambda_ = torch.full((self.num_arms,), 1.0) # "psuedo-counts" for (weakly informed)

                # calculating initial sigma2:
                epsilon = 1/(self.config.actor_rollout_ref.rollout.n**2)
                logit_eps = torch.tensor(logit(1 - epsilon))
                self.r_max = abs(logit_eps)
                sigma2_init = (self.r_max / (self.config.algorithm.bmc.sigma_rule+0.5))**2 #sigma_rule=3
                self.sigma2 = torch.full((self.num_arms,), sigma2_init)
            
                self.penultimates = {}
                self.invalid_train_ids = []
                if self.config.algorithm.bmc.ltt.enable and not self.config.algorithm.bmc.resume_from_checkpoint:
                    self.clusters, self.tree = self.create_tree(self.latents)
                    self.initialize_child_to_parent_map()
                    self.update_tree()

                    if config.algorithm.bmc.targeting:
                        self.init_utilities()
                        self.update_tree()

                    if config.algorithm.bmc.targeting:
                        print(f"[DEBUG] Task tree:")
                        for id_ in self.tree.keys():
                            print(f"{id_}: {self.tree[id_][0]}; Training Data={int(self.node_train_count[id_])}; Test Data={self.node_target_count[id_]}/{int(self.targets.sum())} Utility={self.tree[id_][4]}")
                        
                        print(f'\n[DEBUG] Penultimate Nodes:')
                        for id_ in self.penultimates:
                            print(f'{id_}: Training Data={int(self.node_train_count[id_])} Test Data={int(self.node_target_count[id_])}/{int(self.targets.sum())} Utility={self.penultimates[id_][4]}')

                    self.latents = None
                    del latents

            elif self.config.algorithm.bmc.algo_type=='mopps':
                print('Using MoPPS (only storing alpha and beta)')
                self.alpha = torch.ones(self.num_arms, dtype=torch.float32)
                self.beta = torch.ones(self.num_arms, dtype=torch.float32)

            print("[DEBUG] Initialization complete")
        

        def create_tree(self,latents_,):
            print("[DEBUG] IN create_tree()")

            start_time = time.perf_counter()

            latents = dict(latents_).copy()
            N = latents['latents'].shape[0] #ORIG

            #N = self.num_arms
            #T = latents.shape[0] - self.num_arms

            true_clusters = np.zeros(N, dtype=int)
            child_utilities = np.zeros(N, dtype=int)
            task_tree = {}

            next_cluster_id = [1] # mutable counter
            self.recursion_ends = [0, 0, 0] # track recursion ending conditions. 0: locally euclidean, 1: not enough clusters, 2: size too small

            self.rendered_clusters = {}
            ## Recursion function
            def recursive_split(indices, parent_id):
                nonlocal next_cluster_id, true_clusters, task_tree
                selected_latents = latents['latents'][indices]
                n_local = len(selected_latents)

                if self.config.algorithm.bmc.analyze:
                    self.display_word_clouds(true_clusters)
                print(f'\n\n[DEBUG - Tree Creation] Performing recursive_split for Cluster #{parent_id}; size={n_local}')
                
                ## Minimum size check
                min_allowed = int(self.config.algorithm.bmc.ltt.hdbscan_min_cluster_pct * N)
                if n_local < min_allowed:
                    print(f'[DEBUG - Tree Creation] RECURSION END 2 (size too small); size={n_local})')
                    self.recursion_ends[2] += 1
                    return


                ## Standardization
                latent_std = StandardScaler().fit_transform(selected_latents)

                ## PCA
                pca_full = PCA(svd_solver='full',
                            random_state=self.config.algorithm.bmc.ltt.seed)
                pca_full.fit(latent_std)

                cumvar = np.cumsum(pca_full.explained_variance_ratio_)
                n_components = min(
                    np.searchsorted(cumvar, self.config.algorithm.bmc.ltt.pca_explained_variance) + 1,
                    latent_std.shape[1]
                )
                print(f'[DEBUG - Tree Creation] PCA Components: {n_components}')

                latent_pca = PCA(
                    n_components=n_components,
                    svd_solver='full',
                    random_state=self.config.algorithm.bmc.ltt.seed
                ).fit_transform(latent_std)

                ## Chart Test
                # Test #1: kNN connectivity gate (run first to save time--TwoNN is much slower)
                graph_connected = True

                if n_local >= self.config.algorithm.bmc.ltt.knn_min_points_connectivity:
                    graph = kneighbors_graph(
                        latent_pca,
                        n_neighbors=self.config.algorithm.bmc.ltt.knn_connectivity_threshold,
                        mode="connectivity",
                        include_self=True,
                    )
                    n_components_graph, _ = connected_components(graph)
                    graph_connected = (n_components_graph == 1)
                else:
                    n_components_graph = -1  # "skipped"

                twoNN_dim_threshold = self.config.algorithm.bmc.ltt.twonn_dim_threshold
                twoNN_dim = None
                twoNN_ok = False
                if not graph_connected:
                    print(f"[DEBUG] Chart Test: connectivity FAIL (components={n_components_graph}) -> continue to UMAP/HDBSCAN")
                    #pass
                else:
                    # Test #2: TwoNN (only if connectivity passes; could probably be made faster)
                    X = np.asarray(latent_pca, dtype=np.float64)

                    ## Addressing div by 0 issues caused by PCA duplicates:
                    # Get r1, r2 for tie filtering 
                    nn = NearestNeighbors(n_neighbors=3).fit(X) # detect pairs that cause duplicates / div by 0
                    dists, _ = nn.kneighbors(X)
                    r1 = dists[:, 1]
                    r2 = dists[:, 2]
                    valid = (r1 > 0) & np.isfinite(r1) & np.isfinite(r2)
                    Xv = X[valid] # get valid PCA subset
                    #zero_frac = float(np.mean(~valid))

                    if Xv.shape[0] > self.config.algorithm.bmc.ltt.hdbscan_min_min_samples:
                        twoNN_dim = float(TwoNN().fit_transform(Xv))
                        twoNN_ok = (twoNN_dim <= self.config.algorithm.bmc.ltt.twonn_dim_threshold)
                    #DM: is below needed? not sure we want to force an end here; for now, just skip
                    #    except Exception as e:
                    #        print(f"[DEBUG] TwoNN failed even after filtering ties: {repr(e)}")
                    #else:
                    #    print(f"[DEBUG] TwoNN skipped (too many ties). zero_frac={zero_frac:.3%}, valid={Xv.shape[0]}/{n_local}")

                print(f"[DEBUG] Chart Test\n"
                    f"\tConnectivity: True / {n_components_graph}\n"
                    f"\tTwoNN: {twoNN_ok} / {twoNN_dim}")

                if twoNN_dim is not None and twoNN_ok:
                    print("[DEBUG] RECURSION END (chart test passed)")
                    self.recursion_ends[0] += 1
                    return
                

                ## UMAP
                # try to make projections from a consistent proportion of PCA embeddings
                # in practice, embeddings are too big, and we need to cap the dimension so HDBSCAN can cluster
                umap_dim = min(max(2, round(self.config.algorithm.bmc.ltt.umap_proportion * n_components)),
                 self.config.algorithm.bmc.ltt.umap_max_dim)

                latent_umap = umap.UMAP(
                    n_components=umap_dim,
                    min_dist=self.config.algorithm.bmc.ltt.umap_min_dist,
                    n_neighbors=self.config.algorithm.bmc.ltt.umap_n_neighbors,
                    metric='euclidean',
                    n_jobs=self.config.trainer.n_cpu_cores,
                    random_state=self.config.algorithm.bmc.ltt.seed,
                ).fit_transform(latent_pca)
                print(f"[DEBUG - Tree Creation] UMAP Components: {umap_dim}")

                ## HDBSCAN
                min_cluster_pct = self.config.algorithm.bmc.ltt.hdbscan_min_cluster_pct
                min_cluster_size = max(round(min_cluster_pct * n_local), round(min_cluster_pct * N))
                min_samples = max(self.config.algorithm.bmc.ltt.hdbscan_min_min_samples, int(np.sqrt(min_cluster_size))) # sqrt of min_cluster_size

                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=min_cluster_size,
                    min_samples=min_samples,
                    core_dist_n_jobs=self.config.trainer.n_cpu_cores,
                    metric='euclidean',
                    cluster_selection_method='eom'
                ).fit(latent_umap)
                labels = clusterer.labels_
                unique_clusters = sorted(set(labels[labels != -1]))


                print(f'[DEBUG - Tree Creation] HDBSCAN:\n'
                f'\tNum Clusters: {len(unique_clusters)}\n'
                f'\tLocal Clusters IDs: {unique_clusters}\n'
                f'\tLocal Cluster Persistences {clusterer.cluster_persistence_}')

                # Not enough structure
                if len(unique_clusters) <= 1:
                    print(f"[DEBUG - Tree Creation] RECURSION END 1 (not enough clusters, n < 2)")
                    self.recursion_ends[1] += 1
                    return
                
                ## Leftover / "noise" cluster handling 
                covered_mask = np.zeros(n_local, dtype=bool)
                for lbl in unique_clusters:
                    covered_mask |= (labels == lbl)

                has_leftover = np.any(~covered_mask)

                if has_leftover:
                    leftover_id = next_cluster_id[0]
                    next_cluster_id[0] += 1
                    print(f"[DEBUG - Tree Creation] Creating leftover_id={leftover_id}")

                    # Assign only leftover points
                    leftover_indices = indices[~covered_mask]
                    true_clusters[leftover_indices] = leftover_id

                    # Add leftover as child of parent
                    if parent_id not in task_tree:
                        task_tree[parent_id] = [[],None,None,None,0.0]
                        #utility = latents['targets'][indices].sum() / latents['targets'].sum()
                        #print(f"[DEBUG] Utility for (leftover) Cluster #{parent_id}: {latents['targets'][indices].sum()}/{latents['targets'].sum()} = {utility}")
                        #task_tree[parent_id][4] = utility

                    task_tree[parent_id][0].append(leftover_id)

                ## Process children
                for lbl in unique_clusters:
                    mask = (labels == lbl)
                    child_indices = indices[mask]

                    new_id = next_cluster_id[0]
                    next_cluster_id[0] += 1

                    true_clusters[child_indices] = new_id

                    if parent_id not in task_tree:
                        task_tree[parent_id] = [[],None,None,None,0.0]
                        #print("[DEBUG-TEST] Utility-calc (parent ver.)")
                        # utility = (targets in cluster) / (total targets)
                        #utility = latents['targets'][indices].sum() / latents['targets'].sum()
                        #print(f"[DEBUG] Utility for Cluster #{parent_id}: {latents['targets'][indices].sum()}/{latents['targets'].sum()} = {utility}")
                        #task_tree[parent_id][4] = utility

                    task_tree[parent_id][0].append(new_id)

                    print(f"[DEBUG - Tree Creation] Creating new child cluster #{new_id}; size={len(true_clusters[child_indices])}")
                    print(f"[DEBUG - Tree Creation] Current Task Tree: {task_tree}")

                    recursive_split(child_indices, new_id)

            ## Initial recursion
            recursive_split(np.arange(N), 0)

            elapsed = time.perf_counter() - start_time
            total_recursion_ends = torch.tensor(self.recursion_ends).sum().item()
            print(f"[DEBUG] Tree creation finished in {elapsed:.3f}s")
            #print(f"[DEBUG] Task tree: {task_tree}")

            print(f'[DEBUG] Recursion Statistics -\n'
            f'\tChart Test: {(self.recursion_ends[0]/total_recursion_ends)*100}%\n'
            f'\tLack of Clusters: {(self.recursion_ends[1]/total_recursion_ends)*100}%\n'
            f'\tLack of Size: {(self.recursion_ends[2]/total_recursion_ends)*100}%')

            if self.config.algorithm.bmc.analyze:
                torch.save(task_tree, f"{self.analysis_file}/analysis_tree.pt")
                np.save(f"{self.analysis_file}/analysis_clusters.npy", true_clusters)
            del latents
            return torch.from_numpy(true_clusters), task_tree

        def initialize_child_to_parent_map(self):
            self.child_to_parent_map = {}
            for cluster_id in self.tree.keys():
                for child_id in self.tree[cluster_id][0]:
                    if child_id not in self.tree.keys():
                        self.child_to_parent_map[child_id] = cluster_id


        def update_tree(self):
            #print("[DEBUG] In update_tree():")
            root_id = 0 
            self.subtree_update(root_id) # begin recursive update
            print("[DEBUG] Tree updated!")
                 
        def subtree_update(self, root_id): #note: "root_id" is relative; it's not the absolute root of the tree
            #print(f'[DEBUG - Subtree Update] Subtree update call for Node {root_id}')
            if root_id in self.tree: #non-leaf node
                child_mu = []
                child_sigma2 = []

                invalid = set(self.invalid_train_ids)
                child_ids = [cid for cid in self.tree[root_id][0] if cid not in invalid]
                for n, child_id in enumerate(child_ids): #DM-POI
                    mu, sigma2, _ = self.subtree_update(child_id)
                    child_mu.append(mu)
                    child_sigma2.append(sigma2)

                child_mu = torch.stack(child_mu)
                child_sigma2 = torch.stack(child_sigma2)
            else: #leaf node (or, more specifically, the "penultimate nodes", since the prompts themselves are the leaves)
                child_mu = self.mu[self.clusters[:self.num_arms]==root_id]
                child_sigma2 = self.sigma2[self.clusters[:self.num_arms]==root_id]

            # tau calculation
            mu_var = child_mu.var(unbiased=False) #variance of the means 
            sigma2_mean = child_sigma2.mean() #means of the variances
            tau_p = torch.clamp(mu_var - sigma2_mean, min=0.0)  #random-effects variance estimator / between-child variance / subtree heterogeneity measurement
            
            # precision weight calculation
            precision_weights = 1.0 / (child_sigma2 + tau_p + 1e-8)
            
            # calculate parent posterior
            mu_p = (precision_weights * child_mu).sum() / precision_weights.sum()
            sigma2_p = (1 / precision_weights.sum())

            
            # assign parameters for use in TS
            if root_id in self.tree:
                self.tree[root_id][1] = mu_p
                self.tree[root_id][2] = sigma2_p
                self.tree[root_id][3] = tau_p
            else: # store these in another list to not mess with original tree logic (could probably be refactored)
                if root_id not in self.tree and root_id not in self.penultimates: # if this happens, tree must have just been initialized
                    self.penultimates[root_id] = [None, None, None, None, 0.0]

                self.penultimates[root_id][1] = mu_p
                self.penultimates[root_id][2] = sigma2_p
                self.penultimates[root_id][3] = tau_p

            #print(f'[DEBUG - Subtree Update] Subtree return for Node {root_id} (mu, sigma2, tau): ({mu_p}, {sigma2_p}, {tau_p})')
            return mu_p, sigma2_p, tau_p
        
        def init_utilities(self):
            self.node_train_count = {}
            self.node_target_count = {}
            self.node_target_weight = {}
            self.compute_subtree_counts(0)
            self.precompute_local_utilities(0)

        #DM: style is a little inconsistent from other tree attribute setting methods; fine for now
        def set_node_utility(self, node_id, utility): # for BMC-T
            if node_id in self.tree:
                self.tree[node_id][4] = utility
            else:
                self.penultimates[node_id][4] = utility
        
        def compute_subtree_counts(self, root_id=0): #for BMC-T mainly
            """
            Returns:
                train_count, target_count for subtree rooted at root_id
            Also stores:
                self.node_train_count[root_id]
                self.node_target_count[root_id]
            """

            if root_id in self.tree:  # internal node
                total_train = 0.0
                total_target_weight = 0.0
                total_target_count = 0.0

                for child_id in self.tree[root_id][0]:
                    c_train, w_target, c_target = self.compute_subtree_counts(child_id)
                    total_train += c_train
                    total_target_weight += w_target
                    total_target_count += c_target

            else:  # penultimate node
                indices = np.where(self.clusters == root_id)[0]
                #safer version (probably redundant?)
                targets = self.latents['targets'][indices]
                weights = self.latents["target_weights"][indices]
                total_target_count = float(targets.sum())
                total_target_weight = float((weights * targets).sum())
                total_train = float((1 - targets).sum())

            self.node_train_count[root_id] = total_train
            self.node_target_count[root_id] = total_target_count
            self.node_target_weight[root_id] = total_target_weight

            #print(f"[DEBUG] Getting train / test counts for ID {root_id}: {self.node_train_count[root_id]} / {self.node_target_count[root_id]}")
            if self.node_train_count[root_id] <= 0.0:
                #print(f"[DEBUG] INVALID ID -> {root_id}")
                self.invalid_train_ids.append(root_id)

            return total_train, total_target_weight, total_target_count


        def precompute_local_utilities(self, root_id=0): # for BMC-T
            """
            For each parent, compute each child's utility relative to that sibling set.
            Stores utility in [4]

            Utility is absolute target-overlap with:
            - train-support gating
            - adaptive proportional smoothing on target mass
            """
            if root_id not in self.tree:
                return

            siblings = self.tree[root_id][0]
            k_level = len(siblings)

            N_train = sum(self.node_train_count[c] for c in siblings)
            N_target = sum(self.node_target_count[c] for c in siblings)
            W_target = sum(self.node_target_weight[c] for c in siblings)

            # if there's no target data, don't bother with utility calculation
            if N_target <= 0.0 or W_target <= 0.0:
                for child_id in siblings:
                    self.set_node_utility(child_id, 0.0)
                for child_id in siblings:
                    self.precompute_local_utilities(child_id)
                return

            # adaptive smoothing: a small fraction of average target mass per sibling
            rho = self.config.algorithm.bmc.targeting_rho  # e.g. 0.05
            avg_target_mass = W_target / max(k_level, 1)
            alpha = max(1e-8, rho * avg_target_mass)

            child_utilities = []
            for child_id in siblings:
                c_train = self.node_train_count[child_id]
                w_target = self.node_target_weight[child_id]
                c_target = self.node_target_count[child_id] # not used (for now)

                # support-aware gate
                tau = self.config.algorithm.bmc.ltt.hdbscan_min_cluster_pct * N_train
                gate = c_train / (c_train + tau) if (c_train + tau) > 0 else 0.0

                # weighted absolute target-overlap utility
                p_target = (w_target + alpha) / (W_target + alpha * k_level)
                utility = p_target * gate

                child_utilities.append((child_id, utility))

            #u_max = max([u for _, u in child_utilities], default=0.0) #DM: not needed?

            for child_id, utility in child_utilities:
                self.set_node_utility(child_id, utility)

            print('f[DEBUG] Sanity checking utility:\n')
            print(f"[DEBUG] Node {root_id}: W_target={W_target:.4f}, k={k_level}, alpha={alpha:.6f}") #DM: disable later
            sum_ptarget = 0.0
            for child_id in siblings:
                w_target = self.node_target_weight[child_id]
                p_target = (w_target + alpha) / (W_target + alpha * k_level)
                sum_ptarget += p_target
            print(f"[DEBUG] Node {root_id}: sum_p_target={sum_ptarget}")
            print('f[DEBUG] End of utility sanity check')

            # recurse
            for child_id in siblings:
                self.precompute_local_utilities(child_id)

        ## providing batch awareness:
        def initialize_batch_availability(self): # for new "batch-aware" style of BMC
            """
            Initialize temporary per-batch availability structures.

            batch_available_prompt[prompt_id] == True means this prompt can still be
            sampled during the current batch construction.

            available_counts[node_id] == number of available descendant prompts.
            This applies to both internal tree nodes and penultimate cluster nodes.
            """
            # Assumes train prompt indexing is 0 ... self.num_arms - 1
            self.batch_available_prompt = torch.ones(self.num_arms, dtype=torch.bool)


            self.available_counts = {}
            root_id = 0
            self.initialize_subtree_availability(root_id)
        
        def initialize_subtree_availability(self, node_id): # for new "batch-aware" style of BMC
            """
            Recursively compute available descendant prompt counts under each node.

            Returns:
                int: number of available prompts under this node
            """
            if node_id in self.tree:
                total = 0
                child_ids = self.tree[node_id][0]

                for child_id in child_ids:
                    total += self.initialize_subtree_availability(child_id)

                self.available_counts[node_id] = total
                return total

            else:
                # Penultimate cluster node: its descendants are actual prompt arms
                prompt_ids = np.where(self.clusters[:self.num_arms] == node_id)[0]

                #if prompt_ids.numel() == 0:
                #    total = 0
                #else:
                total = int(self.batch_available_prompt[prompt_ids].sum().item())

                self.available_counts[node_id] = total
                return total
        
        def decrement_batch_availability(self, prompt_id, cluster_id): # for new "batch-aware" style of BMC
            """
            Mark a selected prompt as unavailable for the remainder of the current batch,
            and decrement descendant-availability counts up the ancestor chain.

            Args:
                prompt_id (int): selected prompt arm id
                cluster_id (int): penultimate cluster node that contains the prompt
            """
            prompt_id = int(prompt_id)
            cluster_id = int(cluster_id)

            if not self.batch_available_prompt[prompt_id]:
                raise ValueError(f"[ERROR] Prompt {prompt_id} already unavailable in this batch.")

            self.batch_available_prompt[prompt_id] = False

            # decrement the penultimate cluster first
            self.available_counts[cluster_id] -= 1

            # "walk upward" through parents
            node = cluster_id
            while node in self.child_to_parent_map:
                node = self.child_to_parent_map[node]
                self.available_counts[node] -= 1

        def sample_batch(self):
            #print("[DEBUG] In BanditSampler iteration")
            
            if self.config.algorithm.bmc.round_robin_init and self.config.algorithm.bmc.algo_type=='bmc':
                if not self.sampled.all():
                    print(f'[DEBUG] Round-Robin in progress; Progress={(self.sampled.sum() /  self.sampled.numel())*100}%')
                    unsampled_ids = torch.where(self.sampled==False)[0]
                    if unsampled_ids.numel() >= self.batch_size:
                        indices = unsampled_ids[torch.randperm(unsampled_ids.numel())[:self.batch_size]]
                    else:
                        remaining = self.batch_size - unsampled_ids.numel()
                        sampled_ids = torch.where(self.sampled == True)[0]
                        extra_ids = sampled_ids[torch.randperm(sampled_ids.numel())[:remaining]]
                        indices = torch.cat([unsampled_ids, extra_ids], dim=0)
                    return indices.tolist()
                else:
                    pass
                    #print(f'[DEBUG] Round-Robin completed; skipping...')

            if self.config.algorithm.bmc.ltt.enable:
                with torch.no_grad():
                    self.initialize_batch_availability()

                    if self.available_counts[0] < self.batch_size:
                        raise ValueError(
                            f"[ERROR] Not enough available prompts to sample a unique batch. "
                            f"Available under root: {self.available_counts[0]}, "
                            f"batch size: {self.batch_size}"
                        )

                    samples = torch.zeros(self.batch_size, dtype=torch.int64)
                    self.penultimate_ids = set(self.clusters.tolist()) # could also use self.penultimates (lazy)

                    for b in range(self.batch_size):
                        level = 0  # start from root for this batch element

                        while True:
                            if level in self.tree:
                                # internal node: only keep children with available descendants
                                child_ids = [
                                    cid for cid in self.tree[level][0]
                                    if self.available_counts.get(cid, 0) > 0
                                ]

                                if len(child_ids) == 0:
                                    raise ValueError(
                                        f"[ERROR] No available children under internal node {level}"
                                    )

                                tau = torch.as_tensor(self.tree[level][3], dtype=torch.float32)

                                child_mu = []
                                child_sigma2 = []
                                child_utility = []

                                for c in child_ids:
                                    if c in self.tree:
                                        mu = torch.as_tensor(self.tree[c][1], dtype=torch.float32)
                                        sigma2 = torch.as_tensor(self.tree[c][2], dtype=torch.float32)
                                        utility = torch.as_tensor(self.tree[c][4], dtype=torch.float32)
                                    else:
                                        mu = torch.as_tensor(self.penultimates[c][1], dtype=torch.float32)
                                        sigma2 = torch.as_tensor(self.penultimates[c][2], dtype=torch.float32)
                                        utility = torch.as_tensor(self.penultimates[c][4], dtype=torch.float32)

                                    child_mu.append(mu)
                                    child_sigma2.append(sigma2)
                                    child_utility.append(utility)

                                child_mu = torch.stack(child_mu)
                                child_sigma2 = torch.stack(child_sigma2)
                                child_utility = torch.stack(child_utility)

                                # thompson Sampling over children
                                var = torch.clamp(child_sigma2 + tau, min=1e-8)
                                scores = torch.randn(len(child_ids))
                                scores = scores * torch.sqrt(var) + child_mu

                                if self.config.algorithm.bmc.targeting:
                                    EPS = 1e-8
                                    gamma = self.config.algorithm.bmc.targeting_gamma

                                    u_max = torch.clamp(child_utility.max(), min=EPS)
                                    u_norm = child_utility / u_max
                                    utility_bonus = gamma * self.r_max * u_norm
                                    scores = scores + utility_bonus

                                selection = torch.argmax(scores).item()
                                level = child_ids[selection]

                            elif level in self.penultimate_ids:
                                # penultimate node: choose among available prompts only
                                prompt_ids = torch.where(self.clusters[:self.num_arms] == level)[0].to(self.mu.device)

                                if prompt_ids.numel() == 0:
                                    raise ValueError(f"[ERROR] Cluster {level} has no train prompts.")

                                available_mask = self.batch_available_prompt[prompt_ids]
                                prompt_ids = prompt_ids[available_mask]

                                if prompt_ids.numel() == 0:
                                    raise ValueError(f"[ERROR] Cluster {level} has no available prompts left.")

                                scores = (
                                    self.mu[prompt_ids]
                                    + torch.sqrt(torch.clamp(self.sigma2[prompt_ids], min=1e-8))
                                    * torch.randn(prompt_ids.shape[0])
                                )

                                selection = torch.argmax(scores).item()
                                chosen_prompt = prompt_ids[selection]

                                samples[b] = chosen_prompt

                                # Mark selected prompt unavailable for future selections
                                self.decrement_batch_availability(
                                    prompt_id=chosen_prompt.item(),
                                    cluster_id=level
                                )
                                break

                            else:
                                raise ValueError(f"[ERROR] sample_batch(): non-existent level ID {level}")
 
                indices = samples
                print("[DEBUG] Sampling concluded")
                unique_idx, counts = torch.unique(samples, return_counts=True)
                duplicates = unique_idx[counts > 1]
                if duplicates.numel() > 0:
                    print(f'[ERROR] Duplicates detected in tree.')
                else:
                    print(f'[DEBUG] No duplicates confirmed')
                
            else:
                print("[DEBUG] No tree use; curriculum only (Thompson Sampling)")

                if self.config.algorithm.bmc.algo_type=='bmc': #DM-MOPPS
                    scores = self.mu + torch.sqrt(self.sigma2) * torch.randn(self.num_arms)
                    indices = torch.topk(scores, k=self.batch_size).indices
                elif self.config.algorithm.bmc.algo_type=='mopps':
                    print('[DEBUG] MoPPS version of TS!')
                    
                    beta_samples = torch.distributions.Beta(self.alpha, self.beta).sample()
                    scores = -torch.abs(beta_samples - 0.5)
                    indices = torch.topk(scores, k=self.batch_size).indices
                
            #print(f"[DEBUG] Indices calculated; Shape={indices.shape}")
            return indices.tolist()

        
        def display_word_clouds(self, clusters, raw_prompt_key='raw_prompt'):
            from wordcloud import WordCloud
            from os.path import commonprefix

            num_samples = self.config.algorithm.bmc.prompts_per_node

            for cluster_id in list(set(clusters.astype(int))):
                if cluster_id in self.rendered_clusters:
                    #print(f"[DEBUG] Skipping {cluster_id}")
                    continue
                self.rendered_clusters[cluster_id] = 1

                texts = []
                for i in np.where(clusters==cluster_id)[0]:
                    if self.model_type=="LLM":
                        text = self.dataset[int(i)][raw_prompt_key][0]['content']
                    elif self.model_type=="VLM":
                        text = self.dataset[int(i)][raw_prompt_key][0]['content'][1]['text'] # based on verl's Geo3K example
                    texts.append(text)

                
                # Remove common prefix
                common_prefix = commonprefix(texts)
                if common_prefix:
                    candidate_texts = [text[len(common_prefix):].strip() for text in texts]
                    # Only apply if not all empty
                    if any(candidate_texts):
                        texts = candidate_texts


                # Remove common suffix
                reversed_texts = [t[::-1] for t in texts]
                common_suffix_rev = commonprefix(reversed_texts)
                suffix_len = len(common_suffix_rev)

                if suffix_len > 0:
                    candidate_texts = [t[:-suffix_len].strip() for t in texts]

                    # Only apply if not all empty
                    if any(candidate_texts):
                        texts = candidate_texts
                else:
                    texts = [t.strip() for t in texts]


            
                # Create the word cloud
                all_text = " ".join(texts)
                wc = WordCloud(width=800, height=400, background_color="white").generate(all_text)
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.savefig(f"{self.analysis_file}/latent_wc_{'outliers' if int(cluster_id)==-1 else cluster_id}.png")
                plt.close()
                

                # Write text
                subtree_pct = len(np.where(clusters==cluster_id)[0]) / len(clusters)
                subsample_text = ""

                num_samples = min(num_samples, len(np.where(clusters==cluster_id)[0]))
                #print(f"[DEBUG-TEST] NUM_SAMPLES = {num_samples}")
                for j in range(num_samples):
                    subsample = f"Prompt {j}: {texts[j]}\n"
                    subsample_text = subsample_text + subsample
                display_text = f"Cluster #{cluster_id} Statistics -\n  Percantage of Data:{subtree_pct*100:.2f}%\n  Example Prompts:\n" + subsample_text + "\n\n\n"
                with open(f"{self.analysis_file}/latent_wc_data.txt", "a") as f:
                    f.write(display_text)
                
            print("[DEBUG] Word clouds created")
        
        def display_cluster_statistics(self, clusters):
            print("\n\n**Cluster Statistics:")
            for cluster_id in list(set(clusters.astype(int))):
                print(f"*Cluster #{cluster_id}")
                print(f"\tPercentage of Dataset: {(clusters[clusters==cluster_id].shape[0]/len(clusters))*100}%")
                print(f"\tRaw Size: {clusters[clusters==cluster_id].shape[0]}")

        def __len__(self):
            return len(dataset)
        
        def __iter__(self):
            yield self.sample_batch()

    sampler = BanditSampler(config=config, 
    data_source=dataset, 
    latents=latents,
    batch_size=config.data.train_batch_size)

    return sampler


# For the Worker base class  
from verl.single_controller.base import Worker  
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray.base import RayResourcePool, RayClassWithInitArgs, RayWorkerGroup
from verl import DataProto
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText 
from torch.utils.data import DataLoader
import tqdm

@ray.remote
class LatentExtractor(Worker):  
    def __init__(self, model_path: str, layer_depth: float, batch_size: int):  
        super().__init__()  
        self.model_path = model_path  
        self.layer_depth = layer_depth  
        self.batch_size = batch_size  
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        #self.device = f"cuda:{self.rank}"  # ,- verl sets rank automatically; do not use; may work for non-verl cases

        # Initialize model
        if "VL" in model_path or "vision" in model_path.lower(): # for Vision-Language models (mainly LLama and Qwen)
            self.model = AutoModelForImageTextToText .from_pretrained(  
            model_path,  
            dtype=torch.bfloat16  
            ).to(self.device)
        else: #for LLMs
            self.model = AutoModelForCausalLM.from_pretrained(  
                model_path,  
                dtype=torch.bfloat16  
                ).to(self.device)  
        self.model.eval()  
      
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def extract_latents(self, data: DataProto) -> DataProto:
        input_ids = data.batch["input_ids"].to(self.device)
        attention_mask = data.batch["attention_mask"].to(self.device)
        indices = data.batch["indices"]  # keep on CPU

        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            #num_layers = self.model.config.num_hidden_layers
            num_layers = len(out.hidden_states)
            L = min(round(self.layer_depth * num_layers), num_layers - 1)
            H = out.hidden_states[L]

            mask = attention_mask.unsqueeze(-1).float()
            pooled = (H * mask).sum(dim=1) / mask.sum(dim=1)

        return DataProto.from_dict({
            "latents": pooled.cpu(),
            "indices": indices
        })

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def extract_latents_targeting(self, data: DataProto) -> DataProto:
        input_ids = data.batch["input_ids"].to(self.device)
        attention_mask = data.batch["attention_mask"].to(self.device)
        indices = data.batch["indices"]  # keep on CPU
        targets = data.batch["targets"]
        target_weights = data.batch["target_weights"]

        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )

            #num_layers = self.model.config.num_hidden_layers
            num_layers = len(out.hidden_states)
            L = min(round(self.layer_depth * num_layers), num_layers - 1)
            H = out.hidden_states[L]

            mask = attention_mask.unsqueeze(-1).float()
            pooled = (H * mask).sum(dim=1) / mask.sum(dim=1)


        return DataProto.from_dict({
            "latents": pooled.cpu(),
            "indices": indices,
            "targets": targets,
            "target_weights": target_weights
        })
        
  
 
def extract_latents_distributed(model_path, dataset, layer_depth, batch_size, num_gpus, targeting):  
    ray.init()  
      
    # Create resource pool with all GPUs  
    resource_pool = RayResourcePool(process_on_nodes=[num_gpus], use_gpu=True)  
      
    # Create worker group  
    worker_cls = RayClassWithInitArgs(  
        cls=LatentExtractor,  
        model_path=model_path,  
        layer_depth=layer_depth,  
        batch_size=batch_size  
    )  

    worker_group = RayWorkerGroup(resource_pool, worker_cls)  
    
      
    # Prepare data as DataProto  
    def collate_fn(batch):
        input_ids = pad_sequence([x["input_ids"] for x in batch],
                                batch_first=True, padding_value=0)
        attention_mask = pad_sequence([x["attention_mask"] for x in batch],
                                    batch_first=True, padding_value=0)
        indices = torch.tensor([x["sampler_index"] for x in batch], dtype=torch.long)

        if targeting:
            targets = torch.tensor([x["targets"] for x in batch], dtype=torch.long)
            target_weights = torch.tensor([x["target_weights"] for x in batch], dtype=torch.float)
            

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "indices": indices,
                "targets": targets,
                "target_weights": target_weights
            }
        
        else:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "indices": indices
            }
      
    # Process dataset in batches  
    all_latents = []
    all_indices = []
    all_targets = []
    all_target_weights = []

    dataloader = DataLoader(dataset, batch_size=batch_size,   
                           shuffle=False, collate_fn=collate_fn, drop_last=False)  
    
    if targeting:
        key_list = ["input_ids", "attention_mask", "indices", "targets", "target_weights"]
    else:
        key_list = ["input_ids", "attention_mask", "indices"]

    print(f"Loading latents... (num_gpus={num_gpus}; batch_size={batch_size})")
    for batch in tqdm.tqdm(dataloader):
        ## handling division issues while also ensuring ordering:
        # compute padding to make batch divisible by num_gpus
        batch_size_actual = batch["input_ids"].shape[0]
        pad_size = (num_gpus - (batch_size_actual % num_gpus)) % num_gpus

        if pad_size > 0:
            for key in key_list:
                tensor = batch[key]
                # Repeat last element to pad
                batch[key] = torch.cat([tensor, tensor[-1:].repeat(pad_size, *[1]*(tensor.dim()-1))], dim=0)

        data_proto = DataProto.from_dict(batch)

        if targeting: #DM: two different versions, since we can't pass non data-proto objects (booleans) in these functions
            result = worker_group.extract_latents_targeting(data_proto)
        else:
            result = worker_group.extract_latents(data_proto)

        # Unpad results
        latents = result.batch["latents"][:batch_size_actual]
        indices = result.batch["indices"][:batch_size_actual]
        all_latents.append(latents)
        all_indices.append(indices)
        
        if targeting:
            targets = result.batch["targets"][:batch_size_actual]
            all_targets.append(targets)

            target_weights = result.batch["target_weights"][:batch_size_actual]
            all_target_weights.append(target_weights)

    latents = torch.cat(all_latents, dim=0)
    indices = torch.cat(all_indices, dim=0)
    
    order = torch.argsort(indices)
    latents = latents[order]

    if targeting:
        targets = torch.cat(all_targets, dim=0)
        targets = targets[order]

        target_weights = torch.cat(all_target_weights, dim=0)
        target_weights = target_weights[order]
    else:
        targets = None
        target_weights = None
    

    ray.shutdown()
    return latents, targets, target_weights

if __name__ == "__main__":
    main()