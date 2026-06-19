# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


# TODO: add unit tests

import logging
import os
import re
from enum import Enum

import torch
import numpy as np

import verl.utils.hdfs_io as hdfs_io
from verl.single_controller import WorkerGroup
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.logger import log_with_rank
from verl.workers.engine import BaseEngine
from verl.utils.checkpoint import CheckpointHandler, OrchestrationMode

def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


class BMCCheckpointHandler(CheckpointHandler):
    """
    Checkpoint handler handles the path, global_step of a checkpoint folder.
    Currently, it only works with a single model.
    We can expand it to support multiple models. It is expected to be used with SPMD style (e.g., torchrun)
    """

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.default_local_dir, f"global_step_{step}")
        if self.rank == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = self.max_ckpt_to_keep

        # Use checkpoint manager to save
        actor_path = os.path.join(local_global_step_folder, "actor")
        self.engine.save_checkpoint(
            local_path=actor_path, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )


        if self.rank == 0:
            local_mkdir_safe(local_global_step_folder)  
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")  
          
            dataloader_state_dict = self.train_dataloader.state_dict()  
            torch.save(dataloader_state_dict, dataloader_local_path)  
            print(f"Saved dataloader state to: {dataloader_local_path}")  

            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

            # Store BMC elements
            bmc_elements_path = os.path.join(self.default_local_dir, f"bmc")
            bmc_checkpoint_path = os.path.join(local_global_step_folder, f"bmc")
            hdfs_io.makedirs(bmc_elements_path, exist_ok=True)
            hdfs_io.copy(src=bmc_elements_path, dst=bmc_checkpoint_path, dirs_exist_ok=True)
            print(f"Copied BMC elements from {bmc_elements_path} to {bmc_checkpoint_path}")



        # Copy to HDFS if configured
        if self.rank == 0 and self.default_hdfs_dir:
            hdfs_io.makedirs(self.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.default_hdfs_dir, dirs_exist_ok=True)

        if self.mode == OrchestrationMode.SPMD:
            torch.distributed.barrier()

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.rank,
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        actor_path = os.path.join(checkpoint_path, "actor")  
        bmc_path = os.path.join(checkpoint_path, "bmc")  

        # Use checkpoint manager to load model state
        self.engine.load_checkpoint(actor_path)
        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        # Loading BMC elements:
        self.train_dataloader.sampler.tree = torch.load(os.path.join(bmc_path, 'tree.pt'))
        self.train_dataloader.sampler.clusters = torch.from_numpy(np.load(os.path.join(bmc_path,'leaves.npz'))['data'])
        self.train_dataloader.sampler.sampled = torch.from_numpy(np.load(os.path.join(bmc_path,'sampled_history.npy'))[resume_step, :])
        self.train_dataloader.sampler.staleness = torch.from_numpy(np.load(os.path.join(bmc_path,'staleness_history.npy'))[resume_step, :])
        

        # if tree-only ablation is desired, don't use the beliefs updated from round-robin
        # otherwise, load everything as normal
        """
        if self.train_dataloader.sampler.use_tree and self.train_dataloader.sampler.use_curriculum:
            self.train_dataloader.sampler.mu = torch.from_numpy(np.load(os.path.join(bmc_path,'mu_history.npy'))[resume_step, :])
            self.train_dataloader.sampler.lambda_ = torch.from_numpy(np.load(os.path.join(bmc_path,'lambda_history.npy'))[resume_step, :])
            self.train_dataloader.sampler.sigma2 = torch.from_numpy(np.load(os.path.join(bmc_path,'sigma2_history.npy'))[resume_step, :])
            self.train_dataloader.sampler.update_tree()
        """

        if self.train_dataloader.sampler.use_curriculum:
            self.train_dataloader.sampler.mu = torch.from_numpy(np.load(os.path.join(bmc_path,'mu_history.npy'))[resume_step, :])
            self.train_dataloader.sampler.lambda_ = torch.from_numpy(np.load(os.path.join(bmc_path,'lambda_history.npy'))[resume_step, :])
            self.train_dataloader.sampler.sigma2 = torch.from_numpy(np.load(os.path.join(bmc_path,'sigma2_history.npy'))[resume_step, :])
            self.train_dataloader.sampler.update_tree()
            
        if self.train_dataloader.sampler.use_tree:
            self.train_dataloader.sampler.update_tree()
            
        print('Loaded BMC elements')
        
        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):  
        """Load dataloader state from checkpoint (Ray pattern)"""  
        dataloader_path = os.path.join(checkpoint_path, "data.pt")  
        
        if os.path.exists(dataloader_path):  
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)  
            self.train_dataloader.load_state_dict(dataloader_state_dict)  
            print(f"Loaded dataloader state from {dataloader_path}")  
        else:  
            print(f"Warning: No dataloader state found at {dataloader_path}")