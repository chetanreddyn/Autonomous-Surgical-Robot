"""
author: Zooey He
email: zhuohonghe@gmail.com
date: 2025-04-21
"""

import os
import torch
import pickle
import numpy as np
from einops import rearrange
from yacs.config import CfgNode
from typing import Union, Literal, List

from mlcore.folder import folder_exists
from mlcore.file import file_exists, find_files_in
from mlcore.etc import ETC

from AdaptACT.config.defaults import get_cfg
from AdaptACT.constants import (
    DATASET_STATS_FILENAME_JSON, 
    DATASET_STATS_FILENAME_PKL,
    TRAIN_CFG_FILENAME, 
    CHECKPOINT_FILENAME_REGEX,
    BEST_CHECKPOINT_REGEX
)
from AdaptACT.utils.custom_logging import get_logger
from AdaptACT.policy import make_policy_obj
from AdaptACT.utils.checkpoints import load_checkpoint_weights
from AdaptACT.utils.operations import DatasetStats


"""
AutonomousControl class used as a plugin for robot control loops.

Basic usage:
```python
from AdaptACT.procedures.inference_model import AutonomousController

# Initialize the AutonomousController.
controller = AutonomousController.from_train_dir(
    "path/to/train/run/folder"
)

...

# At each timestep, run:
action = controller.step(img, qpos)

...

# When starting a new rollout, run:
controller.reset()

```
"""

class AutonomousController():
    """
    An interace class which abstracts the inference of the learned controller for the robot system.
    """
    def __init__(
            self, cfg: CfgNode, policy: torch.nn.Module = None, device: Union[str,int] = "cuda:0"
        ):
        """
        Initialize the controller.

        Args:
            cfg: the configuration node object.
            policy: if given use this policy instead of building from the cfg.
            device: the device to place the model and corresponding data.
        """
        self._log = get_logger()
        self._cfg = cfg
        self._device = device

        self._dataset_stats = DatasetStats.from_file(cfg.ROLLOUT.DATASET_STATS)
        
        if self._cfg.ROLLOUT.TEMPORAL_AGG:
            self._query_period = 1
        elif self._cfg.ROLLOUT.QUERY_PERIOD != 0:
            self._query_period = self._cfg.ROLLOUT.QUERY_PERIOD
        else:
            self._query_period = self._cfg.MODEL.CHUNK_SIZE

        self._build_pretrained_model(policy=policy)
        self.reset()
        
    @classmethod
    def from_train_dir(
            cls, 
            train_dir: str,
            ckpt_path: str = None,
            ckpt_strategy: Literal["best", "last"] = "last",
            rollout_len: int = None,
            device: Union[str,int] = "cuda:0",
            opts: List[str] = None
        ) -> "AutonomousController":
        """
        Build an AutonomousController object from a training directory.

        Args:
            train_dir: the filepath to the training directory.
            ckpt_path: If given, load the checkpoint at this path. Otherwise, follow ckpt_strategy
            ckpt_strategy: how to select a checkpoint from the train_dir.
                - "best": selects the lowest-validation-loss checkpoint.
                - "last": selects the checkpoint at the last epoch.
                - "none": uses the checkpoint within the train_cfg.yaml at EXEC.PRETRAIN_PATH
            rollout_len: the number of timesteps to expect for rolling out. 
                By default, uses ROLLOUT.EPISODE_LEN
            device: where to load the model. Options can include 'cpu', 'cuda:0', etc.
        """
        # Input validation.
        assert folder_exists(train_dir), f"Could not build from non-existant train dir: {train_dir}"
        assert file_exists(ckpt_path) if ckpt_path else True, f"No checkpoint found at: {ckpt_path}"
        assert ckpt_strategy in ["best", "last", "none"], f"Unsupported strategy: {ckpt_strategy}"

        cfg_path = os.path.join(train_dir, TRAIN_CFG_FILENAME)
        # Get the dataset_stats file path.
        dataset_stats_pkl_path = os.path.join(train_dir, DATASET_STATS_FILENAME_JSON)
        if not file_exists(dataset_stats_pkl_path):
            dataset_stats_pkl_path = os.path.join(train_dir, DATASET_STATS_FILENAME_PKL)
        
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_list(opts or [])
        cfg.EXEC.MODE = "rollout"
        cfg.ROLLOUT.DATASET_STATS = dataset_stats_pkl_path
        cfg.ROLLOUT.EPISODE_LEN = rollout_len or cfg.ROLLOUT.EPISODE_LEN or cfg.TASK.EPISODE_LEN

        # Set the pretrained checkpoint to load.
        if ckpt_path:
            cfg.EXEC.PRETRAIN_PATH = ckpt_path
        else:
            if ckpt_strategy == "last":
                ckpt_filenames = find_files_in(
                    train_dir, template=CHECKPOINT_FILENAME_REGEX, abs=False
                )
                ckpt_filenames.sort(key=lambda fn: int(fn.split("_")[2]))  # Sort by epoch.
                cfg.EXEC.PRETRAIN_PATH = os.path.join(train_dir, ckpt_filenames[-1])
            
            elif ckpt_strategy == "best":
                cfg.EXEC.PRETRAIN_PATH = find_files_in(
                    train_dir, template=BEST_CHECKPOINT_REGEX, abs=True
                )[-1]

        return cls(cfg, device=device)

    @classmethod
    def from_multiple_train_dirs(
            cls,
            train_dirs: List[str],
            ckpt_paths: List[str] = None,
            ckpt_strategy: Literal["best", "last"] = "last",
            rollout_len: int = None,
            device: Union[str,int] = "cuda:0",
            opts: List[str] = None
        ) -> "AutonomousController":
        """
        Build an AutonomousController object which orchestrates multiple agents.

        Args:
            train_dir: the filepaths to the agents' training directories.
            ckpt_paths: If given, load the checkpoint at this path. Otherwise, follow ckpt_strategy
            ckpt_strategy: how to select a checkpoint from the train_dir.
                - "best": selects the lowest-validation-loss checkpoint.
                - "last": selects the checkpoint at the last epoch.
                - "none": uses the checkpoint within the train_cfg.yaml at EXEC.PRETRAIN_PATH
            rollout_len: the number of timesteps to expect for rolling out. 
                By default, uses ROLLOUT.EPISODE_LEN
            device: where to load the model. Options can include 'cpu', 'cuda:0', etc.
            opts: additional options to override the config.
                For example: ["ROLLOUT.TEMPORAL_AGG", "True", "ROLLOUT.TEMPORAL_AGG_K", "0.01"]
        """
        # Input validation.
        assert isinstance(train_dirs, list) and len(train_dirs) >= 1, f"Invalid train_dirs."
        for train_dir in train_dirs:
            assert folder_exists(train_dir), f"Could not build from non-existant train dir: {train_dir}"
        if ckpt_paths:
            for ckpt_path in ckpt_paths:
                assert file_exists(ckpt_path), f"No checkpoint found at: {ckpt_path}"
        assert ckpt_strategy in ["best", "last", "none"], f"Unsupported strategy: {ckpt_strategy}"

        # Use the first agent's cfg as the starting point for the multi-agent cfg.
        cfg_path = os.path.join(train_dirs[0], TRAIN_CFG_FILENAME)
        # Get the dataset_stats file path.
        dataset_stats_pkl_path = os.path.join(train_dirs[0], DATASET_STATS_FILENAME_JSON)
        if not file_exists(dataset_stats_pkl_path):
            dataset_stats_pkl_path = os.path.join(train_dirs[0], DATASET_STATS_FILENAME_PKL)
            # TODO validate that the dataset stats contents are the same.
        
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        cfg.merge_from_list(opts or [])
        cfg.MODEL.POLICY = "MultiAgentPolicy"
        cfg.EXEC.MODE = "rollout"
        cfg.ROLLOUT.DATASET_STATS = dataset_stats_pkl_path
        cfg.ROLLOUT.EPISODE_LEN = rollout_len or cfg.ROLLOUT.EPISODE_LEN or cfg.TASK.EPISODE_LEN

        # Get the pretrained configurations.
        cfg.MODEL.MULTI_AGENT.CFGS = []
        for train_dir in train_dirs:
            agent_cfg_path = os.path.join(train_dir, TRAIN_CFG_FILENAME)
            cfg.MODEL.MULTI_AGENT.CFGS.append(agent_cfg_path)

        # Set the pretrained checkpoint to load.            
        if not ckpt_paths:  # Detecting checkpoints.
            ckpt_paths = []
            for train_dir in train_dirs:
                if ckpt_strategy == "last":
                    ckpt_filenames = find_files_in(
                        train_dir, template=CHECKPOINT_FILENAME_REGEX, abs=False
                    )
                    ckpt_filenames.sort(key=lambda fn: int(fn.split("_")[2]))  # Sort by epoch.
                    selected_ckpt_path = os.path.join(train_dir, ckpt_filenames[-1])
                    ckpt_paths.append(selected_ckpt_path)

                elif ckpt_strategy == "best":
                    selected_ckpt_path = find_files_in(
                        train_dir, template=BEST_CHECKPOINT_REGEX, abs=True
                    )[-1]
                    ckpt_paths.append(selected_ckpt_path)
        # Set the PRETRAIN_PATHS value with found checkpoints.
        cfg.EXEC.PRETRAIN_PATH = ";".join(ckpt_paths)

        return cls(cfg, device=device)

    def _build_pretrained_model(self, policy: torch.nn.Module = None):
        """
        Instantiate and load the pretrained model.
        """
        if policy:
            self._policy = policy.to(self._device).eval()
        else:
            self._log.info(f'Building policy for autonomous control ...')
            self._policy = make_policy_obj(
                self._cfg.MODEL.POLICY, self._cfg
            ).to(self._device).eval()
            
            self._log.info(f"Built policy with architecture:\n{str(self._policy)}")
            fully_loaded = load_checkpoint_weights(self._policy, self._cfg.EXEC.PRETRAIN_PATH)

            if not fully_loaded:
                raise RuntimeError("Unable to completely load checkpoint weights")

            self._log.info(f"Successfully loaded checkpoint from: {self._cfg.EXEC.PRETRAIN_PATH}")
        
        # Compile the policy for faster inference
        self._policy = torch.compile(self._policy)
        self._log.info("Successfully compiled policy for optimized inference.")
        
        # Enable mixed precision inference
        self._use_amp = True
        self._log.info("Enabled mixed precision (FP16) inference with autocast.")
        
        self._log.info("Successfully obtained policy for rollout.")

    def reset(self):
        """
        Reset this controller object for a new rollout.
        """
        self._t = 0

        if self._cfg.ROLLOUT.TEMPORAL_AGG or self._cfg.ROLLOUT.CONFIDENCE_AGG:
            # Create a table to store actions predicted at each timestamp for aggregation (normalized).
            self._agg_table = torch.full([
                self._cfg.ROLLOUT.EPISODE_LEN,
                self._cfg.ROLLOUT.EPISODE_LEN+self._cfg.MODEL.CHUNK_SIZE,
                self._cfg.MODEL.STATE_DIM
            ], torch.inf).to(self._device)

        # Create an array to store historical joint states (normalized).
        self._history_qpos = torch.zeros(
            (self._cfg.ROLLOUT.EPISODE_LEN, self._cfg.MODEL.STATE_DIM)
        ).to(self._device)

        # Create an array to store historical actions (normalized).
        self._history_actions = torch.zeros(
            (self._cfg.ROLLOUT.EPISODE_LEN, self._cfg.MODEL.STATE_DIM)
        ).to(self._device)
        
        # Pre-allocate GPU tensors for inputs to avoid repeated transfers
        # We'll set the actual buffer size after seeing the first image
        self._gpu_img_buffer = None
        self._gpu_qpos_buffer = torch.zeros(self._cfg.MODEL.STATE_DIM, device=self._device)  # Pre-allocated qpos buffer

    def step(self, img: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """
        Perform a step of the autonomous controller.

        Args:
            img: the image(s) at this timestep. Shape N H W C. Range 0.0-1.0;
                N = number of cameras / images; H = height; W = width; C = channel (RGB)
            qpos: the raw qpos state received from the system. Shape D. D = State Dim.

        Returns:
            autonomous_action: an array containing action prediction at step. Shape D.
        """
        timer = ETC(1)
        
        if self._t >= self._cfg.ROLLOUT.EPISODE_LEN:
            raise RuntimeError(
                "Rollout was already completed. AutonomousController.step() cannot be called any "
                "more times. If rollout length needs to be increased, please consider increasing "
                f"ROLLOUT.EPISODE_LEN. Currently, it is set to {self._cfg.ROLLOUT.EPISODE_LEN}"
            )

        # Prepare the image input - copy directly to GPU buffer
        img_tensor = torch.from_numpy(img).float()
        img_tensor = rearrange(img_tensor, "N H W C -> 1 N C H W")  # Rearrange & add batch dim.
        
        # Initialize GPU buffer on first use with correct dimensions
        if self._gpu_img_buffer is None:
            self._gpu_img_buffer = torch.zeros_like(img_tensor, device=self._device)
        
        self._gpu_img_buffer.copy_(img_tensor, non_blocking=True)  # Direct copy to GPU buffer

        # Process & register the qpos - copy directly to GPU buffer  
        qpos_tensor = torch.from_numpy(qpos).float()
        qpos_tensor = self._dataset_stats.norm_qpos(qpos_tensor, method=self._cfg.MODEL.STATE_NORM)
        self._gpu_qpos_buffer.copy_(qpos_tensor, non_blocking=True)  # Direct copy to GPU buffer
        
        # Use the GPU buffers for processing
        img = self._gpu_img_buffer
        qpos = self._gpu_qpos_buffer
        self._history_qpos[self._t, :] = qpos  # Save the qpos in history.

        # Inference the policy to get autonomous action predictions.
        if self._t % self._query_period == 0:

            # Prepare the qpos input.
            en = self._t + 1
            st = max(0, en - self._cfg.MODEL.STATE_HISTORY)
            qpos_in = self._history_qpos[st:en, :]
            # Add padding to start of qpos input.
            n_pad = self._cfg.MODEL.STATE_HISTORY - qpos_in.shape[0]
            if n_pad > 0:
                pad = qpos_in[0:1,:].repeat(n_pad, 1)
                qpos_in = torch.cat([pad, qpos_in], axis=0)
            qpos_in = qpos_in.unsqueeze(0)  # Add batch dim.

            # Prepare action prompt.
            prompt = None
            if self._cfg.MODEL.ACTION_PROMPT.ENABLE:
                en = self._t
                st = max(0, en - self._cfg.MODEL.ACTION_PROMPT.LEN)
                prompt = self._history_actions[st:en, :]
                # Add zero padding to start of action.
                n_pad = self._cfg.MODEL.ACTION_PROMPT.LEN - prompt.shape[0]
                if n_pad > 0:
                    pad = torch.zeros((n_pad, self._cfg.MODEL.STATE_DIM), device=self._device)
                    prompt = torch.cat([pad, prompt], axis=0)
                prompt = prompt.unsqueeze(0)  # Add batch dim.

            # Call the model.
            if self._cfg.MODEL.CAUSAL_DECODING:
                # Autoregressive calling.
                self._all_actions = torch.zeros(
                    self._query_period, self._cfg.MODEL.STATE_DIM, device=self._device
                )
                for q in range(self._query_period):
                    past_gen_actions = self._all_actions[:q, :]  # GENERATED_LEN x D
                    past_gen_actions = past_gen_actions.unsqueeze(0)  # Add batch dim.
                    with torch.inference_mode():
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            out = self._policy(
                                qpos_in, img, actions=past_gen_actions, prompt=prompt, is_train=False
                            )  # 1 x q x D
                    self._all_actions[q,:] = out[0,-1,:]
            else:
                with torch.inference_mode():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out = self._policy(
                            qpos_in, img, prompt=prompt, is_train=False
                        )  # 1 x QUERY_LEN x D
                self._all_actions = out[0]  # QUERY_LEN x D

        if self._cfg.ROLLOUT.TEMPORAL_AGG:
            self._agg_table[self._t, self._t:self._t+self._cfg.MODEL.CHUNK_SIZE, :] = self._all_actions
            actions_for_t = self._agg_table[:, self._t, :]
            populated_actions = torch.all(actions_for_t != torch.inf, axis=1)
            actions_for_t = actions_for_t[populated_actions]
            k = self._cfg.ROLLOUT.TEMPORAL_AGG_K
            exp_weights = np.exp(-k * np.arange(len(actions_for_t)))
            exp_weights = exp_weights / exp_weights.sum()
            exp_weights = torch.from_numpy(exp_weights).to(device=self._device, dtype=torch.float32).unsqueeze(dim=1)
            normed_action = (actions_for_t * exp_weights).sum(dim=0)

        elif self._cfg.ROLLOUT.CONFIDENCE_AGG:
            assert any([is_tele==1 for is_tele in self._cfg.TASK.TELEOP_JOINTS]), f"Need one teleop joint at least."

            self._agg_table[self._t, self._t:self._t+self._cfg.MODEL.CHUNK_SIZE, :] = self._all_actions
            qpos_til_t = self._history_qpos[:self._t+1, :]  # t x D
            actions_til_t = self._agg_table[:, :self._t+1, :]  # EP_LEN x t x D
            actions_for_t = self._agg_table[:, self._t, :]  # EP_LEN x D
            populated_actions = torch.all(actions_for_t != torch.inf, axis=1)
            actions_til_t = actions_til_t[populated_actions, :, :]  # REL_LEN x t x D
            actions_for_t = actions_for_t[populated_actions]  # REL_LEN x D
            # error = torch.pow((actions_til_t - qpos_til_t), 2)
            error = (actions_til_t - qpos_til_t).abs()
            predicted = torch.all(error != torch.inf, axis=2)
            not_predicted = torch.all(error == torch.inf, axis=2)
            error[not_predicted] = 0.0
            tele_joints = torch.tensor(self._cfg.TASK.TELEOP_JOINTS, dtype=torch.bool, device=self._device)
            error = error * tele_joints
            agg_error = error.sum(axis=(1,2)) / (predicted.sum(dim=1) * tele_joints.sum())
            order = torch.argsort(agg_error)
            # agg_error = agg_error.mean(axis=1)  # Average across state dim.
            # agg_error = torch.pow(error, 16).sum(axis=(1,2)) / predicted.sum(axis=1)
            # conf = 1 / (agg_error + torch.finfo(torch.float32).eps)
            # conf = 1 / torch.exp(20 * agg_error)
            conf = torch.exp(-0.05 * order)
            weights = conf / conf.sum()
            # print(weights)
            weights = weights.unsqueeze(dim=1)  # Already on correct device
            normed_action = (actions_for_t * weights).sum(dim=0)

        else:
            normed_action = self._all_actions[self._t % self._query_period, :]  # D

        # Save the action in history.
        self._history_actions[self._t,:] = normed_action

        # Denormalize the actions to allow for interpretation as command.
        normed_action = normed_action.cpu().detach().numpy()
        autonomous_action = self._dataset_stats.denorm_actions(
            normed_action, method=self._cfg.MODEL.ACTION_NORM
        )

        self._t += 1  # Increment _t

        # Log the passed time.
        # self._log.debug(f"Step took {timer.tick().elapsed(hms=True, as_str=True)}.")
        # print(f"Step took {timer.tick().elapsed(hms=True, as_str=True)}.")

        return autonomous_action
    
    def __str__(self):
        """
        Print out the controller's architecture.
        """
        return str(self._policy)