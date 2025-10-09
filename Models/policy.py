import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from typing import Union, Dict, List
from yacs.config import CfgNode

from mlcore.registry import Registry
from AdaptACT.config.defaults import get_cfg
from AdaptACT.utils.custom_logging import get_logger
from AdaptACT.cvae.models.cvae import CVAE
from AdaptACT.cvae.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer, build_optimizer
from AdaptACT.cvae.models.cvae import get_prompt_len

from AdaptACT.utils.checkpoints import load_checkpoint_weights

import IPython
e = IPython.embed

"""
Registry of available policies. Utility functions to get and build policies.
"""


POLICY_REGISTRY = Registry("RL Policy")


def get_policy_cls(spec_str: str):
    """
    Get the policy class.
    """
    return POLICY_REGISTRY.get(spec_str)


def make_policy_obj(spec_str: str, *args, **kwargs) -> nn.Module:
    """
    Make the policy object.
    """
    return POLICY_REGISTRY.get(spec_str)(*args, **kwargs)


def get_loss_weighting(strategy: str, seq_len: str, *args, device: str = None) -> torch.Tensor:
    """
    Get the vector used for weighting loss across the prediction sequence.

    Args:
        strategy: the loss weighting strategy to use.
        seq_len: the prediction / label sequence length.
        args: additional loss weighting positional arguments used to set up the strategy.
        device: the device to place the tensor on.

    Return:
        weight_tensor: a weight tensor with shape [1, seq_len, 1]. The tensor must add to 1.0
    """
    strategy = strategy.lower()
    if strategy == "constant":  # constant weighting
        w = torch.ones([seq_len])
        w = w / torch.sum(w)
    elif strategy == "cosine":  # cosine decay weighting
        x = torch.linspace(0, seq_len, seq_len)
        a = torch.pi / seq_len
        w = 0.5 * torch.cos(a * x) + 0.5
        w = w / torch.sum(w)
    else:
        raise NotImplementedError(f"Loss weighting strategy '{strategy}' was not implemented")
    # Device handling will be done by register_buffer in the calling module
    return w.reshape([1,-1,1])  # Returns shape (1, seq_len, 1)


@POLICY_REGISTRY.register()
class ACTPolicy(nn.Module):
    """
    Action Chunking Transformer (ACT) Policy.
    """
    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.log = get_logger()
        # Build the CVAE model and optimizer.
        self.model, self.optimizer = build_ACT_model_and_optimizer(cfg)
        
        # Build useful parameters.
        self.kl_weight = cfg.TRAIN.KL_WEIGHT
        self.img_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        self.prompt_len = get_prompt_len(cfg)

        input_mask = torch.tensor(cfg.MODEL.INPUT_MASK).bool()
        if cfg.MODEL.LEFT_STATE_ONLY:
            self.log.warning(
                "DEPRECATED: MODEL.LEFT_STATE_ONLY=True will be deprecated. "
                "Use MODEL.INPUT_MASK=[0,0,0,0,0,0,0,1,1,1,1,1,1,1] instead."
            )
            input_mask = torch.zeros((cfg.MODEL.STATE_DIM,)).bool()
            input_mask[7:] = True # for masking the right arm out of the state dim.
        self.register_buffer('input_mask', input_mask)

        # Achieve different qpos history lengths by arm.
        if cfg.MODEL.STATE_HISTORY_PER_ARM == (0,):
            state_len_mask = torch.ones((cfg.MODEL.STATE_HISTORY, cfg.MODEL.STATE_DIM))
        else:
            state_len_mask = torch.zeros((cfg.MODEL.STATE_HISTORY, cfg.MODEL.STATE_DIM))
            left_len, right_len = cfg.MODEL.STATE_HISTORY_PER_ARM
            if left_len > 0:
                state_len_mask[-left_len:,:7] = 1.0
            if right_len > 0:
                state_len_mask[-right_len:,7:] = 1.0
        self.register_buffer('state_len_mask', state_len_mask)

        # Only calculate loss based on the left arm's prediction and ground truth.
        loss_mask = torch.tensor(cfg.MODEL.LOSS_MASK).bool()
        if cfg.MODEL.LEFT_LOSS_ONLY:
            self.log.warning(
                "DEPRECATED: MODEL.LEFT_LOSS_ONLY=True will be deprecated. "
                "Use MODEL.LOSS_MASK=[0,0,0,0,0,0,0,1,1,1,1,1,1,1] instead."
            )
            loss_mask = torch.zeros((cfg.MODEL.STATE_DIM,)).bool()
            loss_mask[7:] = True  # for masking the right arm out of the loss.
        self.register_buffer('loss_mask', loss_mask)

        loss_weight = get_loss_weighting(
            cfg.TRAIN.LOSS_WEIGHT.STRATEGY, 
            cfg.MODEL.CHUNK_SIZE,
            *cfg.TRAIN.LOSS_WEIGHT.PARAMS
        )
        self.register_buffer('loss_weight', loss_weight)

        self.log.info(f'KL Weight {self.kl_weight}')
        self.log.info(f"Finished building {self.__class__.__name__}")

    def __call__(
            self, 
            qpos,
            image,
            actions=None,
            is_pad=None,
            prompt=None,
            is_train: bool = True,
        ):
        """
        During training, return loss dictionary.
        During rollout, return predicted actions.

        Args:
            qpos: batch, qpos_hist_len, qpos_dim
            image: the visual data
            actions: the labels (actions).
            tgt: used for causal auto-regressive inference. (full set of actions)
            prompt: the prompt actions for the decoder.

        Return:
            loss_dict if is_train, otherwise return the predicted actions.
        """
        env_state = None
        image = self.img_norm(image)
        qpos = qpos * ~self.input_mask  # Mask out right arm, if only using left arm to predict.
        qpos = qpos * self.state_len_mask  # Mask out some states from input.

        # Inference Mode. Just return the prediction!
        if not is_train:
            """
            For causal inference: actions are already generated sequence.
            For chunking inference: actions should be None.
            """
            a_hat, _, (_, _) = self.model(
                qpos, 
                image, 
                None, 
                actions=actions, 
                prompt=prompt, 
                is_train=False
            )
            a_hat = a_hat[:,self.prompt_len:,:]  # Shave off the prompt tokens, SHAPE: B x CS x SD
            return a_hat * ~self.loss_mask  # zero out not-predicted states
        
        # Training Mode. Forward pass and calculate losses!
        else:
            """
            For causal training: actions are target predictions and inputs to the style encoder.
            For chunking training: actions are inputs to the style encoder.
            """
            actions = actions[:, :self.model.chunk_size] # * self.loss_mask
            is_pad = is_pad[:, :self.model.chunk_size]
            a_hat, _, (mu, logvar) = self.model(
                qpos, 
                image, 
                None, 
                actions=actions, 
                is_pad=is_pad, 
                prompt=prompt,
                is_train=True
            )
            a_hat = a_hat[:,self.model.prompt_len:,:]  # Shave off the prompt tokens

            # Calculate loss on prompt output tokens.
            # if prompt is not None:
            #     labels = torch.cat([prompt, actions], dim=1)  # B PL+CS SD
            #     is_pad = torch.cat([
            #         torch.zeros((actions.shape[0], self.model.prompt_len), dtype=bool).cuda(),
            #         is_pad
            #     ], dim=1)
            # else:
            #     labels = actions
            
            # Get the l1 loss over labels.
            vel = actions[:,1:] - actions[:,:-1]
            acc = vel[:,1:] - vel[:,:-1]
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')  # B L SD
            all_l1 = all_l1 * self.loss_weight * ~self.loss_mask
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).sum(dim=1).mean()

            # KL Loss on style vec.
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = {
                "l1": l1,
                "kl": total_kld[0],
                "loss": l1 + self.kl_weight * total_kld[0]
            }
            return loss_dict
           
    def configure_optimizers(self):
        return self.optimizer


@POLICY_REGISTRY.register()
class FrozenGuessACTPolicy(nn.Module):
    """
    Implementation of the ACT model but accepts a guess action sequence.
    """
    def __init__(self, cfg: CfgNode):
        super().__init__()

        self.log = get_logger()

        guesser_cfg = get_cfg()
        guesser_cfg.merge_from_file(cfg.MODEL.GUESSER_CFG)  # build the guesser config
        self.guesser = make_policy_obj(guesser_cfg.MODEL.POLICY, guesser_cfg)  # Build the guesser
        self.guesser.load_state_dict(torch.load(cfg.MODEL.GUESSER_CHKPT))  # Load the checkpoint
        self.log.info(f"Successfully loaded stage-1 from checkpoint: {cfg.MODEL.GUESSER_CHKPT}")
        # Freeze the guesser
        self.guesser.eval()
        for param in self.guesser.parameters():
            param.requires_grad = False
        self.log.info(f"Successfully froze stage-1 model.")

        model, optimizer = build_ACT_model_and_optimizer(cfg)
        self.model: CVAE = model # CVAE transformer
        self.optimizer = optimizer
        self.kl_weight = cfg.TRAIN.KL_WEIGHT
        self.img_norm = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

        # Only use the left arm's qpos for input. Only use the left arm's prediction for loss calc
        if not cfg.MODEL.LEFT_STATE_ONLY:
            input_mask = torch.ones((cfg.MODEL.STATE_DIM,))
        else:
            input_mask = torch.ones((cfg.MODEL.STATE_DIM,))
            input_mask[7:] = 0.0  # for masking the right arm out of the state dim.
        self.register_buffer('input_mask', input_mask)

        # Only calculate loss based on the left arm's prediction and ground truth.
        if not cfg.MODEL.LEFT_LOSS_ONLY:
            loss_mask = torch.ones((cfg.MODEL.STATE_DIM,))
        else:
            loss_mask = torch.ones((cfg.MODEL.STATE_DIM,))
            loss_mask[7:] = 0.0  # for masking the right arm out of the state dim.
        self.register_buffer('loss_mask', loss_mask)

        self.log.info(f'KL Weight {self.kl_weight}')
        self.log.info(f"Finished building {self.__class__.__name__}")

    def __call__(self, qpos, image, labels=None, is_pad=None):
        """
        During training, return loss dictionary.
        During rollout, return predicted actions.
        """
        env_state = None
        image = self.img_norm(image)

        # Perform the "Guess" Policy
        guess_a = self.guesser(qpos, image)
        guess_a = guess_a.detach()  # Break the "guessed action" from the comp graph.

        qpos = qpos * self.input_mask  # Mask out right arm, if only using left arm to predict.

        if labels is not None: # training time
            labels = labels[:, :self.model.chunk_size] * self.loss_mask
            is_pad = is_pad[:, :self.model.chunk_size]
            a_hat, is_pad_hat, (mu, logvar) = self.model(
                qpos, image, env_state, labels, is_pad, guess_actions=guess_a
            )
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(labels, a_hat*self.loss_mask, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state, guess_actions=guess_a)
            return a_hat * self.loss_mask

    def configure_optimizers(self):
        return self.optimizer


@POLICY_REGISTRY.register()
class MultiAgentPolicy(nn.Module):
    """
    Torch model which implements multiple agents using separate ACT policies.
    """
    def __init__(self, cfg: CfgNode):
        """
        Set up the MultiAgentACTPolicy
        """
        super().__init__()

        self.log = get_logger()
        self._cfg = cfg

        self._agent_cfgs = []
        self._agent_masks = []
        for i, agent_cfg_path in enumerate(cfg.MODEL.MULTI_AGENT.CFGS):
            self.log.info(f"Obtaining agent {i} cfg from: {agent_cfg_path}")
            agent_cfg = get_cfg()
            agent_cfg.merge_from_file(agent_cfg_path)
            assert len(agent_cfg.MODEL.MULTI_AGENT.CFGS) == 0, f"Agent policies cannot also be multi-agent"
            self._agent_cfgs.append(agent_cfg)
            self._agent_masks.append(agent_cfg.MODEL.LOSS_MASK)
        agent_masks = torch.tensor(self._agent_masks).bool()  # NUM_AGENTS x STATE_DIM
        self.register_buffer('_agent_masks', agent_masks)

        # Asserting uniformity of certain output dimensions.
        assert len(set(a_cfg.MODEL.STATE_DIM for a_cfg in self._agent_cfgs)) == 1
        assert len(set(a_cfg.MODEL.CHUNK_SIZE for a_cfg in self._agent_cfgs)) == 1

        # Store each model in a policy object.
        self.agents = torch.nn.ModuleList(
            make_policy_obj(agent_cfg.MODEL.POLICY, a_cfg)
            for a_cfg in self._agent_cfgs
        )
        self.optimizer = build_optimizer(cfg, self)
        self.log.info(f"Finished building {self.__class__.__name__}")
    
    def load_all_agent_checkpoints(self, pretrain_paths: List[str] = None):
        """
        Load all single agent checkpoints into the corresponding agents.

        Args:
            pretrained_paths: If given, use these checkpoint files. If not given, use agents' 
                EXEC.PRETRAIN_PATHs. "pretrain_paths" should be share corresponding indicies
                with the self.agents.

        Return:
            full_load: True, if all model weights and checkpoint weights match. False if otherwise.
        """
        if not pretrain_paths:
            pretrain_paths = [a_cfg.EXEC.PRETRAIN_PATH for a_cfg in self._agent_cfgs]
        
        assert len(pretrain_paths) == len(self.agents), "Num policy agent != Num checkpoints"

        full_load = True
        for pretrain_path, agent_module in zip(pretrain_paths, self.agents):
            agent_full_load = load_checkpoint_weights(model=agent_module, path=pretrain_path)
            full_load = full_load and agent_full_load

        return full_load

    def __call__(
            self, 
            qpos,
            image,
            actions=None,
            is_pad=None,
            prompt=None,
            is_train: bool = True,
        ):
        """
        Execute a forward pass. Compute losses if training.
        """
        if not is_train:  # Inference Mode.
            a_hats = torch.stack([
                agent(
                    qpos=qpos, 
                    image=image, 
                    actions=actions,
                    is_pad=is_pad,
                    prompt=prompt,
                    is_train=False
                )
                for agent in self.agents
            ], dim=1)  # B x NUM_AGENTS x CS x SD

            counts = (~self._agent_masks).sum(dim=0)  # NUM_AGENTS x SD -> SD
            sums = a_hats.sum(dim=1)  # B x NUM_AGENTS x CS x SD -> B x CS x SD
            avg_a_hat = sums / counts
            # import pdb; pdb.set_trace()
            return avg_a_hat  # B x CS x SD

        else:  # Training Mode.
            loss_dicts = [
                agent(
                    qpos=qpos, 
                    image=image, 
                    actions=actions,
                    is_pad=is_pad,
                    prompt=prompt,
                    is_train=is_train
                )
                for agent in self.agents
            ]
            avg_loss = {
                "l1": sum([ld["l1"] for ld in loss_dicts]) / len(loss_dicts),
                "kl": sum([ld["kl"] for ld in loss_dicts]) / len(loss_dicts),
                "loss": sum([ld["loss"] for ld in loss_dicts]) / len(loss_dicts)
            }
            return avg_loss
        
    def configure_optimizers(self):
        return self.optimizer


@POLICY_REGISTRY.register()
class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, labels=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if labels is not None: # training time
            labels = labels[:, 0]
            a_hat = self.model(qpos, image, env_state, labels)
            mse = F.mse_loss(labels, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
