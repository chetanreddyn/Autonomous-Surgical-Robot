# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
CVAE model and criterion classes.
"""
import torch
from torch import nn
from torch.nn import Transformer
from torch.autograd import Variable
import logging
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer, Transformer
from typing import List, Any

import numpy as np
from yacs.config import CfgNode

import IPython
e = IPython.embed


def reparametrize(mu: torch.Tensor, logvar: torch.Tensor):
    """
    Sample a vector centered around mu using a std defined by logvar.

    Args:
        mu: the expected value.
        logvar: the log of the variance.
    """
    std = logvar.div(2).exp()  # = sqrt( e ^ log )
    eps = Variable(std.data.new(std.size()).normal_())  # Sample a value on a normal dist N(0,1)
    return mu + std * eps


def get_sinusoid_encoding_table(n_position: int, d_hid: int):
    """
    Create a sinusoidal encoding tensor.

    Args:
        n_positions: The maximum token length. The length of the encoding.
        d_hid: The dimension of each sinusoidal embedding token.
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def get_prompt_len(cfg: CfgNode):
    """
    A simple function to get the prompt length given a CfgNode.
    """
    return cfg.MODEL.ACTION_PROMPT.LEN if cfg.MODEL.ACTION_PROMPT.ENABLE else 0


class CVAE(nn.Module):
    """This is the CVAE module that performs action chunk predictions"""
    def __init__(
            self,
            backbones: List[Any],
            transformer: Transformer,
            style_encoder: TransformerEncoder,
            state_dim: int,
            chunk_size: int,
            camera_names: List[str], 
            guess_encoder = None,
            num_guess_queries = None,
            state_history_len: int = 1,
            decoder_causal_mask: bool = False,
            prompt_len: int = 0,
        ):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            encoder: the style encoder.
            state_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         CVAE can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            state_history_len: The amount of state (qpos) history to input.
            decoder_causal_mask: whether to apply a causal mask for the decoder.
            prompt_length: the expected length of the prompt to the transformer decoder.
        """
        super().__init__()
        # Class Attributes.
        self.chunk_size = chunk_size
        self.camera_names = camera_names
        self.hidden_dim = transformer.d_model
        self.state_history = state_history_len
        self.prompt_len = prompt_len
        self.state_dim = state_dim

        # Action Encoders
        self.style_encoder = style_encoder  # previously just named "encoder"
        self.guess_encoder = guess_encoder

        # Image Backbones.
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
        else:
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # QPOS projection for transformer input.
        self.encoder_qpos_proj = nn.Linear(state_dim, self.hidden_dim)  # Previously named input_proj_robot_state

        # CVAE Transformer.
        self.transformer: Transformer = transformer
        
        # CVAE Transformer Encoder Variables.
        if decoder_causal_mask:
            mask_size = self.prompt_len + self.chunk_size
            decoder_causal_mask = nn.Transformer.generate_square_subsequent_mask(mask_size)  # mask_size x mask_size
            self.register_buffer('decoder_causal_mask', decoder_causal_mask)
            # self.decoder_causal_mask = torch.zeros_like(self.decoder_causal_mask)
            self.action_embed = nn.Linear(state_dim, self.hidden_dim)
        else:
            self.decoder_causal_mask = None
            self.action_embed = None
 
        # if prompt_len > 0:
        self.prompt_embed = nn.Linear(state_dim, self.hidden_dim)
        
        # Style Encoder Parameters
        self.style_latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, self.hidden_dim) # extra cls token embedding
        self.style_encoder_action_proj = nn.Linear(state_dim, self.hidden_dim) # project action to embedding (previously named encoder_action_proj)
        self.style_encoder_qpos_proj = nn.Linear(state_dim, self.hidden_dim)  # project qpos to embedding (previously named encoder_joint_proj)
        self.latent_proj = nn.Linear(self.hidden_dim, self.style_latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+state_history_len+chunk_size, self.hidden_dim)) # [CLS], qpos, a_seq
        self.latent_out_proj = nn.Linear(self.style_latent_dim, self.hidden_dim) # project latent sample to embedding

        self.additional_pos_embed = nn.Embedding(1+state_history_len, self.hidden_dim) # learned position embedding for proprio and style
        
        # Pre-allocate padding mask buffer to avoid repeated tensor creation
        self.register_buffer('cls_joint_is_pad_template', torch.full((1, 1+state_history_len), False))

        self.guess_pos_embed = None
        if guess_encoder:
            self.guess_cls_embed = nn.Embedding(1, self.hidden_dim) # extra cls token embedding
            self.guess_action_proj = nn.Linear(state_dim, self.hidden_dim) # project guess action to embedding
            self.guess_joint_proj = nn.Linear(state_dim, self.hidden_dim)  # project qpos to embedding
            self.guess_latent_proj = nn.Linear(self.hidden_dim, self.style_latent_dim*2) # project hidden state to latent std, var
            self.register_buffer('guess_pos_table', get_sinusoid_encoding_table(1+state_history_len+num_guess_queries, self.hidden_dim)) # [CLS], qpos, a_seq
            self.guess_latent_out_proj = nn.Linear(self.style_latent_dim, self.hidden_dim) # project latent sample to embedding
            self.guess_pos_embed = nn.Embedding(1, self.hidden_dim)  # Learned position embedding for guess feature.

        # CVAE Transformer Decoder Variables.
        self.query_embed = nn.Embedding(self.prompt_len+self.chunk_size, self.hidden_dim)  # Decoder tgt pos embed.
        self.action_head = nn.Linear(self.hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(self.hidden_dim, 1)

    def forward(
            self, qpos, image, env_state, 
            actions=None, is_pad=None, guess_actions=None, 
            prompt=None, is_train: bool = True
        ):
        """
        qpos: (batch, qpos_hist_len, qpos_dim)
        image: (batch x chunk_size x num_cams x channel x height x width)
        env_state: unused
        actions: the action labels. (batch, seq, state_dim)
        guess: a guess for the actions (batch, seq, state_dim)
        prompt: the prompt used to predict the following actions.  (batch, prompt_len, state_dim)
        is_train: whether we are training.
        """
        # if len(qpos.shape) == 2:
        #     qpos = torch.unsqueeze(qpos, axis=1)
        bs, _, _ = qpos.shape
        
        # Obtain style feature
        if is_train:
            # Compute the style feature.
            style_encoder_output = self.forward_style_encoder(qpos, actions, is_pad)
            
            # Interpret the style encoding as mu and log(var)
            style_latent = self.latent_proj(style_encoder_output)
            mu = style_latent[:, :self.style_latent_dim]
            logvar = style_latent[:, self.style_latent_dim:]

            # Sample a style feature from the N(mu, e^[logvar]) distribution.
            latent_sample = reparametrize(mu, logvar)
            style_feat = self.latent_out_proj(latent_sample)  # (bs, hidden_dim)
            style_feat = torch.unsqueeze(style_feat, dim=1)  # (bs, 1, hidden_dim)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.style_latent_dim], dtype=torch.float32, device=qpos.device)
            style_feat = self.latent_out_proj(latent_sample)  # (bs, hidden_dim)
            style_feat = torch.unsqueeze(style_feat, dim=1)  # (bs, 1, hidden_dim)

        guess_feat = None
        if self.guess_encoder:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.guess_action_proj(guess_actions) # (bs, seq, hidden_dim)
            qpos_embed = self.guess_joint_proj(qpos)  # (bs, hist_len, hidden_dim)
            cls_embed = self.guess_cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            # cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            # is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.guess_pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.guess_encoder(encoder_input, pos=pos_embed)#, src_key_padding_mask=is_pad) # TODO: Do we want masking turned off?
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.guess_latent_proj(encoder_output)
            g_mu = latent_info[:, :self.style_latent_dim]
            g_logvar = latent_info[:, self.style_latent_dim:]
            latent_sample = reparametrize(g_mu, g_logvar)
            guess_feat = self.guess_latent_out_proj(latent_sample)
            guess_feat = torch.unsqueeze(guess_feat, dim=1)  # (bs, 1, hidden_dim)

        # Compute the image features using vision backbones.
        img_feats, img_pos = self.forward_vision_backbone(image)
        # import pdb; pdb.set_trace()

        # Compute proprioception features.
        proprio_feats = self.encoder_qpos_proj(qpos)  # (bs, hist_len, 1)
        
        # Prepare the train-time tgt (action predicted shifted right)
        tgt, tgt_pos_embed, causal_mask = self.assemble_tgt(bs, prompt, actions)

        # Pass the data to the encoder.
        # Potential bug in hs, see: https://github.com/tonyzhaozh/act/issues/25
        hs = self.transformer(
            img_feats,
            None, 
            tgt_pos_embed,
            img_pos, 
            style_feat=style_feat, 
            proprio_feats=proprio_feats, 
            additional_pos_embed=self.additional_pos_embed.weight,
            guess_feat=guess_feat,
            guess_pos=self.guess_pos_embed.weight if self.guess_pos_embed else None,
            tgt=tgt,
            tgt_mask=causal_mask,
        )[-1] # Potential bug fix, index with [-1] instead of [0] # B x CS x D  
        # print(hs.shape)
        a_hat = self.action_head(hs)  # Pass tokens through action head.

        is_pad_hat = self.is_pad_head(hs)  # Predict whether tokens are padded.
        return a_hat, is_pad_hat, [mu, logvar]


    def forward_style_encoder(self, qpos, actions, is_pad) -> torch.Tensor:
        """
        Perform a forward pass on the style encoder. This function will assemble the style encoder
        input consisting of [cls, qpos, actions] and return the [cls] token at the encoder's output.

        B = Batch Size
        HL = History Length
        JD = Joint Dimensions
        CS = Prediction chunk size
        D = Hidden Dimension

        Args:
            qpos: the joint states, SHAPE: B x HL x JD
            actions: the action labels, SHAPE: B x CS x JD

        Returns:
            style_vec: the style embedding, SHAPE: [1+HL+CS] x B x D
        """
        B = qpos.shape[0]
        # project action sequence to embedding dim, and concat with a CLS token
        action_embed = self.style_encoder_action_proj(actions)  # (B, seq, hidden_dim)
        qpos_embed = self.style_encoder_qpos_proj(qpos)  # (B, hist_len, hidden_dim)
        cls_embed = self.cls_embed.weight # (1, hidden_dim)
        cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(B, 1, 1) # (B, 1, hidden_dim)
        style_encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (B, 1+hist+seq, hidden_dim)
        style_encoder_input = style_encoder_input.permute(1, 0, 2) # (seq+1, B, hidden_dim)
        # do not mask cls token
        cls_joint_is_pad = self.cls_joint_is_pad_template.expand(B, -1) # Reuse pre-allocated buffer
        is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (B, 1+hist+seq)
        # obtain position embedding
        pos_embed = self.pos_table.clone().detach()
        pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
        # query model
        style_encoder_output = self.style_encoder(
            style_encoder_input, pos=pos_embed, src_key_padding_mask=is_pad
        )[0]  # take cls output only
        return style_encoder_output


    def forward_vision_backbone(self, image):
        """
        Compute the forward pass on the vision backbones.

        B: batch size
        NC: number of cameras
        C: image channels
        H: image height
        W: image width
        D: hidden dimension

        Args:
            image: the visual input, SHAPE: B x NC x C x H x W

        Returns:
            image_feats: features extracted by the backbone, SHAPE: B x 1 x D x H x (W*NC)
            image_pos: position embeddings, SHAPE: B x 1 x D x H x (W*NC)
        """
        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        for cam_id, cam_name in enumerate(self.camera_names):
            # Shared backbone
            if len(self.backbones) == 1:
                img_feat, img_pos = self.backbones[0](image[:,cam_id])
            # Unique backbones for each view.
            else:
                img_feat, img_pos = self.backbones[cam_id](image[:,cam_id])

            img_feat, img_pos = img_feat[0], img_pos[0]  # take the last layer values
            all_cam_features.append(self.input_proj(img_feat))
            all_cam_pos.append(img_pos)
        
        # fold camera dimension into width dimension
        img_feats = torch.cat(all_cam_features, axis=3)  # SHAPE: B x D x H x (W*NC)
        img_feats = torch.unsqueeze(img_feats, 1)  # SHAPE: B x 1 x D x H x (W*NC)
        img_pos = torch.cat(all_cam_pos, axis=3)
        img_pos = torch.unsqueeze(img_pos, 1)  # SHAPE: B x 1 x D x H x (W*NC)

        return img_feats, img_pos


    def assemble_tgt(
            self, batch_size: int, prompt: torch.Tensor=None, actions: torch.Tensor=None
        ) -> torch.Tensor:
        """
        Given some variables, assemble the target sequence.

        Args:
            batch_size: the batch size
            prompt: the prompt sequence - SHAPE B x PL x SD
            actions: the action sequence - SHAPE B x CS x SD

        Returns:
            tgt: the target vector - SHAPE B x (PL+CS) x HD
            tgt_pos: the target pos embed - SHAPE B x (PL+CS) x HD
            tgt_mask: a causal mask used within the decoder. Returns None when not self.is_causal.
        """
        # Get device for tensor creation
        device = next(self.parameters()).device
        
        if self.prompt_len > 0:  # Expecting a prompt
            assert prompt is not None
        else:
            prompt = torch.zeros((batch_size, 0, self.state_dim), device=device)

        # Form the tgt vector
        tgt_sz = self.prompt_len + self.chunk_size
        if self.is_causal:
            concat_seq = [self.bos(bsz=batch_size), prompt, actions]
            state_tgt = torch.cat(concat_seq, dim=1)[:,:tgt_sz,:]  # B x TS x SD
            tgt = self.action_embed(state_tgt)
        else:
            concat_seq = [
                self.prompt_embed(prompt),
                torch.zeros((batch_size, self.chunk_size, self.hidden_dim), device=device)
            ]
            tgt = torch.cat(concat_seq, dim=1)

        # Form the trg_pos_embed vector
        if self.is_causal:
            # tgt_pos_embed = self.sine_pos_embed(self.prompt_len+self.chunk_size, self.hidden_dim)
            tgt_pos_embed = self.query_embed.weight
            tgt_pos_embed = tgt_pos_embed[:tgt.shape[1],:]
        else:
            tgt_pos_embed = self.query_embed.weight  # Action chunking uses learned weights.

        # Prepare the causal mask
        if self.is_causal:
            # TODO: potentially modify mask for prompt tokens not to be causal.
            tgt_mask = self.decoder_causal_mask[:tgt.shape[1], :tgt.shape[1]]
        else:
            tgt_mask = None

        return tgt, tgt_pos_embed, tgt_mask


    def bos(self, bsz: int = 1) -> torch.Tensor:
        """
        A BOS token for the causal decoding.

        Returns:
            bos: zero tensor of shape B x 1 x SD
        """
        device = next(self.parameters()).device
        return torch.zeros(bsz, 1, self.state_dim, device=device)
    
    def sine_pos_embed(self, l: int = 16, d: int = 512) -> torch.Tensor:
        """
        Sinusoidal positional embedding vector.

        Returns:
            sine_pos_embed: sinusoidal pos embedding with shape L x D
        """
        device = next(self.parameters()).device
        return get_sinusoid_encoding_table(l, d)[0].to(device)
        
    @property
    def is_causal(self) -> bool:
        """
        Whether the CVAE decoder is causal or chunking.
        """
        return bool( self.decoder_causal_mask is not None )
    

class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            chunk_size: number of object queries, ie detection slot. This is the maximal number of objects
                         CVAE can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth) -> torch.nn.Module:
    """
    Return a multi-layer perceptron module with hidden_depth.
    """
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(cfg: CfgNode) -> torch.nn.Module:
    """
    Build a transformer encoder module according to the configuration.
    """
    d_model = cfg.MODEL.HIDDEN_DIM # 256
    dropout = cfg.MODEL.DROPOUT # 0.1
    nhead = cfg.MODEL.N_HEADS # 8
    dim_feedforward = cfg.MODEL.FF_DIM # 2048
    num_encoder_layers = cfg.MODEL.N_LAYERS_1 # 4 # TODO shared with VAE decoder
    normalize_before = cfg.MODEL.PRE_NORM # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build_backbones(cfg: CfgNode) -> List[torch.nn.Module]:
    """
    Build the image backbones used to process camera input.
    """
    backbones = []
    if cfg.MODEL.SHARED_BACKBONE:
        backbone = build_backbone(cfg)
        backbones.append(backbone)
    else:
        for cam in cfg.TASK.CAM_NAMES:
            backbones.append(build_backbone(cfg))
    return backbones


def build_cvae(cfg: CfgNode) -> torch.nn.Module:
    """
    Build a randomly initialized CVAE model and return the module object.

    Args:
        cfg: the configuration node to build the CVAE off of.
    """
    log = logging.getLogger()
    
    backbones = build_backbones(cfg)
    transformer = build_transformer(cfg)
    encoder = build_encoder(cfg)

    guess_encoder = None
    if "guess" in cfg.MODEL.POLICY.lower():
        # Build the encoder for guessed actions.
        guess_encoder = build_encoder(cfg)

    prompt_len = get_prompt_len(cfg)

    model = CVAE(
        backbones,
        transformer,
        encoder,
        state_dim=cfg.MODEL.STATE_DIM,
        chunk_size=cfg.MODEL.CHUNK_SIZE,
        camera_names=cfg.TASK.CAM_NAMES,
        guess_encoder=guess_encoder,
        num_guess_queries=cfg.MODEL.GUESS_CHUNK_SIZE,
        state_history_len=cfg.MODEL.STATE_HISTORY,
        decoder_causal_mask=cfg.MODEL.CAUSAL_DECODING,
        prompt_len=prompt_len
    )

    # Summarize the backbone size.
    log.info(">>> Logging Backbone Statistics")
    log_model_size(model.backbones, log)

    # Summarize the entire model's size.
    log.info(">>> Logging CVAE Statistics")
    log_model_size(model, log)

    return model


def log_model_size(model: torch.nn.Module, log: logging.Logger = None):
    """
    Log the model's num params, num train params, num frozen params.

    Args:
        model: the torch.nn.Module
        log: the logger to use, if None just print.
    """
    disp_func = log.info if log else print

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    disp_func("Total number of trainable parameters: %.2fM" % (n_parameters/1e6,))
    disp_func("Total number of untrainable (frozen) parameters: %.2fM" % (n_frozen/1e6,))
    disp_func("Total number of parameters: %.2fM" %((n_frozen+n_parameters)/1e6))



def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

