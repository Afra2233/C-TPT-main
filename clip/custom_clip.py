
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip import load, tokenize
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
from data.imagnet_prompts import imagenet_classes
from data.fewshot_datasets import fewshot_datasets
from data.cls_to_names import *

import ipdb

_tokenizer = _Tokenizer()

DOWNLOAD_ROOT='~/.cache/clip'
# DOWNLOAD_ROOT='~/.cache/clip'

# ==========================================================================================
def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if 'vision_encoder_state_dict' in ckpt and isinstance(ckpt['vision_encoder_state_dict'], dict):
            print("[INFO] Using ckpt['vision_encoder_state_dict']")
            return ckpt['vision_encoder_state_dict']

        for k in ['state_dict', 'model_state_dict', 'model', 'net']:
            if k in ckpt and isinstance(ckpt[k], dict):
                print(f"[INFO] Using ckpt['{k}']")
                return ckpt[k]
    return ckpt
# def _extract_state_dict(ckpt):
#     """
#     Robustly extract a state_dict from different checkpoint formats.
#     Supports keys like:
#       - ckpt['state_dict']
#       - ckpt['model']
#       - ckpt['model_state_dict']
#       - raw state_dict
#     """
#     if isinstance(ckpt, dict):
#         for k in ['state_dict', 'model_state_dict', 'model', 'net']:
#             if k in ckpt and isinstance(ckpt[k], dict):
#                 return ckpt[k]
#     return ckpt
# ==========================================================================================


def _strip_prefix_if_present(state_dict, prefixes=('module.', 'model.', 'clip.')):
    new_sd = {}
    for k, v in state_dict.items():
        new_k = k
        for p in prefixes:
            if new_k.startswith(p):
                new_k = new_k[len(p):]
        new_sd[new_k] = v
    return new_sd

# ==================================================================
def load_clip_with_tecoa(arch, device, download_root, robust_ckpt_path=None):
    clip_model, embed_dim, preprocess = load(arch, device=device, download_root=download_root)

    if robust_ckpt_path is None:
        print("[INFO] Loading original CLIP weights.")
        return clip_model, embed_dim, preprocess

    print(f"[INFO] Loading TeCoA robust checkpoint from: {robust_ckpt_path}")
    ckpt = torch.load(robust_ckpt_path, map_location="cpu")

    print(f"[INFO] checkpoint type: {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"[INFO] checkpoint top-level keys: {list(ckpt.keys())[:20]}")
        print(f"[INFO] number of top-level keys: {len(ckpt.keys())}")

    if isinstance(ckpt, dict) and 'vision_encoder_state_dict' in ckpt:
        state_dict = ckpt['vision_encoder_state_dict']
        print("[INFO] Using ckpt['vision_encoder_state_dict']")
    else:
        state_dict = _extract_state_dict(ckpt)

    print(f"[INFO] extracted state_dict num keys: {len(state_dict)}")
    print("[INFO] first 30 extracted keys:")
    for i, k in enumerate(state_dict.keys()):
        if i >= 30:
            break
        print(k)

    state_dict = _strip_prefix_if_present(state_dict)

    visual_sd = clip_model.visual.state_dict()
    filtered_sd = {}
    skipped = []

    for k, v in state_dict.items():
        if k in visual_sd and visual_sd[k].shape == v.shape:
            filtered_sd[k] = v
        else:
            skipped.append(k)

    missing_in_ckpt = [k for k in visual_sd.keys() if k not in filtered_sd]
    msg = clip_model.visual.load_state_dict(filtered_sd, strict=False)

    print(f"[INFO] Loaded {len(filtered_sd)} matching visual keys from robust checkpoint.")
    print(f"[INFO] Skipped {len(skipped)} unmatched visual keys from checkpoint.")
    print(f"[INFO] Missing {len(missing_in_ckpt)} visual model keys not found in checkpoint.")
    print(f"[INFO] visual load_state_dict msg: {msg}")

    return clip_model, embed_dim, preprocess

# def load_clip_with_tecoa(arch, device, download_root, robust_ckpt_path=None):
#     """
#     Load standard CLIP architecture, then optionally overwrite weights
#     using a TeCoA adversarially fine-tuned checkpoint.
#     """
#     clip_model, embed_dim, preprocess = load(arch, device=device, download_root=download_root)

#     if robust_ckpt_path is None:
#         print("[INFO] Loading original CLIP weights.")
#         return clip_model, embed_dim, preprocess

#     print(f"[INFO] Loading TeCoA robust checkpoint from: {robust_ckpt_path}")
#     ckpt = torch.load(robust_ckpt_path, map_location=device)
#     state_dict = _extract_state_dict(ckpt)
#     state_dict = _strip_prefix_if_present(state_dict)

#     model_sd = clip_model.state_dict()
#     filtered_sd = {}
#     skipped = []

#     for k, v in state_dict.items():
#         if k in model_sd and model_sd[k].shape == v.shape:
#             filtered_sd[k] = v
#         else:
#             skipped.append(k)

#     missing_in_ckpt = [k for k in model_sd.keys() if k not in filtered_sd]

#     msg = clip_model.load_state_dict(filtered_sd, strict=False)

#     print(f"[INFO] Loaded {len(filtered_sd)} matching keys from robust checkpoint.")
#     print(f"[INFO] Skipped {len(skipped)} unmatched keys from checkpoint.")
#     print(f"[INFO] Missing {len(missing_in_ckpt)} model keys not found in checkpoint.")
#     print(f"[INFO] load_state_dict msg: {msg}")

#     return clip_model, embed_dim, preprocess
# ==========================================================================================
# ====================================================
class ClipImageEncoder(nn.Module):
    def __init__(self, device, arch="ViT-L/14", image_resolution=224, n_class=1000):
        super(ClipImageEncoder, self).__init__()
        # ==========================================================================================

        # clip, embed_dim, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)
        clip, embed_dim, _ = load_clip_with_tecoa(arch, device=device, download_root=DOWNLOAD_ROOT, robust_ckpt_path=None)
        # ==========================================================================================

        self.encoder = clip.visual
        del clip.transformer
        torch.cuda.empty_cache()
        
        self.cls_head = nn.Linear(embed_dim, n_class)
    
    @property
    def dtype(self):
        return self.encoder.conv1.weight.dtype

    def forward(self, image):
        x = self.encoder(image.type(self.dtype))
        output = self.cls_head(x)
        return output


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, classnames, batch_size=None, n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False, robust_ckpt_path=None):
        super().__init__()
        n_cls = len(classnames)
        self.learned_cls = learned_cls
        dtype = clip_model.dtype
        self.dtype = dtype
        self.device = clip_model.visual.conv1.weight.device
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.ctx_dim = ctx_dim
        self.batch_size = batch_size
        self.robust_ckpt_path = robust_ckpt_path

        # self.ctx, prompt_prefix = self.reset_prompt(ctx_dim, ctx_init, clip_model)

        if ctx_init:
            # use given words to initialize context vectors
            print("Initializing the contect with given words: [{}]".format(ctx_init))
            ctx_init = ctx_init.replace("_", " ")
            if '[CLS]' in ctx_init:
                ctx_list = ctx_init.split(" ")
                split_idx = ctx_list.index("[CLS]")
                ctx_init = ctx_init.replace("[CLS] ", "")
                ctx_position = "middle"
            else:
                split_idx = None
            self.split_idx = split_idx
            n_ctx = len(ctx_init.split(" "))
            prompt = tokenize(ctx_init).to(self.device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            print("Random initialization: initializing a generic context")
            ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        
        self.prompt_prefix = prompt_prefix
        
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        # batch-wise prompt tuning for test-time adaptation
        if self.batch_size is not None: 
            ctx_vectors = ctx_vectors.repeat(batch_size, 1, 1)  #(N, L, D)
        self.ctx_init_state = ctx_vectors.detach().clone()
        self.ctx = nn.Parameter(ctx_vectors) # to be optimized

        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [prompt_prefix + " " + name + "." for name in classnames]
        else:
            print("Random initialization: initializing a learnable class token")
            cls_vectors = torch.empty(n_cls, 1, ctx_dim, dtype=dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [prompt_prefix + " " + cls_token + "." for _ in classnames]

            self.cls_init_state = cls_vectors.detach().clone()
            self.cls = nn.Parameter(cls_vectors) # to be optimized

        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        if self.learned_cls:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx + 1:, :])  # ..., EOS
        else:
            self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.ctx_init = ctx_init
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = ctx_position
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.classnames = classnames

    def reset(self):
        ctx_vectors = self.ctx_init_state
        self.ctx.copy_(ctx_vectors) # to be optimized
        if self.learned_cls:
            cls_vectors = self.cls_init_state
            self.cls.copy_(cls_vectors)

    def reset_classnames(self, classnames, arch):
        self.n_cls = len(classnames)
        if not self.learned_cls:
            classnames = [name.replace("_", " ") for name in classnames]
            name_lens = [len(_tokenizer.encode(name)) for name in classnames]
            prompts = [self.prompt_prefix + " " + name + "." for name in classnames]
        else:
            cls_vectors = torch.empty(self.n_cls, 1, self.ctx_dim, dtype=self.dtype) # assume each learnable cls_token is only 1 word
            nn.init.normal_(cls_vectors, std=0.02)
            cls_token = "X"
            name_lens = [1 for _ in classnames]
            prompts = [self.prompt_prefix + " " + cls_token + "." for _ in classnames]
            # TODO: re-init the cls parameters
            # self.cls = nn.Parameter(cls_vectors) # to be optimized
            self.cls_init_state = cls_vectors.detach().clone()
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts]).to(self.device)
# =====================================================================================
        # clip, _, _ = load(arch, device=self.device, download_root=DOWNLOAD_ROOT)
        clip, _, _ = load_clip_with_tecoa(arch, device=self.device, download_root=DOWNLOAD_ROOT,robust_ckpt_path=getattr(self, "robust_ckpt_path", None))
# =====================================================================================
        with torch.no_grad():
            embedding = clip.token_embedding(tokenized_prompts).type(self.dtype)

        self.token_prefix = embedding[:, :1, :]
        self.token_suffix = embedding[:, 1 + self.n_ctx :, :]  # CLS, EOS

        self.name_lens = name_lens
        self.tokenized_prompts = tokenized_prompts
        self.classnames = classnames

    def forward(self, init=None):
        # the init will be used when computing CLIP directional loss
        if init is not None:
            ctx = init
        else:
            ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)
        elif not ctx.size()[0] == self.n_cls:
            ctx = ctx.unsqueeze(1).expand(-1, self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix
        if self.batch_size is not None: 
            # This way only works for single-gpu setting (could pass batch size as an argument for forward())
            prefix = prefix.repeat(self.batch_size, 1, 1, 1)
            suffix = suffix.repeat(self.batch_size, 1, 1, 1)

        if self.learned_cls:
            assert self.class_token_position == "end"
        if self.class_token_position == "end":
            if self.learned_cls:
                cls = self.cls
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        cls,     # (n_cls, 1, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
            else:
                prompts = torch.cat(
                    [
                        prefix,  # (n_cls, 1, dim)
                        ctx,     # (n_cls, n_ctx, dim)
                        suffix,  # (n_cls, *, dim)
                    ],
                    dim=-2,
                )
        elif self.class_token_position == "middle":
            # TODO: to work with a batch of prompts
            if self.split_idx is not None:
                half_n_ctx = self.split_idx # split the ctx at the position of [CLS] in `ctx_init`
            else:
                half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts


class ClipTestTimeTuning(nn.Module):
# ==================================================================================
# def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        # n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False):
        # super(ClipTestTimeTuning, self).__init__()
    def __init__(self, device, classnames, batch_size, criterion='cosine', arch="ViT-L/14",
                        n_ctx=16, ctx_init=None, ctx_position='end', learned_cls=False,
                        robust_ckpt_path=None):
       
# ==================================================================================       
        super(ClipTestTimeTuning, self).__init__()
        clip, _, _ = load_clip_with_tecoa(
            arch, device=device, download_root=DOWNLOAD_ROOT,
            robust_ckpt_path=robust_ckpt_path
        )
        # clip, _, _ = load(arch, device=device, download_root=DOWNLOAD_ROOT)  
        self.image_encoder = clip.visual
        self.text_encoder = TextEncoder(clip)
        self.logit_scale = clip.logit_scale.data
        # prompt tuning
        # ======================================================================
        # self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls)
        self.prompt_learner = PromptLearner(clip, classnames, batch_size, n_ctx, ctx_init, ctx_position, learned_cls,robust_ckpt_path=robust_ckpt_path)
        # =======================================================================
        self.criterion = criterion
        self.enable_image_grad = False
        
    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

    # restore the initial state of the prompt_learner (tunable prompt)
    def reset(self):
        self.prompt_learner.reset()

    def reset_classnames(self, classnames, arch):
        self.prompt_learner.reset_classnames(classnames, arch)

    def get_text_features(self):
        text_features = []
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        t_features = self.text_encoder(prompts, tokenized_prompts)
        text_features.append(t_features / t_features.norm(dim=-1, keepdim=True))
        text_features = torch.stack(text_features, dim=0)

        return torch.mean(text_features, dim=0)

    def inference(self, image):
        # with torch.no_grad():
        #     image_features = self.image_encoder(image.type(self.dtype))
        if self.enable_image_grad:
            image_features = self.image_encoder(image.type(self.dtype))
        else:
            with torch.no_grad():
                image_features = self.image_encoder(image.type(self.dtype))

        text_features = self.get_text_features()
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        #[c-tpt] --------------------------------------------
        if self.l2_norm_cal:
            prompt_mean = text_features.mean(0)
            feature_distance = text_features - prompt_mean
            l2_norm = torch.linalg.norm(feature_distance, dim=-1)
            l2_norm_mean = l2_norm.mean()
            
            #for saving to csv file
            self.l2_norm_mean = l2_norm_mean.item()
            
            #for training
            self.l2_norm_mean_training = l2_norm_mean
        
        #-----------------------------------------------------

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
# =======================================================================

# =======================================================================

    def forward(self, input):
        if isinstance(input, Tuple):
            view_0, view_1, view_2 = input
            return self.contrast_prompt_tuning(view_0, view_1, view_2)
        elif len(input.size()) == 2:
            return self.directional_prompt_tuning(input)
        else:
            return self.inference(input)

# ===========================================================
# def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False):
def get_coop(clip_arch, test_set, device, n_ctx, ctx_init, learned_cls=False, robust_ckpt_path=None):
# ===========================================================
    if test_set in fewshot_datasets:
        classnames = eval("{}_classes".format(test_set.lower()))
    elif test_set == 'bongard':
        if learned_cls:
            classnames = ['X', 'X']
        else:
            classnames = ['True', 'False']
    else:
        classnames = imagenet_classes
# ===========================================================
    # model = ClipTestTimeTuning(device, classnames, None, arch=clip_arch,
    #                         n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls)
    model = ClipTestTimeTuning(
    device, classnames, None, arch=clip_arch,
    n_ctx=n_ctx, ctx_init=ctx_init, learned_cls=learned_cls,
    robust_ckpt_path=robust_ckpt_path)
# ===========================================================
    return model

