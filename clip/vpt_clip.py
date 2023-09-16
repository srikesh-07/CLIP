from typing import Union, Tuple
import math
from functools import reduce
from operator import mul

import torch
from torch import nn

from .model import CLIP, Transformer, VisionTransformer

NUM_TOKENS = 20
INITIATION = "random"
LOCATION = "prepend"
DEEP = True
DROPOUT = 0.2


class PromptedTransformer(Transformer):
    def __init__(self,
                 num_tokens: int,
                 weight_init: str,
                 location: str,
                 deep: bool,
                 dropout: float,
                 patch_size: int,
                 output_dim: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        # self.weight_init = weight_init
        self.location = location
        self.deep = deep
        self.prompt_dropout = torch.nn.Dropout(dropout)

        if weight_init == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, (patch_size, patch_size), 1) + output_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, output_dim, dtype=torch.float16))

            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.deep:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    kwargs["layers"] - 1,
                    num_tokens, output_dim, dtype=torch.float16
                ))
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def train(self, mode: bool = True):
        super().train(mode)
        self.training = mode

        self.resblocks.eval()
        self.resblocks.requires_grad_(False)

        self.prompt_dropout.requires_grad_(mode)
        self.prompt_embeddings.requires_grad_(mode)
        if hasattr(self, "deep_prompt_embeddings"):
          self.deep_prompt_embeddings.requires_grad_(mode)

        if mode:
            self.prompt_dropout.train()
        else:
            self.resblocks.eval()
            self.prompt_dropout.eval()

        return self

    def incorporate_prompt(self, x: torch.Tensor):
        B = x.shape[0]
        if self.location == "prepend":
            x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.prompt_embeddings.expand(B, -1, -1)),
                    x[:, 1:, :]
                ), dim=1)
        else:
            raise ValueError("Other prompt locations are not supported")
        return x

    def forward(self, x: torch.Tensor): #(patch_size x B x width)
        # print(f"Input Shape: {x.shape}")
        x = x.permute([1, 0, 2]) # (B x patch_size_embed x width)
        # print(f"Before Prompt Shape: {x.shape}")
        x = self.incorporate_prompt(x)
        # print(f"After Prompt Shape: {x.shape}")
        x = x.permute([1, 0, 2]) #(patch_size x B x width)
        # print(f"After Resizing Prompt Shape: {x.shape}")

        if self.deep:
            # print(f"Input Block Shape: {x.shape}")
            with torch.no_grad():
              x = self.resblocks[0](x)
            # print(f"Output Block Shape: {x.shape}")
            B = x.shape[1]
            # print(f"Batch Size {B}")
            for block_id in range(1, self.layers):
                x = x.permute([1, 0, 2])
                # print(f"Before Deep Prompt Shape: {x.shape}")
                x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.deep_prompt_embeddings[block_id - 1].expand(B, -1, -1)
                    ),
                    x[:, (1 + self.num_tokens):, :]
                ), dim=1)
                # print(f"After Deep Prompt Shape: {x.shape}")
                x = x.permute([1, 0, 2])
                # print(f"Input Block Shape: {x.shape}")
                with torch.no_grad():
                  x = self.resblocks[block_id](x)
                # print(f"Output Block Shape: {x.shape}")
        else:
            with torch.no_grad():
              x = self.resblocks(x)
        # print(f"Return Shape: {x.shape}")
        return x


class PromptedVisionTransformer(VisionTransformer):
    def __init__(self,
                 input_resolution: int,
                 patch_size: int,
                 width: int,
                 layers: int,
                 heads: int,
                 output_dim: int,
                 # vpt
                 num_tokens: int,
                 weight_init: str,
                 location: str,
                 deep: bool,
                 dropout: float):
        super().__init__(input_resolution, patch_size, width, layers, heads, output_dim)
        self.transformer = PromptedTransformer(num_tokens=num_tokens,
                                               weight_init=weight_init,
                                               location=location,
                                               deep=deep,
                                               dropout=dropout,
                                               width=width,
                                               layers=layers,
                                               heads=heads,
                                               patch_size=patch_size,
                                               output_dim=width)

    def train(self, mode: bool = True):
        super().train(False)
        self.requires_grad_(False)
        if mode:
            self.training = mode
            self.transformer.requires_grad_(mode)
            self.transformer.train(mode)
        return self


class PromptedVisualCLIP(CLIP):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # vpt
                 num_tokens: int = NUM_TOKENS,
                 weight_init: str = INITIATION,
                 location: str = LOCATION,
                 deep: bool = DEEP,
                 dropout: float = DROPOUT,
                 ):
        super().__init__(embed_dim=embed_dim,
                         image_resolution=image_resolution,
                         vision_layers=vision_layers,
                         vision_width=vision_width,
                         vision_patch_size=vision_patch_size,
                         context_length=context_length,
                         vocab_size=vocab_size,
                         transformer_width=transformer_width,
                         transformer_heads=transformer_heads,
                         transformer_layers=transformer_layers
                         )
        vision_heads = vision_width // 64
        print("Loaded VPT CLIP..")
        self.visual = PromptedVisionTransformer(input_resolution=image_resolution,
                                                patch_size=vision_patch_size,
                                                width=vision_width,
                                                layers=vision_layers,
                                                heads=vision_heads,
                                                output_dim=embed_dim,
                                                num_tokens=num_tokens,
                                                weight_init=weight_init,
                                                location=location,
                                                deep=deep,
                                                dropout=dropout
                                                )

    def train(self, mode: bool = True):
        super().train(False)
        self.requires_grad_(False)
        if mode:
            print("Loading CLIP in Training Mode")
            self.training = mode
            self.visual.requires_grad_(mode)
            self.visual.train(mode)
        print("\n\nTotal Parameters of Model: ", sum(p.numel() for p in model.parameters()))
        print("Trainable Parameters of Model: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
        print("\n\nTrainable Parameters: ")
        count = 1
        for name, p in model.named_parameters():
            if p.requires_grad:
                print(f"{count} - {name}")
                count += 1
        return self
            

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def custom_build_model(state_dict: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = PromptedVisualCLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    out = model.load_state_dict(state_dict, strict=False)
    for key in out[0]:
      print(f"[WARNING] Missing State Dict for Layer - {key}")
    for key in out[1]:
      print(f"[WARNING] Unexpected State Dict for Layer - {key}")
    return model.eval()
