from typing import Union, Tuple

import torch
from torch import nn

from model import CLIP, Transformer, VisionTransformer

NUM_TOKENS = 0
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
                 **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        # self.weight_init = weight_init
        self.location = location
        self.deep = deep

        self.prompt_dropout = torch.nn.Dropout(dropout)

        if weight_init == "random":
            val = math.sqrt(6. / float(3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim))  # noqa

            self.prompt_embeddings = nn.Parameter(torch.zeros(
                1, num_tokens, self.embed_dim))

            nn.init.uniform_(self.prompt_embeddings.data, -val, val)
            if self.deep:
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    len(self.blocks) - 1,
                    num_tokens, self.embed_dim
                ))
                nn.init.uniform_(
                    self.deep_prompt_embeddings.data, -val, val)

        else:
            raise ValueError("Other initiation scheme is not supported")

    def train(self, mode: bool = True):
        if mode:
            self.prompt_dropout.train()
            self.resblocks.eval()
            self.norm.train()
        else:
            for module in self.children():
                module.train(mode)

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

    def forward(self, x: torch.Tensor):
        x = self.incorporate_prompt(x)
        if DEEP:
            x = self.resblocks[0](x)
            B = x.shape[0]
            for block_id in range(1, self.layers):
                x = torch.cat((
                    x[:, :1, :],
                    self.prompt_dropout(
                        self.deep_prompt_embeddings[block_id - 1].expand(B, -1, -1)
                    ),
                    x[:, (1 + self.num_tokens):, :]
                ), dim=1)
                x = self.resblocks[block_id](x)
        else:
            x = self.resblocks(x)

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
                                               heads=heads)


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
