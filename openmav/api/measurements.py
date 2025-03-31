from dataclasses import dataclass
from typing import List

import torch

# this has to be reasonably stable
# idea is that people can pass their own plugins which assume existence of this
# choosing torch.tensor is ok now since this project 80% works around hf transformers


@dataclass
class ModelMeasurements:
    mlp_activations: torch.Tensor
    mlp_normalized: torch.Tensor
    attention_entropy_values: torch.Tensor
    attention_entropy_values_normalized: torch.Tensor
    generated_text: str
    predicted_char: str
    next_token_probs: torch.Tensor
    top_ids: torch.Tensor
    top_probs: torch.Tensor
    logits: torch.Tensor
    decoded_tokens: List[str]
