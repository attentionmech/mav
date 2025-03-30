import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from openmav.backends.model_backend import ModelBackend


class TransformersBackend(ModelBackend):
    def __init__(
        self, model_name, model_obj=None, tokenizer_obj=None, device="cpu", seed=42
    ):
        self.model_name = model_name
        self.device = device
        self.model_obj = model_obj
        self.tokenizer_obj = tokenizer_obj

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.initialize()

    def initialize(self):

        try:
            if self.model_obj:
                self.model = self.model_obj.to(self.device)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_attentions=True,
                    attn_implementation="eager",
                ).to(self.device)

            if self.tokenizer_obj:
                self.tokenizer = self.tokenizer_obj
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = (
                    self.tokenizer.eos_token or self.tokenizer.unk_token
                )

            if self.model.config.pad_token_id is None:
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(
        self,
        input_ids,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
    ):

        input_tensor = torch.tensor([input_ids]).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                input_tensor,
                do_sample=temperature > 0,
                max_new_tokens=1,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                return_dict_in_generate=True,
                output_scores=True,
                output_hidden_states=True,
                output_attentions=True,
                pad_token_id=self.tokenizer.eos_token_id,  # warning TODO: find correct way to handle this
            )

        return {
            "logits": outputs.scores[-1].unsqueeze(0).cpu(),  # Last step logits
            "hidden_states": outputs.hidden_states[-1],  # Last step hidden states
            "attentions": outputs.attentions[-1],  # Last step attentions
        }

    def tokenize(self, text):
        return self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")[
            "input_ids"
        ].to(self.device)

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)
