# /// script
# dependencies = [
#   "datasets",
#   "torch",
#   "transformers[torch]",
#   "openmav@git+https://github.com/attentionmech/mav",
# ]
# ///


# run this using: uv run examples/vis_train_loop.py

import os
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback
from datasets import load_dataset
from openmav.mav import MAV

DEVICE="mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_embd=256,
    n_layer=2,
    n_head=2,
    attn_implementation="eager"
)

model = GPT2LMHeadModel(config)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir="/tmp/temp_output",
    # output_dir="nul",  # Windows (uncomment if using Windows)
    overwrite_output_dir=False,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_strategy="no",  # Disables checkpointing
    logging_dir=None,  # Prevents logging to disk
    report_to="none",  # No logging/reporting
)


class InferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, model, eval_dataset, eval_steps=100):
        self.tokenizer = tokenizer
        self.model = model
        self.eval_dataset = eval_dataset
        self.eval_steps = eval_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.eval_steps == 0 and state.global_step > 0:
            self.perform_inference(state.global_step)

    def perform_inference(self, step):
        self.model.eval()
        with torch.no_grad():
            MAV("gpt2", "Once upon a time", model_obj=self.model, tokenizer_obj=self.tokenizer, max_new_tokens=20, refresh_rate=0.1, device=DEVICE)            
        self.model.train()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    callbacks=[InferenceCallback(tokenizer, model, tokenized_datasets)],
)

trainer.train()

# tokenizer.save_pretrained("./gpt2-small-tinystories")
# model.save_pretrained("./gpt2-small-tinystories")
