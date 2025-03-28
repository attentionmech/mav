# /// script
# dependencies = [
#   "rich",
#   "torch",
#   "transformers",
# ]
# ///

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
from rich.console import Console
from rich.text import Text
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
import argparse


def compute_entropy(attn_matrix):
    """Compute entropy of attention distributions per layer."""
    entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + 1e-9), dim=-1)
    return entropy.mean(dim=-1).cpu().numpy()


class ModelActivationVisualizer:
    def __init__(
        self,
        model_name,
        max_new_tokens=100,
        max_bar_length=20,
        aggregation="mean",
        refresh_rate=0.2,
    ):
        """
        Initialize the visualizer with a specified Hugging Face model.
        Model Activation Visualizer (MAV)
        """
        self.console = Console()
        self.live = Live(auto_refresh=False)
        self.aggregation = aggregation
        self.model_name = model_name
        self.refresh_rate = refresh_rate

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

        self.max_new_tokens = max_new_tokens
        self.max_bar_length = max_bar_length

    def generate_with_visualization(self, prompt):
        """
        Generate text and visualize model activations and attention entropy dynamically.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        generated_ids = inputs["input_ids"].tolist()[0]

        self.console.show_cursor(False)
        self.live.start()

        try:
            for _ in range(self.max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(torch.tensor([generated_ids]))
                    logits = outputs.logits
                    hidden_states = outputs.hidden_states
                    attentions = outputs.attentions

                next_token_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
                top_probs, top_ids = torch.topk(next_token_probs, 8)
                next_token_id = top_ids[0].item()
                generated_ids.append(next_token_id)

                mlp_activations = self._process_mlp_activations(hidden_states)
                entropy_values = np.array(
                    [compute_entropy(attn[:, :, -1, :]) for attn in attentions]
                )

                self._render_visualization(
                    generated_ids,
                    next_token_id,
                    mlp_activations,
                    top_ids,
                    top_probs,
                    logits,
                    entropy_values,
                )

                time.sleep(self.refresh_rate)

        finally:
            self.live.stop()
            self.console.show_cursor(True)

    def _process_mlp_activations(self, hidden_states):
        """
        Process MLP (Feedforward) layer activations based on the selected aggregation method.
        """
        activations = torch.stack([layer[:, -1, :] for layer in hidden_states])

        if self.aggregation == "mean":
            return activations.mean(dim=-1).numpy()
        elif self.aggregation == "l2":
            return torch.norm(activations, p=2, dim=-1).numpy()
        elif self.aggregation == "max_abs":
            return activations.abs().max(dim=-1).values.numpy()
        else:
            raise ValueError(
                "Invalid aggregation method. Choose from: mean, l2, max_abs."
            )

    def _render_visualization(
        self,
        generated_ids,
        next_token_id,
        mlp_activations,
        top_ids,
        top_probs,
        logits,
        entropy_values,
    ):
        """
        Render the activation and entropy visualization using Rich library.
        """
        max_abs_act = np.max(np.abs(mlp_activations))
        mlp_normalized = (mlp_activations / max_abs_act) * self.max_bar_length

        max_entropy = np.max(entropy_values)
        max_entropy = max(max_entropy, 1e-9)

        entropy_normalized = (entropy_values / max_entropy) * self.max_bar_length

        generated_text = self.tokenizer.decode(
            generated_ids[:-1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        predicted_char = self.tokenizer.decode(
            [next_token_id], clean_up_tokenization_spaces=True
        )

        layout = Layout()

        activations_str = ""
        for i, (mlp_act, raw_mlp) in enumerate(zip(mlp_normalized, mlp_activations)):
            mlp_act_scalar = (
                mlp_act.item() if isinstance(mlp_act, np.ndarray) else mlp_act
            )
            raw_mlp_scalar = (
                raw_mlp.item() if isinstance(raw_mlp, np.ndarray) else raw_mlp
            )
            mlp_bar = "█" * int(abs(mlp_act_scalar))
            mlp_color = "yellow" if raw_mlp_scalar >= 0 else "magenta"

            activations_str += (
                f"[bold white]Layer {i:2d}[/] | "
                f"[bold yellow]:[/] [{mlp_color}]{mlp_bar.ljust(self.max_bar_length)}[/] [bold yellow]{raw_mlp_scalar:+.4f}[/]\n"
            )

        activations_panel = Panel(
            activations_str, title="MLP Activations", border_style="cyan"
        )

        entropy_str = ""
        for i, (entropy_val, entropy_norm) in enumerate(
            zip(entropy_values, entropy_normalized)
        ):
            entropy_val = float(entropy_val)
            entropy_norm = int(abs(float(entropy_norm)))
            entropy_bar = "█" * entropy_norm
            entropy_str += f"[bold white]Layer {i + 1:2d}[/] | [bold yellow]:[/] [{entropy_bar.ljust(self.max_bar_length)}] {entropy_val:.4f}\n"

        entropy_panel = Panel(
            entropy_str, title="Attention Entropy", border_style="magenta"
        )

        top_preds_str = "    ".join(
            f"[bold magenta]{self.tokenizer.decode([token_id], clean_up_tokenization_spaces=True)}[/] "
            f"([bold yellow]{prob:.1%}[/bold yellow], [bold cyan]{logit:.2f}[/bold cyan])"
            for token_id, prob, logit in zip(
                top_ids.tolist(), top_probs.tolist(), logits[0, -1, top_ids].tolist()
            )
        )

        predictions_panel = Panel(
            f"{top_preds_str}",
            title="Top Predictions",
            border_style="blue",
        )

        highlighted_text = Text(generated_text, style="bold bright_red")
        highlighted_text.append(predicted_char, style="bold on green")

        top_panel = Panel(
            highlighted_text, title=f"MAV: {self.model_name}", border_style="green"
        )

        layout.split_column(
            Layout(predictions_panel, size=5),
            Layout(name="bottom_panel"),  # Placeholder for the bottom row
        )

        layout["bottom_panel"].split_row(
            Layout(top_panel, ratio=1),
            Layout(activations_panel, ratio=2),
            Layout(entropy_panel, ratio=2),
        )

        self.live.update(layout, refresh=True)


def main():
    parser = argparse.ArgumentParser(
        description="Model Activation and Entropy Visualizer"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model name (default: gpt2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a timeline ",
        help="Initial prompt for text generation",
    )
    parser.add_argument(
        "--tokens", type=int, default=100, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["mean", "l2", "max_abs"],
        default="l2",
        help="Aggregation method (mean, l2, max_abs)",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=0.2,
        help="Refresh rate for visualization",
    )

    args = parser.parse_args()

    # avoiding empty prompt for now just for sanity
    if args.prompt is None or len(args.prompt) == 0:
        print("Prompt cannot be empty.")
        return

    visualizer = ModelActivationVisualizer(
        model_name=args.model, max_new_tokens=args.tokens, aggregation=args.aggregation
    )
    visualizer.generate_with_visualization(args.prompt)


if __name__ == "__main__":
    main()
