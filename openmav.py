# /// script
# dependencies = [
#   "rich",
#   "torch",
#   "transformers",
#   "numpy",
# ]
# ///

# @author: attentionmech

import argparse
import time

import numpy as np
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

def compute_entropy(attn_matrix):
    """Compute entropy of attention distributions per layer."""
    entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + 1e-9), dim=-1)
    return entropy.mean(dim=-1).cpu().numpy()


class DataConverter:
    @staticmethod
    def process_mlp_activations(hidden_states, aggregation="l2"):
        """
        Process MLP activations based on specified aggregation method.

        Args:
            hidden_states (torch.Tensor): Hidden states from the model
            aggregation (str): Aggregation method to use

        Returns:
            numpy.ndarray: Processed MLP activations
        """
        activations = torch.stack([layer[:, -1, :] for layer in hidden_states])

        if aggregation == "mean":
            return activations.mean(dim=-1).numpy()
        elif aggregation == "l2":
            return torch.norm(activations, p=2, dim=-1).numpy()
        elif aggregation == "max_abs":
            return activations.abs().max(dim=-1).values.numpy()
        else:
            raise ValueError(
                "Invalid aggregation method. Choose from: mean, l2, max_abs."
            )

    @staticmethod
    def process_entropy(attentions):
        """
        Compute entropy values for attention matrices.

        Args:
            attentions (list): Attention matrices from the model

        Returns:
            numpy.ndarray: Entropy values for each layer
        """
        return np.array([compute_entropy(attn[:, :, -1, :]) for attn in attentions])

    @staticmethod
    def normalize_activations(activations, max_bar_length=20):
        """
        Normalize activations for visualization.

        Args:
            activations (numpy.ndarray): Raw activation values
            max_bar_length (int): Maximum length of visualization bar

        Returns:
            numpy.ndarray: Normalized activations
        """
        max_abs_act = np.max(np.abs(activations))
        return (activations / max_abs_act) * max_bar_length

    @staticmethod
    def normalize_entropy(entropy_values, max_bar_length=20):
        """
        Normalize entropy values for visualization.

        Args:
            entropy_values (numpy.ndarray): Raw entropy values
            max_bar_length (int): Maximum length of visualization bar

        Returns:
            numpy.ndarray: Normalized entropy values
        """
        max_entropy = np.max(entropy_values)
        max_entropy = max(max_entropy, 1e-9)
        return (entropy_values / max_entropy) * max_bar_length


class ModelActivationVisualizer:
    def __init__(
        self,
        backend,
        max_new_tokens=100,
        max_bar_length=20,
        aggregation="l2",
        refresh_rate=0.2,
        interactive=False,
    ):
        self.backend = backend
        self.console = Console()
        self.live = Live(auto_refresh=False)
        self.aggregation = aggregation
        self.refresh_rate = refresh_rate
        self.max_new_tokens = max_new_tokens
        self.max_bar_length = max_bar_length
        self.interactive = interactive
        self.data_converter = DataConverter()

    def generate_with_visualization(self, prompt):
        inputs = self.backend.tokenize(prompt)
        generated_ids = inputs.tolist()[0]

        self.console.show_cursor(False)
        self.live.start()

        try:
            for _ in range(self.max_new_tokens):
                outputs = self.backend.generate(generated_ids)
                logits = outputs["logits"]
                hidden_states = outputs["hidden_states"]
                attentions = outputs["attentions"]

                next_token_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
                top_probs, top_ids = torch.topk(next_token_probs, 8)
                next_token_id = top_ids[0].item()
                generated_ids.append(next_token_id)

                # Use DataConverter to process data
                mlp_activations = self.data_converter.process_mlp_activations(
                    hidden_states, self.aggregation
                )
                entropy_values = self.data_converter.process_entropy(attentions)

                self._render_visualization(
                    generated_ids,
                    next_token_id,
                    mlp_activations,
                    top_ids,
                    top_probs,
                    logits,
                    entropy_values,
                )

                if self.interactive:
                    user_input = self.console.input("")
                    if user_input.lower() == "q":
                        break
                else:
                    time.sleep(self.refresh_rate)

        finally:
            self.live.stop()
            self.console.show_cursor(True)

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
        # Normalize activations and entropy using DataConverter
        mlp_normalized = self.data_converter.normalize_activations(
            mlp_activations, self.max_bar_length
        )
        entropy_normalized = self.data_converter.normalize_entropy(
            entropy_values, self.max_bar_length
        )

        generated_text = self.backend.decode(
            generated_ids[:-1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        predicted_char = self.backend.decode(
            [next_token_id], clean_up_tokenization_spaces=True
        )

        layout = Layout()

        # Render MLP activations panel
        activations_str = self._create_activations_panel_content(
            mlp_normalized, mlp_activations
        )
        activations_panel = Panel(
            activations_str, title="MLP Activations", border_style="cyan"
        )

        # Render entropy panel
        entropy_str = self._create_entropy_panel_content(
            entropy_values, entropy_normalized
        )
        entropy_panel = Panel(
            entropy_str, title="Attention Entropy", border_style="magenta"
        )

        # Render top predictions panel
        top_preds_str = self._create_top_predictions_panel_content(
            top_ids, top_probs, logits
        )
        predictions_panel = Panel(
            f"{top_preds_str}",
            title="Top Predictions",
            border_style="blue",
        )

        # Render generated text panel
        highlighted_text = Text(generated_text, style="bold bright_red")
        highlighted_text.append(predicted_char, style="bold on green")
        top_panel = Panel(
            highlighted_text,
            title=f"MAV: {self.backend.model_name}",
            border_style="green",
        )

        # Layout composition
        layout.split_column(
            Layout(predictions_panel, size=5),
            Layout(name="bottom_panel"),
        )

        layout["bottom_panel"].split_row(
            Layout(top_panel, ratio=1),
            Layout(activations_panel, ratio=2),
            Layout(entropy_panel, ratio=2),
        )

        self.live.update(layout, refresh=True)

    def _create_activations_panel_content(self, mlp_normalized, mlp_activations):
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
        return activations_str

    def _create_entropy_panel_content(self, entropy_values, entropy_normalized):
        entropy_str = ""
        for i, (entropy_val, entropy_norm) in enumerate(
            zip(entropy_values, entropy_normalized)
        ):
            entropy_val = float(entropy_val)
            entropy_norm = int(abs(float(entropy_norm)))
            entropy_bar = "█" * entropy_norm
            entropy_str += f"[bold white]Layer {i + 1:2d}[/] | [bold yellow]:[/] [{entropy_bar.ljust(self.max_bar_length)}] {entropy_val:.4f}\n"
        return entropy_str

    def _create_top_predictions_panel_content(self, top_ids, top_probs, logits):
        return "    ".join(
            f"[bold magenta]{self.backend.decode([token_id], clean_up_tokenization_spaces=True)}[/] "
            f"([bold yellow]{prob:.1%}[/bold yellow], [bold cyan]{logit:.2f}[/bold cyan])"
            for token_id, prob, logit in zip(
                top_ids.tolist(), top_probs.tolist(), logits[0, -1, top_ids].tolist()
            )
        )


class ModelBackend:
    def __init__(self, model_name):
        self.model_name = model_name

    def initialize(self):
        raise NotImplementedError("Subclasses must implement initialize()")

    def generate(self, input_ids):
        raise NotImplementedError("Subclasses must implement generate()")

    def tokenize(self, text):
        raise NotImplementedError("Subclasses must implement tokenize()")

    def decode(self, token_ids, **kwargs):
        raise NotImplementedError("Subclasses must implement decode()")


class TransformersBackend(ModelBackend):
    def __init__(self, model_name):
        super().__init__(model_name)
        self.initialize()

    def initialize(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate(self, input_ids):
        with torch.no_grad():
            outputs = self.model(torch.tensor([input_ids]))
        return {
            "logits": outputs.logits,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
        }

    def tokenize(self, text):
        return self.tokenizer(text, return_tensors="pt")["input_ids"]

    def decode(self, token_ids, **kwargs):
        return self.tokenizer.decode(token_ids, **kwargs)


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

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode (press Enter to continue)",
        default=False,
    )

    args = parser.parse_args()

    if args.prompt is None or len(args.prompt) == 0:
        print("Prompt cannot be empty.")
        return

    backend = TransformersBackend(args.model)
    visualizer = ModelActivationVisualizer(
        backend=backend,
        max_new_tokens=args.tokens,
        aggregation=args.aggregation,
        refresh_rate=args.refresh_rate,
        interactive=args.interactive,
    )
    visualizer.generate_with_visualization(args.prompt)


if __name__ == "__main__":
    main()
