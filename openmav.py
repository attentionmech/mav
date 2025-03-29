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
import warnings

import numpy as np
import torch
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer


warnings.filterwarnings("ignore")

class DataConverter:

    @staticmethod
    def compute_entropy(attn_matrix):
        """Compute entropy of attention distributions per layer."""
        entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + 1e-9), dim=-1)
        return entropy.mean(dim=-1).cpu().numpy()

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
          
        if aggregation == "l2":
            return torch.norm(activations, p=2, dim=-1).cpu().numpy()
        elif aggregation == "max_abs":
            return activations.abs().max(dim=-1).values.cpu().numpy()
        else:
            raise ValueError(
                "Invalid aggregation method. Choose from: l2, max_abs."
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
        return np.array(
            [DataConverter.compute_entropy(attn[:, :, -1, :].cpu()) for attn in attentions]
        )

    @staticmethod
    def normalize_activations(activations, scale_type="linear", max_bar_length=20):
        """
        Normalize activations for visualization.

        Args:
            activations (numpy.ndarray): Raw activation values
            max_bar_length (int): Maximum length of visualization bar

        Returns:
            numpy.ndarray: Normalized activations
        """
        return DataConverter.apply_scaling(activations, scale_type, max_bar_length)

    @staticmethod
    def normalize_entropy(entropy_values, scale_type="linear", max_bar_length=20):
        """
        Normalize entropy values for visualization.

        Args:
            entropy_values (numpy.ndarray): Raw entropy values
            max_bar_length (int): Maximum length of visualization bar

        Returns:
            numpy.ndarray: Normalized entropy values
        """
        return DataConverter.apply_scaling(entropy_values, scale_type, max_bar_length)

    @staticmethod
    def apply_scaling(values, scale_type="linear", max_bar_length=20):
        """
        Apply scaling transformation to values for better visualization.

        Args:
            values (numpy.ndarray): Input values to be scaled.
            scale_type (str): Scaling method - 'linear', 'log', or 'minmax'.
            max_bar_length (int): Maximum length for visualization bars.

        Returns:
            numpy.ndarray: Scaled values.
        """
        values = np.array(values)  # Ensure input is a NumPy array

        if scale_type == "log":
            values = np.log1p(np.abs(values))  # log(1 + x) to handle zero values safely
        elif scale_type == "minmax":
            min_val, max_val = np.min(values), np.max(values)
            if max_val - min_val > 1e-9:  # Prevent division by zero
                values = (values - min_val) / (max_val - min_val)
            else:
                values = np.zeros_like(
                    values
                )  # If all values are the same, return zeros

        if np.max(values) > 0:
            return (values / np.max(values)) * max_bar_length  # Scale to max_bar_length
        return values


class ModelActivationVisualizer:
    def __init__(
        self,
        backend,
        max_new_tokens=100,
        max_bar_length=20,
        aggregation="l2",
        refresh_rate=0.2,
        interactive=False,
        scale="linear",
        limit_chars=20,
    ):
        self.backend = backend
        self.console = Console()
        self.live = Live(auto_refresh=False)
        self.aggregation = aggregation
        self.refresh_rate = refresh_rate
        self.max_new_tokens = max_new_tokens
        self.max_bar_length = max_bar_length
        self.interactive = interactive
        self.scale = scale
        self.limit_chars = limit_chars

        self.data_converter = DataConverter()

    def generate_with_visualization(
        self,
        prompt,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
    ):
        inputs = self.backend.tokenize(prompt)
        generated_ids = inputs.tolist()[0]

        self.console.show_cursor(False)
        self.live.start()

        try:
            for _ in range(self.max_new_tokens):
                outputs = self.backend.generate(
                    generated_ids,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    min_p=min_p,
                    repetition_penalty=repetition_penalty,
                )
                logits = outputs["logits"]
                hidden_states = outputs["hidden_states"]
                attentions = outputs["attentions"]

                next_token_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
                top_probs, top_ids = torch.topk(next_token_probs, 20)
                next_token_id = torch.multinomial(
                    next_token_probs, num_samples=1
                ).item()
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
                    next_token_probs,
                )

                if self.interactive:
                    user_input = self.console.input("")
                    if user_input.lower() == "q":
                        break
                else:
                    if self.refresh_rate > 0:
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
        next_token_probs,
    ):
        mlp_normalized = self.data_converter.normalize_activations(
            mlp_activations, scale_type=self.scale, max_bar_length=self.max_bar_length
        )
        entropy_normalized = self.data_converter.normalize_entropy(
            entropy_values, scale_type=self.scale, max_bar_length=self.max_bar_length
        )

        generated_text = self.backend.decode(
            generated_ids[:-1],
            skip_special_tokens=True,  # TODO explicity set these kwargs in interface
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

        prob_bin_panel = Panel(
            self._create_prob_bin_panel(next_token_probs),
            title="Output Distribution",
            border_style="yellow",
        )

        generated_text = generated_text[-self.limit_chars :]

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
            Layout(top_panel, ratio=2),
            Layout(activations_panel, ratio=3),
            Layout(entropy_panel, ratio=3),
            Layout(prob_bin_panel, ratio=2),
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
                f"[bold yellow]:[/] [{mlp_color}]{mlp_bar.ljust(self.max_bar_length)}[/] [bold yellow]{raw_mlp_scalar:+.1f}[/]\n"
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
            entropy_str += f"[bold white]Layer {i + 1:2d}[/] | [bold yellow]:[/] [{entropy_bar.ljust(self.max_bar_length)}] {entropy_val:.1f}\n"
        return entropy_str

    def _create_top_predictions_panel_content(self, top_ids, top_probs, logits):
        # Create list of formatted entries
        entries = [
            f"[bold magenta]{self.backend.decode([token_id], clean_up_tokenization_spaces=True).strip()[:10] or ' ':<10}[/] "
            f"([bold yellow]{prob:>5.1%}[/bold yellow], [bold cyan]{logit:>4.1f}[/bold cyan])"
            for token_id, prob, logit in zip(
                top_ids.tolist(), top_probs.tolist(), logits[0, -1, top_ids].tolist()
            )
        ]

        # Chunk into groups of 5 and join with newlines
        chunked = [entries[i : i + 5] for i in range(0, len(entries), 5)]
        return "\n".join("    ".join(chunk) for chunk in chunked)

    def _create_prob_bin_panel(self, next_token_probs, num_bins=20):
        """
        Create a histogram-like panel for token probabilities after sorting.

        Args:
            next_token_probs (torch.Tensor): Probabilities of the next token.
            num_bins (int): Number of bins to divide the sorted probabilities.

        Returns:
            str: Formatted string representing probability bins.
        """
        next_token_probs = next_token_probs.cpu().numpy()

        sorted_probs = np.sort(next_token_probs)

        sorted_probs = sorted_probs[
            -100:
        ]  # TODO: find better way to organise params for this project

        bin_edges = np.linspace(0, len(sorted_probs), num_bins + 1, dtype=int)
        bin_sums = [
            np.sum(sorted_probs[bin_edges[i] : bin_edges[i + 1]])
            for i in range(num_bins)
        ]

        max_sum = max(bin_sums) if max(bin_sums) > 0 else 1
        bar_chars = ["█" * int((s / max_sum) * self.max_bar_length) for s in bin_sums]

        bin_labels = [
            f"{sorted_probs[bin_edges[i + 1] - 1]:.4f}" for i in range(num_bins)
        ]

        bin_output = "\n".join(
            f"[bold yellow]{label}[/]: [bold cyan]{bar}[/]"
            for label, bar, s in zip(bin_labels, bar_chars, bin_sums)
        )

        return bin_output


class ModelBackend:
    def __init__(self, model_name, device="cpu"):
        pass

    def initialize(self):
        raise NotImplementedError("Subclasses must implement initialize()")

    def generate(
        self,
        input_ids,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
    ):
        raise NotImplementedError("Subclasses must implement generate()")

    def tokenize(self, text):
        raise NotImplementedError("Subclasses must implement tokenize()")

    def decode(self, token_ids, **kwargs):
        raise NotImplementedError("Subclasses must implement decode()")


class TransformersBackend(ModelBackend):
    def __init__(self, model_name, device="cpu"):
        self.model_name = model_name
        self.device = device
        self.initialize()

    def initialize(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_attentions=True,
                attn_implementation="eager",
            ).to(self.device)

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
        "--max-new-tokens", type=int, default=200, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["l2", "max_abs"],
        default="l2",
        help="Aggregation method (l2, max_abs)",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=0.1,
        help="Refresh rate for visualization",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode (press Enter to continue)",
        default=False,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run the model on (cpu, cuda, mps).",
    )

    parser.add_argument(
        "--scale",
        type=str,
        choices=["linear", "log", "minmax"],
        default="linear",
        help="Scaling method for visualization (linear, log, minmax).",
    )

    parser.add_argument(
        "--limit-chars",
        type=int,
        default=250,
        help="Limit the number of tokens for visualization.",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (higher values = more randomness, default: 0.0)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="top-k sampling (set to 0 to disable, default: 50)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top-p (nucleus) filtering (set to 1.0 to disable, default: 1.0)",
    )

    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="min_p sampling (default: 0.0)",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty for repeated words (default: 1.0, higher values discourage repetition)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers"],
        help="Backend to use for model provider (currently on transformers)",
    )

    # random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    torch.manual_seed(parser.parse_args().seed)
    np.random.seed(parser.parse_args().seed)

    args = parser.parse_args()

    if args.prompt is None or len(args.prompt) == 0:
        print("Prompt cannot be empty.")
        return

    backend = None

    if args.backend == "transformers":
        backend = TransformersBackend(args.model, args.device)
    else:
        raise ValueError(f"Unsupported backend: {args.backend}")

    visualizer = ModelActivationVisualizer(
        backend=backend,
        max_new_tokens=args.max_new_tokens,
        aggregation=args.aggregation,
        refresh_rate=args.refresh_rate,
        interactive=args.interactive,
        scale=args.scale,
        limit_chars=args.limit_chars,
    )

    visualizer.generate_with_visualization(
        args.prompt,
        temperature=args.temp,
        top_k=args.top_k,
        top_p=args.top_p,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
    )


if __name__ == "__main__":
    main()
