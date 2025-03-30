import time
import numpy as np
import torch

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from openmav.converters.data_converter import DataConverter

class ConsoleMAV:
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

