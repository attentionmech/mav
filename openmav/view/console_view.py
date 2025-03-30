import time
import numpy as np
import torch

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from openmav.processors.data_processor import MAVGenerator


class ConsoleMAV:
    """
    Handles UI loop
    """

    def __init__(
        self,
        backend,
        refresh_rate=0.2,
        interactive=False,
        limit_chars=20,
        temperature=0,
        top_k=40,
        top_p=1,
        min_p=0,
        repetition_penalty=1,
        max_new_tokens=1,
        aggregation="l2",
        scale="linear",
        max_bar_length=20,
        num_grid_rows=1,
        selected_panels=None,
    ):
        self.backend = backend
        self.console = Console()
        self.live = Live(auto_refresh=False)
        self.refresh_rate = refresh_rate
        self.interactive = interactive
        self.limit_chars = limit_chars
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.min_p = min_p
        self.repetition_penalty = repetition_penalty
        self.max_new_tokens = max_new_tokens
        self.aggregation = aggregation
        self.scale = scale
        self.max_bar_length = max_bar_length
        self.num_grid_rows = num_grid_rows
        self.selected_panels = selected_panels

        self.generator = MAVGenerator(
            backend,
            max_new_tokens=self.max_new_tokens,
            aggregation=self.aggregation,
            scale=self.scale,
            max_bar_length=self.max_bar_length,
        )

    def ui_loop(self, prompt):
        """
        Runs the UI loop, updating the display with new data from MAVGenerator.
        """
        self.console.show_cursor(False)
        self.live.start()

        try:
            for data in self.generator.generate_tokens(
                prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
            ):
                self._render_visualization(data)

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

    def _render_visualization(self, data):
        """
        Handles UI updates based on provided data.
        """
        layout = Layout()

        selected_panels = self.selected_panels

        panel_definitions = {
            "top_predictions": Panel(
                self._create_top_predictions_panel_content(
                    data["top_ids"], data["top_probs"], data["logits"]
                ),
                title="Top Predictions",
                border_style="blue",
            ),
            "mlp_activations": Panel(
                self._create_activations_panel_content(
                    data["mlp_normalized"], data["mlp_activations"]
                ),
                title="MLP Activations",
                border_style="cyan",
            ),
            "attention_entropy": Panel(
                self._create_entropy_panel_content(
                    data["entropy_values"], data["entropy_normalized"]
                ),
                title="Attention Entropy",
                border_style="magenta",
            ),
            "output_distribution": Panel(
                self._create_prob_bin_panel(data["next_token_probs"]),
                title="Output Distribution",
                border_style="yellow",
            ),
            "generated_text": Panel(
                Text(
                    data["generated_text"][-self.limit_chars :], style="bold bright_red"
                ).append(data["predicted_char"], style="bold on green"),
                title="Generated text",
                border_style="green",
            ),
        }

        if selected_panels is None:
            selected_panels = list(panel_definitions.keys())

        panels = [
            panel_definitions[key]
            for key in selected_panels
            if key in panel_definitions
        ]
        num_rows = max(1, self.num_grid_rows)
        num_columns = (
            len(panels) + num_rows - 1
        ) // num_rows  # Best effort even distribution

        title_bar = Layout(Panel("MAV", border_style="white"), size=3)
        rows = [Layout() for _ in range(num_rows)]
        layout.split_column(title_bar, *rows)

        for i in range(num_rows):
            row_panels = panels[i * num_columns : (i + 1) * num_columns]
            if row_panels:
                rows[i].split_row(*[Layout(panel) for panel in row_panels])

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
            f"([bold yellow]{prob:>5.1%}[/bold yellow], [bold cyan]{logit:>4.1f}[/bold cyan])\n"
            for token_id, prob, logit in zip(
                top_ids.tolist(), top_probs.tolist(), logits[0, -1, top_ids].tolist()
            )
        ]

        return "\n".join(entry for entry in entries)

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
