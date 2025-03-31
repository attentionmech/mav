import numpy as np
import torch
from rich.text import Text

from openmav.api.measurements import ModelMeasurements
from openmav.view.panels.panel_base import PanelBase


class TopPredictionsPanel(PanelBase):
    def __init__(
        self, measurements: ModelMeasurements, max_bar_length=20, limit_chars=50
    ):
        super().__init__("Top Predictions", "blue", max_bar_length, limit_chars)
        self.measurements = measurements

    def get_panel_content(self):
        entries = [
            f"[bold magenta]{token:<10}[/] "
            f"([bold yellow]{prob:>5.1%}[/bold yellow], [bold cyan]{logit:>4.1f}[/bold cyan])"
            for token, prob, logit in zip(
                self.measurements.decoded_tokens,
                self.measurements.top_probs.tolist(),
                self.measurements.logits[0, -1, self.measurements.top_ids].tolist(),
            )
        ]
        return "\n".join(entries)


class MlpActivationsPanel(PanelBase):
    def __init__(
        self, measurements: ModelMeasurements, max_bar_length=20, limit_chars=50
    ):
        super().__init__("MLP Activations", "cyan", max_bar_length, limit_chars)
        self.measurements = measurements

    def get_panel_content(self):
        activations_str = ""
        for i, (mlp_act, raw_mlp) in enumerate(
            zip(self.measurements.mlp_normalized, self.measurements.mlp_activations)
        ):
            mlp_act_scalar = (
                mlp_act.item()
                if isinstance(mlp_act, (torch.Tensor, np.ndarray))
                else float(mlp_act)
            )
            raw_mlp_scalar = (
                raw_mlp.item()
                if isinstance(raw_mlp, (torch.Tensor, np.ndarray))
                else float(raw_mlp)
            )
            mlp_bar = "█" * int(abs(mlp_act_scalar))
            mlp_color = "yellow" if raw_mlp_scalar >= 0 else "magenta"

            activations_str += (
                f"[bold white]Layer {i:2d}[/] | "
                f"[bold yellow]:[/] [{mlp_color}]{mlp_bar.ljust(self.max_bar_length)}[/] [bold yellow]{raw_mlp_scalar:+.1f}[/]\n"
            )
        return activations_str


class AttentionEntropyPanel(PanelBase):
    def __init__(
        self, measurements: ModelMeasurements, max_bar_length=20, limit_chars=50
    ):
        super().__init__("Attention Entropy", "magenta", max_bar_length, limit_chars)
        self.measurements = measurements

    def get_panel_content(self):
        entropy_str = ""
        for i, (entropy_val, entropy_norm) in enumerate(
            zip(
                self.measurements.attention_entropy_values,
                self.measurements.attention_entropy_values_normalized,
            )
        ):
            entropy_val = float(entropy_val)
            entropy_norm = int(abs(float(entropy_norm)))
            entropy_bar = "█" * entropy_norm
            entropy_str += f"[bold white]Layer {i + 1:2d}[/] | [bold yellow]:[/] [{entropy_bar.ljust(self.max_bar_length)}] {entropy_val:.1f}\n"
        return entropy_str


class OutputDistributionPanel(PanelBase):
    def __init__(
        self,
        measurements: ModelMeasurements,
        max_bar_length=20,
        limit_chars=50,
        num_bins=20,
    ):
        super().__init__(
            "Output Distribution", "yellow", max_bar_length, limit_chars=None
        )
        self.measurements = measurements
        self.num_bins = num_bins

    def get_panel_content(self):
        next_token_probs = self.measurements.next_token_probs.cpu().numpy()
        sorted_probs = np.sort(next_token_probs)[
            -100:
        ]  # Taking top 100 highest probabilities

        bin_edges = np.linspace(0, len(sorted_probs), self.num_bins + 1, dtype=int)
        bin_sums = [
            np.sum(sorted_probs[bin_edges[i] : bin_edges[i + 1]])
            for i in range(self.num_bins)
        ]

        max_sum = max(bin_sums) if max(bin_sums) > 0 else 1
        bar_chars = ["█" * int((s / max_sum) * self.max_bar_length) for s in bin_sums]

        bin_labels = [
            f"{sorted_probs[bin_edges[i + 1] - 1]:.4f}" for i in range(self.num_bins)
        ]

        return "\n".join(
            [
                f"[bold yellow]{label}[/]: [bold cyan]{bar}[/]"
                for label, bar in zip(bin_labels, bar_chars)
            ][::-1]
        )


class GeneratedTextPanel(PanelBase):
    def __init__(
        self, measurements: ModelMeasurements, max_bar_length=20, limit_chars=50
    ):
        super().__init__("Generated Text", "green", max_bar_length, limit_chars)
        self.measurements = measurements

    def get_panel_content(self):
        text = Text(
            self.measurements.generated_text[-self.limit_chars :],
            style="bold bright_red",
        )
        text.append(self.measurements.predicted_char, style="bold on green")
        return text
