import numpy as np
from rich.text import Text


# this should majorly go to plugins system
class PanelProvider:

    def __init__(self, max_bar_length, limit_chars):
        self.max_bar_length = max_bar_length
        self.limit_chars = limit_chars

    def create_activations_panel_content(self, mlp_normalized, mlp_activations):
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

    def create_entropy_panel_content(self, entropy_values, entropy_normalized):
        entropy_str = ""
        for i, (entropy_val, entropy_norm) in enumerate(
            zip(entropy_values, entropy_normalized)
        ):
            entropy_val = float(entropy_val)
            entropy_norm = int(abs(float(entropy_norm)))
            entropy_bar = "█" * entropy_norm
            entropy_str += f"[bold white]Layer {i + 1:2d}[/] | [bold yellow]:[/] [{entropy_bar.ljust(self.max_bar_length)}] {entropy_val:.1f}\n"
        return entropy_str

    def create_top_predictions_panel_content(
        self, decoded_tokens, top_ids, top_probs, logits
    ):

        entries = [
            f"[bold magenta]{token:<10}[/] "
            f"([bold yellow]{prob:>5.1%}[/bold yellow], [bold cyan]{logit:>4.1f}[/bold cyan])"
            for token, prob, logit in zip(
                decoded_tokens, top_probs.tolist(), logits[0, -1, top_ids].tolist()
            )
        ]

        return "\n".join(entries)

    def create_generated_text_panel(self, generated_text, predicted_char):
        return Text(
            generated_text[-self.limit_chars :], style="bold bright_red"
        ).append(predicted_char, style="bold on green")

    def create_prob_bin_panel(self, next_token_probs, num_bins=20):
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
            [
                f"[bold yellow]{label}[/]: [bold cyan]{bar}[/]"
                for label, bar, s in zip(bin_labels, bar_chars, bin_sums)
            ][::-1]
        )

        return bin_output
