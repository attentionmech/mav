import time
import numpy as np
import torch

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from openmav.processors.data_processor import MAVDataProcessor, MAVGenerator


class ConsoleMAV:
    """
    Handles UI and visualization.
    """

    def __init__(self, backend, refresh_rate=0.2, interactive=False, limit_chars=20):
        self.backend = backend
        self.console = Console()
        self.live = Live(auto_refresh=False)
        self.refresh_rate = refresh_rate
        self.interactive = interactive
        self.limit_chars = limit_chars
        self.generator = MAVGenerator(backend)  # Use MAVGenerator for token generation

    def ui_loop(self, prompt, **gen_kwargs):
        """
        Runs the UI loop, updating the display with new data from MAVGenerator.
        """
        self.console.show_cursor(False)
        self.live.start()

        try:
            for data in self.generator.generate_tokens(prompt, **gen_kwargs):
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

        activations_panel = Panel(
            self._create_activations_panel_content(data["mlp_normalized"], data["mlp_activations"]),
            title="MLP Activations",
            border_style="cyan",
        )

        entropy_panel = Panel(
            self._create_entropy_panel_content(data["entropy_values"], data["entropy_normalized"]),
            title="Attention Entropy",
            border_style="magenta",
        )

        predictions_panel = Panel(
            self._create_top_predictions_panel_content(data["top_ids"], data["top_probs"], data["logits"]),
            title="Top Predictions",
            border_style="blue",
        )

        prob_bin_panel = Panel(
            self._create_prob_bin_panel(data["next_token_probs"]),
            title="Output Distribution",
            border_style="yellow",
        )

        highlighted_text = Text(data["generated_text"], style="bold bright_red")
        highlighted_text.append(data["predicted_char"], style="bold on green")
        top_panel = Panel(
            highlighted_text,
            title=f"MAV: {self.backend.model_name}",
            border_style="green",
        )

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
