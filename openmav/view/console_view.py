import time

import numpy as np
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from openmav.view.panels.panel_provider import PanelProvider


class ConsoleMAV:
    """
    Handles UI loop
    """

    def __init__(
        self,
        data_provider,
        model_name,
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
        version=None,
    ):
        self.console = Console()
        self.data_provider = data_provider
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
        self.version = version
        self.model_name = model_name

    def ui_loop(self, prompt):
        """
        Runs the UI loop, updating the display with new data from MAVGenerator.
        """
        self.console.show_cursor(False)
        self.live.start()

        try:
            for data in self.data_provider.generate_tokens(
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
        panel_provider = PanelProvider(max_bar_length=self.max_bar_length)

        selected_panels = self.selected_panels

        panel_definitions = {
            "top_predictions": Panel(
                panel_provider.create_top_predictions_panel_content(
                    data["decoded_tokens"],
                    data["top_ids"],
                    data["top_probs"],
                    data["logits"],
                ),
                title="Top Predictions",
                border_style="blue",
            ),
            "mlp_activations": Panel(
                panel_provider.create_activations_panel_content(
                    data["mlp_normalized"], data["mlp_activations"]
                ),
                title="MLP Activations",
                border_style="cyan",
            ),
            "attention_entropy": Panel(
                panel_provider.create_entropy_panel_content(
                    data["entropy_values"], data["entropy_normalized"]
                ),
                title="Attention Entropy",
                border_style="magenta",
            ),
            "output_distribution": Panel(
                panel_provider.create_prob_bin_panel(data["next_token_probs"]),
                title="Output Distribution",
                border_style="yellow",
            ),
            "generated_text": Panel(
                panel_provider.create_generated_text_panel(
                    data["generated_text"], data["predicted_char"], self.limit_chars
                ),
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

        if not panels:
            # print exception that no valid panels are provided
            raise ValueError("No valid panels provided")

        num_rows = max(1, self.num_grid_rows)
        num_columns = (
            len(panels) + num_rows - 1
        ) // num_rows  # Best effort even distribution

        title_bar = Layout(
            Panel(
                f"| OpenMAV v{self.version} | {self.model_name}", border_style="white"
            ),
            size=3,
        )
        rows = [Layout() for _ in range(num_rows)]
        layout.split_column(title_bar, *rows)

        for i in range(num_rows):
            row_panels = panels[i * num_columns : (i + 1) * num_columns]
            if row_panels:
                rows[i].split_row(*[Layout(panel) for panel in row_panels])

        self.live.update(layout, refresh=True)
