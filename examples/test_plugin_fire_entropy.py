import random

from openmav.mav import MAV
from openmav.view.panels.panel_base import PanelBase


class EntropyFire(PanelBase):
    def __init__(self, measurements, max_bar_length=20, limit_chars=50):
        super().__init__(
            "Entropy Fire",
            "red",
            max_bar_length=max_bar_length,
            limit_chars=limit_chars,
        )
        self.measurements = measurements

    def get_panel_content(self):
        fire_str = ""
        entropy_vals = self.measurements.attention_entropy_values[
            :10
        ]  # Take first 10 entropy values

        for i, entropy in enumerate(entropy_vals):
            intensity = min(int(entropy), 9)  # Normalize entropy to a scale of 0-9
            fire_colors = [
                "blue",
                "cyan",
                "green",
                "yellow",
                "orange",
                "red",
                "bright_red",
            ]
            color = fire_colors[intensity % len(fire_colors)]
            fire_str += f"[{color}]🔥[/]" * (intensity + 1) + "\n"

        return fire_str


MAV(
    "gpt2",
    "hello world",
    num_grid_rows=1,
    selected_panels=["generated_text", "EntropyFire"],
    external_panels=[EntropyFire],
    max_new_tokens=10,
)
