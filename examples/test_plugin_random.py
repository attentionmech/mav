import random

from openmav.mav import MAV
from openmav.view.panels.panel_base import PanelBase


class RandomBars(PanelBase):
    def __init__(self, measurements, max_bar_length=20, limit_chars=50):
        super().__init__("Random Bars", "green", max_bar_length, limit_chars)
        self.measurements = measurements

    def get_panel_content(self):
        bars_str = ""
        for i in range(5):
            value = random.uniform(-1, 1)
            bar_length = int(abs(value) * self.max_bar_length)
            bar = "█" * bar_length
            color = "cyan" if value >= 0 else "red"
            bars_str += f"[bold white]Bar {i:2d}[/] | [{color}]{bar.ljust(self.max_bar_length)}[/] [bold yellow]{value:+.2f}[/]\n"
        return bars_str


MAV(
    "gpt2",
    "hello world ",
    num_grid_rows=1,
    selected_panels=["generated_text", "RandomBars"],
    external_panels=[RandomBars],
    max_new_tokens=5,
)
