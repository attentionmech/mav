from abc import ABC, abstractmethod

from rich.panel import Panel


class PanelBase(ABC):
    def __init__(
        self, title: str, border_style: str, max_bar_length: int, limit_chars: int
    ):
        self.title = title
        self.border_style = border_style
        self.max_bar_length = max_bar_length
        self.limit_chars = limit_chars

    @abstractmethod
    def get_panel_content(self):
        pass

    def get_panel(self):
        return Panel(
            self.get_panel_content(), title=self.title, border_style=self.border_style
        )
