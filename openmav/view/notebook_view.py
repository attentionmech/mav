import time
from IPython.display import display, clear_output
from openmav.view.panels.panel_creator import PanelCreator

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel

class NotebookMAV:
    """
    Handles UI loop for Jupyter Notebooks.
    """
    
    def __init__(
        self,
        data_provider,
        model_name,
        refresh_rate=0.2,
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
        interactive=None,
    ):
        self.data_provider = data_provider
        self.refresh_rate = refresh_rate
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
        self.panel_creator = PanelCreator(
            max_bar_length=max_bar_length, limit_chars=limit_chars
        )

    def ui_loop(self, prompt):
        """
        Runs the UI loop, updating the display with new data from MAVGenerator.
        """
        console = Console()
        try:
            for data in self.data_provider.generate_tokens(
                prompt,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
                min_p=self.min_p,
                repetition_penalty=self.repetition_penalty,
            ):
                self._render_visualization(console, data)
                if self.refresh_rate > 0:
                    time.sleep(self.refresh_rate)
        finally:
            clear_output(wait=True)

    def _render_visualization(self, console, data):
        """
        Handles UI updates based on provided data.
        """
        panel_definitions = self.panel_creator.get_panels(data)
        selected_panels = self.selected_panels or list(panel_definitions.keys())
        panels = [panel_definitions[key] for key in selected_panels if key in panel_definitions]

        if not panels:
            raise ValueError("No valid panels provided")
        
        layout = Layout()
        layout.split_column(*[Layout(Panel(panel)) for panel in panels])
        
        clear_output(wait=True)
        console.print(layout)