import inspect
import re
from typing import List, Optional

from openmav.api.measurements import ModelMeasurements
from openmav.view.panels import internal_panels  # Import the entire module
from openmav.view.panels.panel_base import PanelBase


def capital_to_snake(text):

    result = [text[0].lower()]
    for char in text[1:]:
        if char.isupper():
            result.append("_")
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


class PanelCreator:
    def __init__(
        self,
        max_bar_length=20,
        limit_chars=50,
        num_bins=20,
        selected_panels=None,
        external_panels: Optional[List[PanelBase]] = None,
    ):
        self.max_bar_length = max_bar_length
        self.limit_chars = limit_chars
        self.num_bins = num_bins
        self.selected_panels = selected_panels
        self.external_panels = external_panels

    def get_panels(self, measurements: ModelMeasurements):
        # Get all classes in `internal_panels` that inherit from `PanelBase`

        panel_classes = {
            capital_to_snake(name[: -len("Panel")]): cls
            for name, cls in inspect.getmembers(internal_panels, inspect.isclass)
            if issubclass(cls, PanelBase) and cls is not PanelBase
        }

        # import pdb; pdb.set_trace()

        # Create panel instances dynamically
        panel_definitions = {
            name: panel_cls(
                measurements, self.max_bar_length, self.limit_chars
            ).get_panel()
            for name, panel_cls in panel_classes.items()
        }

        if self.selected_panels is None:
            self.selected_panels = list(panel_definitions.keys())

        panels = [
            panel_definitions[key]
            for key in self.selected_panels
            if key in panel_definitions
        ]

        if not panels:
            # print exception that no valid panels are provided
            raise ValueError("No valid panels provided")

        return panels
