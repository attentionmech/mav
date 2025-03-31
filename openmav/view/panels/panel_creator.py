import inspect
import re
from typing import List, Optional

from openmav.api.measurements import ModelMeasurements
from openmav.view.panels import internal_panels
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
        self.external_panels = external_panels or []  # Ensure external_panels is never None

    def get_panels(self, measurements: ModelMeasurements):
        # Get internal panel classes (still removing "Panel" suffix)
        internal_panel_classes = {
            capital_to_snake(name[: -len("Panel")]): cls
            for name, cls in inspect.getmembers(internal_panels, inspect.isclass)
            if issubclass(cls, PanelBase) and cls is not PanelBase
        }

        # Get external panel classes (using full name without "Panel" removal)
        external_panel_classes = {}
        if self.external_panels:
            for panel in self.external_panels:
                if isinstance(panel, type) and issubclass(panel, PanelBase):
                    #don't want to make any effort on external panel naming convention..
                    panel_name = panel.__name__
                    external_panel_classes[panel_name] = panel
                elif isinstance(panel, PanelBase):
                    # If it's an instance, get its class full name
                    panel_name = capital_to_snake(panel.__class__.__name__)
                    external_panel_classes[panel_name] = panel.__class__

        all_panel_classes = {**internal_panel_classes, **external_panel_classes}

        panel_definitions = {
            name: panel_cls(
                measurements, self.max_bar_length, self.limit_chars
            ).get_panel()
            for name, panel_cls in all_panel_classes.items()
        }

        if self.selected_panels is None:
            self.selected_panels = list(panel_definitions.keys())

        panels = [
            panel_definitions[key]
            for key in self.selected_panels
            if key in panel_definitions
        ]

        if not panels:
            raise ValueError("No valid panels provided")

        return panels