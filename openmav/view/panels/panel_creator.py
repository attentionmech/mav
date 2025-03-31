from openmav.api.measurements import ModelMeasurements
from openmav.view.panels.internal_panels import (AttentionEntropyPanel,
                                                 GeneratedTextPanel,
                                                 MLPActivationsPanel,
                                                 OutputDistributionPanel,
                                                 TopPredictionsPanel)


class PanelCreator:
    def __init__(self, max_bar_length=20, limit_chars=50, num_bins=20):
        self.max_bar_length = max_bar_length
        self.limit_chars = limit_chars
        self.num_bins = num_bins

    def get_panels(self, measurements: ModelMeasurements):
        return {
            "top_predictions": TopPredictionsPanel(
                measurements, self.max_bar_length, self.limit_chars
            ).get_panel(),
            "mlp_activations": MLPActivationsPanel(
                measurements, self.max_bar_length, self.limit_chars
            ).get_panel(),
            "attention_entropy": AttentionEntropyPanel(
                measurements, self.max_bar_length, self.limit_chars
            ).get_panel(),
            "output_distribution": OutputDistributionPanel(
                measurements, self.max_bar_length, self.num_bins
            ).get_panel(),
            "generated_text": GeneratedTextPanel(
                measurements, self.max_bar_length, self.limit_chars
            ).get_panel(),
        }
