from rich.panel import Panel

from openmav.view.panels.panel_provider import PanelProvider


class PanelCreator:

    def __init__(self, max_bar_length=20, limit_chars=50):
        self.panel_provider = PanelProvider(
            max_bar_length=max_bar_length, limit_chars=limit_chars
        )

    def get_panels(self, data):

        panel_definitions = {
            "top_predictions": Panel(
                self.panel_provider.create_top_predictions_panel_content(
                    data["decoded_tokens"],
                    data["top_ids"],
                    data["top_probs"],
                    data["logits"],
                ),
                title="Top Predictions",
                border_style="blue",
            ),
            "mlp_activations": Panel(
                self.panel_provider.create_activations_panel_content(
                    data["mlp_normalized"], data["mlp_activations"]
                ),
                title="MLP Activations",
                border_style="cyan",
            ),
            "attention_entropy": Panel(
                self.panel_provider.create_entropy_panel_content(
                    data["entropy_values"], data["entropy_normalized"]
                ),
                title="Attention Entropy",
                border_style="magenta",
            ),
            "output_distribution": Panel(
                self.panel_provider.create_prob_bin_panel(data["next_token_probs"]),
                title="Output Distribution",
                border_style="yellow",
            ),
            "generated_text": Panel(
                self.panel_provider.create_generated_text_panel(
                    data["generated_text"], data["predicted_char"]
                ),
                title="Generated text",
                border_style="green",
            ),
        }
        return panel_definitions
