import numpy as np
import torch
from rich.text import Text

from openmav.api.measurements import ModelMeasurements
from openmav.view.panels.panel_base import PanelBase


class TopPredictionsPanel(PanelBase):
    def __init__(
        self,
        measurements: ModelMeasurements,
        max_bar_length: int = 20,
        limit_chars: int = 50,
    ):
        super().__init__(
            title="Top Predictions",
            border_style="blue",
            max_bar_length=max_bar_length,
            limit_chars=limit_chars,
        )
        self.measurements = measurements

    def get_panel_content(self):
        entries = [
            f"[bold magenta]{token:<10}[/] "
            f"([bold yellow]{prob:>5.1%}[/bold yellow], [bold cyan]{logit:>4.1f}[/bold cyan])"
            for token, prob, logit in zip(
                self.measurements.decoded_tokens,
                self.measurements.top_probs.tolist(),
                self.measurements.logits[0, -1, self.measurements.top_ids].tolist(),
            )
        ]
        return "\n".join(entries)


class MlpActivationsPanel(PanelBase):
    def __init__(
        self,
        measurements: ModelMeasurements,
        max_bar_length: int = 20,
        limit_chars: int = 50,
    ):
        super().__init__(
            title="MLP Activations",
            border_style="cyan",
            max_bar_length=max_bar_length,
            limit_chars=limit_chars,
        )
        self.measurements = measurements

    def get_panel_content(self):
        activations_str = ""
        for i, (mlp_act, raw_mlp) in enumerate(
            zip(self.measurements.mlp_normalized, self.measurements.mlp_activations)
        ):
            mlp_act_scalar = (
                mlp_act.item()
                if isinstance(mlp_act, (torch.Tensor, np.ndarray))
                else float(mlp_act)
            )
            raw_mlp_scalar = (
                raw_mlp.item()
                if isinstance(raw_mlp, (torch.Tensor, np.ndarray))
                else float(raw_mlp)
            )
            mlp_bar = "█" * int(abs(mlp_act_scalar))
            mlp_color = "yellow" if raw_mlp_scalar >= 0 else "magenta"

            activations_str += (
                f"[bold white]Layer {i:2d}[/] | "
                f"[bold yellow]:[/] [{mlp_color}]{mlp_bar.ljust(self.max_bar_length)}[/] [bold yellow]{raw_mlp_scalar:+.1f}[/]\n"
            )
        return activations_str


class AttentionEntropyPanel(PanelBase):
    def __init__(
        self,
        measurements: ModelMeasurements,
        max_bar_length: int = 20,
        limit_chars: int = 50,
    ):
        super().__init__(
            title="Attention Entropy",
            border_style="magenta",
            max_bar_length=max_bar_length,
            limit_chars=limit_chars,
        )
        self.measurements = measurements

    def get_panel_content(self):
        entropy_str = ""
        for i, (entropy_val, entropy_norm) in enumerate(
            zip(
                self.measurements.attention_entropy_values,
                self.measurements.attention_entropy_values_normalized,
            )
        ):
            entropy_val = float(entropy_val)
            entropy_norm = int(abs(float(entropy_norm)))
            entropy_bar = "█" * entropy_norm
            entropy_str += f"[bold white]Layer {i + 1:2d}[/] | [bold yellow]:[/] [{entropy_bar.ljust(self.max_bar_length)}] {entropy_val:.1f}\n"
        return entropy_str


class TokenAsciiArtPanel(PanelBase):
    def __init__(
        self,
        measurements: ModelMeasurements,
        max_bar_length: int = 20,
        limit_chars: int = 50,
    ):
        super().__init__(
            title="Token ASCII Art",
            border_style="yellow",
            max_bar_length=max_bar_length,
            limit_chars=None,
        )
        self.measurements = measurements
        self.last_tokens = []  # Store token history
        self.max_token_history = 20  # Increased from 3 to 10 for longer words
        
        # ASCII art patterns for big text
        self.alphabet = {
            'a': [
                "  █████  ",
                " ██   ██ ",
                "███████  ",
                "██   ██  ",
                "██   ██  ",
            ],
            'b': [
                "██████  ",
                "██   ██ ",
                "██████  ",
                "██   ██ ",
                "██████  ",
            ],
            'c': [
                " ██████ ",
                "██      ",
                "██      ",
                "██      ",
                " ██████ ",
            ],
            'd': [
                "██████  ",
                "██   ██ ",
                "██   ██ ",
                "██   ██ ",
                "██████  ",
            ],
            'e': [
                "███████ ",
                "██      ",
                "█████   ",
                "██      ",
                "███████ ",
            ],
            'f': [
                "███████ ",
                "██      ",
                "█████   ",
                "██      ",
                "██      ",
            ],
            'g': [
                " ██████  ",
                "██       ",
                "██   ███ ",
                "██    ██ ",
                " ██████  ",
            ],
            'h': [
                "██   ██ ",
                "██   ██ ",
                "███████ ",
                "██   ██ ",
                "██   ██ ",
            ],
            'i': [
                "██ ",
                "██ ",
                "██ ",
                "██ ",
                "██ ",
            ],
            'j': [
                "     ██ ",
                "     ██ ",
                "     ██ ",
                "██   ██ ",
                " █████  ",
            ],
            'k': [
                "██   ██ ",
                "██  ██  ",
                "█████   ",
                "██  ██  ",
                "██   ██ ",
            ],
            'l': [
                "██      ",
                "██      ",
                "██      ",
                "██      ",
                "███████ ",
            ],
            'm': [
                "███    ███ ",
                "████  ████ ",
                "██ ████ ██ ",
                "██  ██  ██ ",
                "██      ██ ",
            ],
            'n': [
                "███    ██ ",
                "████   ██ ",
                "██ ██  ██ ",
                "██  ██ ██ ",
                "██   ████ ",
            ],
            'o': [
                " ██████  ",
                "██    ██ ",
                "██    ██ ",
                "██    ██ ",
                " ██████  ",
            ],
            'p': [
                "██████  ",
                "██   ██ ",
                "██████  ",
                "██      ",
                "██      ",
            ],
            'q': [
                " ██████  ",
                "██    ██ ",
                "██    ██ ",
                "██ ▄▄ ██ ",
                " ██████  ",
                "    ▀▀   ",
            ],
            'r': [
                "██████  ",
                "██   ██ ",
                "██████  ",
                "██   ██ ",
                "██   ██ ",
            ],
            's': [
                " ██████  ",
                "██       ",
                " █████   ",
                "     ██  ",
                "██████   ",
            ],
            't': [
                "████████ ",
                "   ██    ",
                "   ██    ",
                "   ██    ",
                "   ██    ",
            ],
            'u': [
                "██    ██ ",
                "██    ██ ",
                "██    ██ ",
                "██    ██ ",
                " ██████  ",
            ],
            'v': [
                "██    ██ ",
                "██    ██ ",
                "██    ██ ",
                " ██  ██  ",
                "  ████   ",
            ],
            'w': [
                "██     ██ ",
                "██  █  ██ ",
                "██ ███ ██ ",
                "████ ████ ",
                "██     ██ ",
            ],
            'x': [
                "██   ██ ",
                " ██ ██  ",
                "  ███   ",
                " ██ ██  ",
                "██   ██ ",
            ],
            'y': [
                "██    ██ ",
                " ██  ██  ",
                "  ████   ",
                "   ██    ",
                "   ██    ",
            ],
            'z': [
                "███████ ",
                "    ██  ",
                "  ██    ",
                "██      ",
                "███████ ",
            ],
            ' ': [
                "    ",
                "    ",
                "    ",
                "    ",
                "    ",
            ],
            '.': [
                "   ",
                "   ",
                "   ",
                "   ",
                "██ ",
            ],
            ',': [
                "    ",
                "    ",
                "    ",
                " ██ ",
                "██  ",
            ],
            '!': [
                "██ ",
                "██ ",
                "██ ",
                "   ",
                "██ ",
            ],
            '?': [
                "██████  ",
                "     ██ ",
                "  ████  ",
                "        ",
                "  ██    ",
            ],
            '1': [
                " ██ ",
                "███ ",
                " ██ ",
                " ██ ",
                "███ ",
            ],
            '2': [
                "████  ",
                "   ██ ",
                " ███  ",
                "██    ",
                "█████ ",
            ],
            '3': [
                "████  ",
                "   ██ ",
                " ███  ",
                "   ██ ",
                "████  ",
            ],
            '4': [
                "██  ██ ",
                "██  ██ ",
                "█████  ",
                "    ██ ",
                "    ██ ",
            ],
            '5': [
                "█████ ",
                "██    ",
                "████  ",
                "   ██ ",
                "████  ",
            ],
            '6': [
                " ████  ",
                "██     ",
                "█████  ",
                "██  ██ ",
                " ████  ",
            ],
            '7': [
                "█████ ",
                "   ██ ",
                "  ██  ",
                " ██   ",
                "██    ",
            ],
            '8': [
                " ████  ",
                "██  ██ ",
                " ████  ",
                "██  ██ ",
                " ████  ",
            ],
            '9': [
                " ████  ",
                "██  ██ ",
                " █████ ",
                "    ██ ",
                " ████  ",
            ],
            '0': [
                " ████  ",
                "██  ██ ",
                "██  ██ ",
                "██  ██ ",
                " ████  ",
            ],
        }

    def get_panel_content(self):
        # Get the new token
        token = self.measurements.predicted_char
        
        # Add the token to our history
        self.last_tokens.append(token)
        
        # Keep only the last N tokens for display
        if len(self.last_tokens) > self.max_token_history:
            self.last_tokens = self.last_tokens[-self.max_token_history:]
            
        # Join the tokens to form the current word
        word = ''.join(self.last_tokens).strip()
        
        # If the word is empty, use a single space
        if not word:
            word = " "
            
        # Create the ASCII art for the word
        art_lines = [""] * 5  # Most characters are 5 lines tall
        
        # For each character in the word
        for char in word.lower():
            char_art = self.alphabet.get(char, self.alphabet.get(' '))
            
            # Add this character's art to each line
            for i in range(min(len(art_lines), len(char_art))):
                art_lines[i] += char_art[i]
        
        # If the ASCII art is too wide, truncate it with an ellipsis
        max_width = 80  # Maximum width for display
        if any(len(line) > max_width for line in art_lines):
            for i in range(len(art_lines)):
                if len(art_lines[i]) > max_width:
                    art_lines[i] = art_lines[i][:max_width-3] + "..."
        
        # Choose random colors for fun
        colors = ["red", "green", "blue", "magenta", "cyan", "yellow"]
        import random
        color = random.choice(colors)
        
        # Format with rich text formatting
        formatted_lines = [f"[bold {color}]{line}[/]" for line in art_lines]
        
        # Add the original token at the bottom for clarity
        token_display = word
        if not token_display.strip():
            token_display = "[SPACE]"
            
        return "\n".join(formatted_lines) + f"\n\n[bold white]'{token_display}'[/]"


class GeneratedTextPanel(PanelBase):
    def __init__(
        self,
        measurements: ModelMeasurements,
        max_bar_length: int = 20,
        limit_chars: int = 50,
    ):
        super().__init__(
            title="Generated Text",
            border_style="green",
            max_bar_length=max_bar_length,
            limit_chars=limit_chars,
        )
        self.measurements = measurements

    def get_panel_content(self):
        text = Text(
            self.measurements.generated_text[-self.limit_chars :],
            style="bold bright_red",
        )
        text.append(self.measurements.predicted_char, style="bold on green")
        return text
