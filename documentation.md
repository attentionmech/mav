# OpenMAV Documentation
> Model Activity Visualizer for Large Language Models

## 1. Introduction

OpenMAV (Model Activity Visualizer) is a Python-based tool designed to provide real-time visualizations of the inner workings of Large Language Models (LLMs) during text generation. The tool offers insights into model internals through an interactive, terminal-based interface.

## 2. High-Level Overview

OpenMAV operates by intercepting and processing the internal states of an LLM during token generation. It leverages the Hugging Face `transformers` library (or potentially other backends in the future) to load and interact with pre-trained models. The core workflow involves:

1. **Initialization:** The user specifies a model (e.g., `gpt2`), a prompt, and various visualization parameters. OpenMAV loads the model and tokenizer using the `transformers` library.

2. **Token Generation:** OpenMAV initiates a token generation loop, feeding the prompt (or previously generated tokens) to the model.

3. **State Interception:** During each generation step, OpenMAV captures the model's internal states, including:
   - Hidden layer activations (MLP activations)
   - Attention matrices
   - Logits and probabilities of the next token

4. **Data Processing:** The captured data is processed to compute relevant metrics (e.g., attention entropy, activation norms).

5. **Visualization:** The processed data is used to update a dynamic, text-based user interface (UI) using the `rich` library. This UI presents various panels that visualize the model's internal state.

6. **Iteration:** Steps 2-5 are repeated until the desired number of tokens are generated or the user interrupts the process.

## 3. Key Components

OpenMAV is built upon several key components that work together to achieve its visualization capabilities.

### 3.1. `openmav.mav.MAV` (Main Entry Point)

- **Role:** This is the main function/class that users interact with to start a visualization session. It orchestrates the entire process, from model loading to UI rendering.

- **Responsibilities:**
  - Parses command-line arguments (if running from the command line)
  - Initializes the model backend (e.g., `TransformersBackend`)
  - Creates a `StateFetcher` to handle token generation and data processing
  - Creates a `MainLoopManager` to manage the UI loop
  - Starts the visualization loop

- **Customization:** The `MAV` function accepts numerous parameters, allowing users to customize:
  - Model name
  - Prompt
  - Sampling parameters (temperature, top-k, top-p)
  - Visualization settings (refresh rate, selected panels, grid layout)
  - Device (CPU, CUDA, MPS)
  - External Panels

### 3.2. `openmav.backends.model_backend_transformers.TransformersBackend` (Model Backend)

- **Role:** Provides an abstraction layer for interacting with LLMs. It encapsulates the details of loading and running a specific type of model (currently, only Hugging Face `transformers` models are supported).

- **Responsibilities:**
  - Loads the pre-trained model and tokenizer from the Hugging Face Hub
  - Tokenizes input text using the tokenizer
  - Generates the next token by calling the model's `generate` method
  - Decodes token IDs back into text
  - Moves tensors to the specified device (CPU, CUDA, MPS)

- **Abstraction:** The backend design allows for potential future support of other model frameworks (e.g., PyTorch, TensorFlow) by implementing additional backend classes.

- **Key Methods:**
  - `initialize()`: Loads the model and tokenizer
  - `generate(input_ids, ...)`: Generates the next token and returns the model's internal states (logits, hidden states, attention)
  - `tokenize(text)`: Converts text to token IDs
  - `decode(token_ids, ...)`: Converts token IDs back to text

### 3.3. `openmav.processors.state_fetcher.StateFetcher` (State Fetcher)

- **Role:** Handles the iterative process of generating tokens and fetching the model's internal state at each step. It acts as an intermediary between the `TransformersBackend` and the `MainLoopManager`.

- **Key Methods:**
  - `fetch_next(prompt, ...)`: The main generator function that yields processed data for each token generated.

### 3.4. `openmav.processors.state_processor.StateProcessor` (State Processor)

- **Role:** Processes the raw model outputs (hidden states, attention matrices, logits) into meaningful metrics for visualization.

- **Responsibilities:**
  - Computes attention entropy
  - Processes MLP activations
  - Normalizes activations and entropy values
  - Decodes token IDs into text
  - Packages all processed data into a `ModelMeasurements` object

### 3.5. `openmav.api.measurements.ModelMeasurements` (Data Container)

- **Role:** A dataclass that holds all processed model data passed to the UI for visualization.

### 3.6. `openmav.view.main_loop_manager.MainLoopManager` (UI Manager)

- **Role:** Manages the text-based UI using the `rich` library.

- **Responsibilities:**
  - Creates a `rich.console.Console` object for output
  - Creates a `rich.live.Live` object for dynamic updates
  - Manages panel selection and layout

### 3.7. `openmav.view.panels.*` (UI Panels)

- **Role:** Visualize the model's internal state.
- **Customization:** Users can create custom panels by subclassing `PanelBase`.

## 5. Creating Custom Panels (Plugins)

Example:

```python
from openmav.mav import MAV
from openmav.view.panels.panel_base import PanelBase

class EntropyFire(PanelBase):
    def __init__(self, measurements):
        super().__init__("Entropy Fire", "red")
        self.measurements = measurements
    
    def get_panel_content(self):
        return "🔥 " * int(self.measurements.attention_entropy_values[0])

MAV("gpt2", "hello world", selected_panels=["generated_text", "EntropyFire"], external_panels=[EntropyFire])
```

Check [measurements.py](https://github.com/attentionmech/mav/blob/main/openmav/api/measurements.py) for metrics available.

## 6. Command-Line Usage

Run:

```sh
mav --help
```

## Command-Line Flags

The Model Activation Visualizer (MAV) can be configured using the following command-line flags:

| Flag                   | Type    | Default              | Description                                                  |
|------------------------|---------|----------------------|--------------------------------------------------------------|
| `--model`              | `str`   | `"gpt2"`             | Hugging Face model name. Specifies the model to use for text generation (e.g., `gpt2`, `bert-base-uncased`). |
| `--prompt`             | `str`   | `"Once upon a timeline "` | Initial prompt for text generation. The model starts generating text from this prompt. |
| `--max-new-tokens`     | `int`   | `200`                | Number of tokens to generate. Determines the maximum number of tokens the model will produce. |
| `--aggregation`        | `str`   | `"l2"`               | Aggregation method (`l2`, `max_abs`). Specifies how MLP activations are aggregated across layers. |
| `--refresh-rate`       | `float` | `0.2`                | Refresh rate for visualization (in seconds). Controls how often the UI updates in non-interactive mode. |
| `--interactive`        |         | `False`              | Enable interactive mode (press Enter to continue). Pauses after each token generation, waiting for user input. |
| `--device`             | `str`   | `"cpu"`              | Device to run the model on (`cpu`, `cuda`, `mps`). Selects the device for computation. |
| `--scale`              | `str`   | `"linear"`           | Scaling method for visualization (`linear`, `log`, `minmax`). Controls how activation values are scaled for display. |
| `--limit-chars`        | `int`   | `400`                | Limit the number of characters displayed in the generated text panel. |
| `--temp`               | `float` | `0.0`                | Sampling temperature. Controls the randomness of token sampling (higher = more random). |
| `--top-k`              | `int`   | `50`                 | Top-k sampling. Considers only the *k* most likely next tokens (set to 0 to disable). |
| `--top-p`              | `float` | `1.0`                | Top-p (nucleus) sampling. Selects tokens from the smallest set with cumulative probability exceeding *p* (set to 1.0 to disable). |
| `--min-p`              | `float` | `0.0`                | Minimal Probability value. |
| `--repetition-penalty` | `float` | `1.0`                | Penalty for repeated words. Discourages the model from repeating itself (higher = stronger penalty). |
| `--backend`            | `str`   | `"transformers"`     | Backend to use for the model (`transformers`). Currently, only the Hugging Face Transformers backend is supported. |
| `--seed`               | `int`   | `42`                 | Random seed for reproducibility. Ensures consistent results with the same input and parameters. |
| `--max-bar-length`     | `int`   | `35`                 | Maximum length of UI bars (in characters). Controls the length of bars used in the visualization panels. |
| `--selected-panels`    | `str`   | See Below            | List of selected panels to display. Specify panel names separated by spaces. |
| `--num-grid-rows`      | `int`   | `2`                  | The number of rows in the grid layout for panels. |
| `--version`            |         |                      | Displays the application version and exits. |

**Note on `--selected-panels`:**

The default list of selected panels is: `generated_text top_predictions output_distribution mlp_activations attention_entropy`

You can customize this list to display only the panels you are interested in.

## Internal Panels

`mav` comes with a set of built-in visualization panels that provide insights into the model's internal state during text generation. These panels can be selected using the `--selected-panels` command-line flag. Here's a description of each:

### 1. `generated_text`

*   **Description:** Displays the generated text so far, highlighting the most recently predicted token.
*   **Content:**
    *   The panel shows the generated text, limited by the `--limit-chars` flag.
    *   The most recently predicted token is highlighted (typically in green).
    *   The text is rendered using Rich's `Text` object for styling.
*   **Use Case:** Provides a clear view of the text the model is currently producing.

### 2. `top_predictions`

*   **Description:** Shows the top predicted tokens for the next position in the sequence, along with their probabilities and logits.
*   **Content:**
    *   Displays a list of the most likely tokens the model could generate next.
    *   For each token, it shows:
        *   The token itself.
        *   The probability (as a percentage) of that token being selected.
        *   The logit value (the raw output of the model before softmax).
*   **Use Case:** Helps understand the model's confidence in its predictions and see the alternatives it considered.

### 3. `output_distribution`

*   **Description:** Visualizes the distribution of probabilities across all possible tokens for the next position.
*   **Content:**
    *   Presents a histogram-like representation of the probability distribution.
    *   Sorted token probabilities are binned, and the sum of probabilities in each bin is represented by a bar.
    *   The height of the bar indicates the combined probability mass within that bin.
*   **Use Case:** Provides a high-level view of the model's uncertainty and the shape of the probability distribution.

### 4. `mlp_activations`

*   **Description:** Displays the activations of the Multi-Layer Perceptron (MLP) layers in the model.
*   **Content:**
    *   Shows the activation values for each MLP layer.
    *   Uses bars to represent the magnitude of the activation.
    *   The color of the bar typically indicates the sign (positive/negative) of the activation.
*   **Use Case:** Allows visualizing the internal computations of the feedforward networks within the model. Can provide insights into which layers are most active and how they are contributing to the output.

### 5. `attention_entropy`

*   **Description:** Shows the entropy of the attention distributions in each layer of the model.
*   **Content:**
    *   Displays the entropy values for each attention layer.
    *   Uses bars to represent the magnitude of the entropy.
    *   Higher entropy typically indicates more diverse attention patterns (the model is attending to a wider range of inputs). Lower entropy indicates more focused attention.
*   **Use Case:** Helps understand how the model is attending to different parts of the input sequence. Can indicate whether the model is focusing on specific words or relationships.

**Customization:**

You can customize the appearance of these panels (e.g., the maximum bar length, the number of characters displayed) using the command-line flags described in the previous section.

**Adding/Modifying Panels:**

While these panels are built-in, `mav` is designed to be extensible. You can create your own custom panels to visualize different aspects of the model's state. Refer to the documentation on creating plugins for more information.