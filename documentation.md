## 1. Introduction

OpenMAV (Model Activity Visualizer) is a Python-based tool designed to provide real-time visualizations of the inner workings of Large Language Models (LLMs) during text generation. 

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
  - Parses command-line arguments (if running from the command line).
  - Initializes the model backend (e.g., `TransformersBackend`).
  - Creates a `StateFetcher` to handle token generation and data processing.
  - Creates a `MainLoopManager` to manage the UI loop.
  - Starts the visualization loop.
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
  - Loads the pre-trained model and tokenizer from the Hugging Face Hub.
  - Tokenizes input text using the tokenizer.
  - Generates the next token by calling the model's `generate` method.
  - Decodes token IDs back into text.
  - Moves tensors to the specified device (CPU, CUDA, MPS).
- **Abstraction:** The backend design allows for potential future support of other model frameworks (e.g., PyTorch, TensorFlow) by implementing additional backend classes.
- **Key Methods:**
  - `initialize()`: Loads the model and tokenizer.
  - `generate(input_ids, ...)`: Generates the next token and returns the model's internal states (logits, hidden states, attention).
  - `tokenize(text)`: Converts text to token IDs.
  - `decode(token_ids, ...)`: Converts token IDs back to text.

### 3.3. `openmav.processors.state_fetcher.StateFetcher` (State Fetcher)

- **Role:** Handles the iterative process of generating tokens and fetching the model's internal state at each step. It acts as an intermediary between the `TransformersBackend` and the `MainLoopManager`.
- **Key Methods:**
  - `fetch_next(prompt, ...)`: The main generator function that yields processed data for each token generated.

### 3.4. `openmav.processors.state_processor.StateProcessor` (State Processor)

- **Role:** Processes the raw model outputs (hidden states, attention matrices, logits) into meaningful metrics for visualization.
- **Responsibilities:**
  - Computes attention entropy.
  - Processes MLP activations.
  - Normalizes activations and entropy values.
  - Decodes token IDs into text.
  - Packages all processed data into a `ModelMeasurements` object.

### 3.5. `openmav.api.measurements.ModelMeasurements` (Data Container)

- **Role:** A dataclass that holds all processed model data passed to the UI for visualization.

### 3.6. `openmav.view.main_loop_manager.MainLoopManager` (UI Manager)

- **Role:** Manages the text-based UI using the `rich` library.
- **Responsibilities:**
  - Creates a `rich.console.Console` object for output.
  - Creates a `rich.live.Live` object for dynamic updates.
  - Manages panel selection and layout.

### 3.7. `openmav.view.panels.*` (UI Panels)

- **Role:** Visualize the model's internal state.
- **Customization:** Users can create custom panels by subclassing `PanelBase`.

## 4. Features

- **Real-time Visualization**
- **Customizable Panels**
- **Interactive Mode**
- **Support for Hugging Face Transformers**
- **Command-Line Interface**
- **Training Loop Visualization**
- **Plugin System**

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

## 6. Command-Line Usage

Run:

```sh
mav --help
```

Options include:

- `--model`
- `--prompt`
- `--max-new-tokens`
- `--aggregation`
- `--refresh-rate`
- `--interactive`
- `--device`
- `--selected-panels`
- `--num-grid-rows`
- more...


## 7. Examples

Check out examples/ folder for custom plugin examples, as well as how to integrate this into training loop
