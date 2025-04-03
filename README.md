<div align="center">
  <img width="500" height="200" alt="MAV Logo" src="https://github.com/user-attachments/assets/2474be26-6601-4bb2-b71a-f2eb529fad53" />
</div>
<br>
<div align="center">
  <a href="https://pypi.org/project/openmav/"><img src="https://img.shields.io/pypi/v/openmav.svg" alt="PyPI"></a>
  <a href="https://pypi.org/project/openmav/"><img src="https://img.shields.io/pypi/pyversions/openmav" alt="Python Versions"></a>
  <a href="https://pypi.org/project/openmav/"><img src="https://img.shields.io/pypi/dm/openmav" alt="PyPI - Downloads"></a>
  <a href="https://github.com/attentionmech/mav"><img src="https://img.shields.io/github/stars/attentionmech/mav" alt="GitHub Repo stars"></a>
  <a href="https://github.com/attentionmech/mav/actions/workflows/test.yml"><img src="https://github.com/attentionmech/mav/actions/workflows/test.yml/badge.svg" alt="Build Status"></a>
  <a href="https://pypi.org/project/openmav/"><img src="https://img.shields.io/pypi/l/openmav" alt="License"></a>
</div>

# MAV - Model Activity Visualiser

> **Visualize the internal workings of Large Language Models as they generate text**

<div align="center">
  <img width="100%" alt="MAV Demo" src="https://github.com/user-attachments/assets/8bca6807-325f-4863-ab1a-fe25bd7f4c46" />
</div>

## 🚀 Getting Started

### Method 1: Using `uv` (Recommended)

```sh
# Run with PyPI package
uv run --with openmav mav

# Or run directly from GitHub
uv run --with git+https://github.com/attentionmech/mav mav --model gpt2 --prompt "hello mello"
```

**Note**: You can replace `gpt2` with any other Hugging Face model compatible with transformers:

- `HuggingFaceTB/SmolLM-135M`
- `gpt2-medium`
- `gpt2-large`
- `meta-llama/Llama-3.2-1B`

For gated repos, ensure you have done `huggingface-cli login` and your environment has access to it.


### Method 2: Using `pip`

1. Set up and activate a virtual environment
2. Install the package:
   ```sh
   # From PyPI
   pip install openmav
   
   # Or from GitHub
   pip install git+https://github.com/attentionmech/mav
   ```
3. Run the visualizer:
   ```sh
   mav --model gpt2 --prompt "hello mello"
   ```
4. Or import in your code:
   ```python
   from openmav.mav import MAV
   MAV("gpt2", "Hello")
   ```

### Method 3: Local Development

1. Clone the repository:
   ```sh
   git clone https://github.com/attentionmech/mav
   cd mav
   ```
2. Set up and activate a virtual environment
3. Install in development mode:
   ```sh
   pip install .
   ```
4. Run the visualizer:
   ```sh
   mav --model gpt2 --prompt "hello mello"
   ```

### Method 4: Jupyter Notebook/Colab

<a href="https://colab.research.google.com/gist/attentionmech/507312c98a6f49f420ec539c301dcb2d/openmav.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 📚 Documentation & Tutorials

### Documentation

Check out the [documentation.md](documentation.md) file for detailed information.

### Tutorials

#### Custom Plugin Development
<a href="https://colab.research.google.com/gist/attentionmech/56062b4f6112c793b2c9360ee5a7dfb9/openmav.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Writing Custom Plugin Panel"/></a>

#### Advanced Usage Examples

```sh
# Run MAV with a training loop and custom model
uv run examples/test_vis_train_loop.py

# Run with custom panel configuration
uv run --with git+https://github.com/attentionmech/mav mav \
  --model gpt2 \
  --num-grid-rows 3 \
  --selected-panels generated_text attention_entropy top_predictions \
  --max-bar-length 20 \
  --refresh-rate 0 \
  --max-new-tokens 10000
```

## 🎥 Demos

- [Basic plugins](https://x.com/attentionmech/status/1906769030540824963)
- [Interactive mode](https://x.com/attentionmech/status/1905732784314081511)
- [Limit characters](https://x.com/attentionmech/status/1905760510445850709)
- [Sample with temperature](https://x.com/attentionmech/status/1905886861245259857)
- [Running with custom model](https://x.com/attentionmech/status/1906172982294376755)
- [Panel selection](https://x.com/attentionmech/status/1906304032798339124)
- [Running in Colab notebook](https://x.com/attentionmech/status/1906657159355789593)

> **Note**: Explore additional options using the command line help, as many sampling parameters are exposed.

## 👥 Contributing

Clone the repository and install the package in development mode:

```sh
git clone https://github.com/attentionmech/mav
cd mav

# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## 🧠 Trivia

This project started from a small tweet while testing a simple terminal ui loop: <a href="https://x.com/attentionmech/status/1905018536042570084">tweet</a>

<img src="https://github.com/user-attachments/assets/3a6252fe-2bdb-4c9c-84a7-8404de5f2382" height=500 width=600/>


