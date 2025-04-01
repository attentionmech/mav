

<div align="center">
    <img width="500" height="200" alt="Screenshot" src="https://github.com/user-attachments/assets/2474be26-6601-4bb2-b71a-f2eb529fad53" />
</div>
<br>

<div align="center">
    <img src="https://img.shields.io/pypi/v/openmav.svg" alt="PyPI">
    <img src="https://img.shields.io/pypi/pyversions/openmav" alt="Python Versions">
    <img src="https://img.shields.io/pypi/dm/openmav" alt="PyPI - Downloads">
    <img src="https://img.shields.io/github/stars/attentionmech/mav" alt="GitHub Repo stars">
    <img src="https://github.com/attentionmech/mav/actions/workflows/test.yml/badge.svg" alt="Build Status">
    <img src="https://img.shields.io/pypi/l/openmav" alt="License">
   <hr>
</div>
<br>

# Introduction

MAV - Model Activity Visualiser (for LLMs)

![test (1)](https://github.com/user-attachments/assets/8bca6807-325f-4863-ab1a-fe25bd7f4c46)


## Getting started  

#### METHOD 1: If `uv` is installed:  

```sh
uv run --with openmav mav
```

or 

```sh
uv run --with git+https://github.com/attentionmech/mav mav --model gpt2 --prompt "hello mello"
```  
<hr>

#### METHOD 2: Without `uv`:

1. Set up and activate a virtual environment  
2. Install the package:  
   
   ```sh
   pip install openmav
   ```  
   or

   ```sh
   pip install git+https://github.com/attentionmech/mav
   ```  
3. Run:  
   ```sh
   mav --model gpt2 --prompt "hello mello"
   ```
4. or Import
   ```python
   from openmav.mav import MAV

   MAV("gpt2", "Hello")
   ```
<hr>

#### METHOD 3: Locally from scratch

1. git clone https://github.com/attentionmech/mav  
2. cd mav
3. Set up and activate a virtual environment  
4. Install the package:  
   ```sh
   pip install .
   ```  
5. Run:  
   ```sh
   mav --model gpt2 --prompt "hello mello"
   ```

<hr>

#### METHOD 4: Inside Jupyter notebook/Colab

<a href="https://colab.research.google.com/gist/attentionmech/507312c98a6f49f420ec539c301dcb2d/openmav.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<br>


You can replace `gpt2` with other Hugging Face models for example:  
- `meta-llama/Llama-3.2-1B`  
- `HuggingFaceTB/SmolLM-135M` 
- `gpt2-medium`
- `gpt2-large`


<hr>

## Tutorials

#### Writing your custom plugin tutorial in colab

<a href="https://colab.research.google.com/gist/attentionmech/56062b4f6112c793b2c9360ee5a7dfb9/openmav.ipynb">writing custom plugin panel</a>

#### running MAV with a training loop with a custom model (not pretrained one)

`uv run examples/test_vis_train_loop.py`

#### running MAV with custom panel selection and arrangement

`uv run --with git+https://github.com/attentionmech/mav mav --model gpt2 --num-grid-rows 3 --selected-panels generated_text attention_entropy top_predictions --max-bar-length 20 --refresh-rate 0 --max-new-tokens 10000`

## Demos

- [Basic plugins](https://x.com/attentionmech/status/1906769030540824963)
- [Entropy Fire plugin](https://x.com/attentionmech/status/1906775229214663092)
- [interactive mode](https://x.com/attentionmech/status/1905732784314081511)
- [limit chars](https://x.com/attentionmech/status/1905760510445850709)
- [sample with temperature](https://x.com/attentionmech/status/1905886861245259857)
- [running with custom model](https://x.com/attentionmech/status/1906172982294376755)
- [panel selection](https://x.com/attentionmech/status/1906304032798339124)
- [running in colab notebook](https://x.com/attentionmech/status/1906657159355789593)

Note: explore it using the command line help as well, since many sampling params are exposed.

## Contributing

Clone the repository and install the package in development mode:

```sh
git clone https://github.com/attentionmech/mav
cd mav

# recommended
uv sync

# if you don't use uv
pip install -e .
```
