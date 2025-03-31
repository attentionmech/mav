

<div align="center">
    <img width="270" height="250" alt="Screenshot" src="https://github.com/user-attachments/assets/a55ae5cb-9eff-4e43-a5c6-a81a1a9eb50f" />
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

MAV - Model Activations Visualiser

## Getting started  

#### METHOD 1: If `uv` is installed:  

```sh
uv run --with openmav mav
```

or 

```sh
uv run --with git+https://github.com/attentionmech/mav mav --model gpt2 --prompt "hello mello"
```  

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


#### METHOD 4: Inside Jupyter notebook/Colab

<a href="https://colab.research.google.com/gist/attentionmech/507312c98a6f49f420ec539c301dcb2d/openmav.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

<br>


You can replace `gpt2` with other Hugging Face models for example:  
- `meta-llama/Llama-3.2-1B`  
- `HuggingFaceTB/SmolLM-135M` 
- `gpt2-medium`
- `gpt2-large`


## UI

![image](https://github.com/user-attachments/assets/ed0dee9d-fe31-43a4-8a4a-303631680bc3)


## Tutorials

#### running MAV with a training loop with a custom model (not pretrained one)

`uv run examples/vis_train_loop.py`

#### running MAV with custom panel selection and arrangement

`uv run --with git+https://github.com/attentionmech/mav mav --model gpt2 --num-grid-rows 3 --selected-panels generated_text attention_entropy top_predictions --max-bar-length 20 --refresh-rate 0 --max-new-tokens 10000`

## Demos

- [interactive mode](https://x.com/attentionmech/status/1905732784314081511)
- [limit chars](https://x.com/attentionmech/status/1905760510445850709)
- [sample with temperature](https://x.com/attentionmech/status/1905886861245259857)
- [running with custom model](https://x.com/attentionmech/status/1906172982294376755)
- [panel selection](https://x.com/attentionmech/status/1906304032798339124)
- [running in colab notebook](https://x.com/attentionmech/status/1906657159355789593)

Note: explore it using the command line help as well, since many sampling params are exposed.
