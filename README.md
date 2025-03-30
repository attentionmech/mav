


<div align="center">
<img width=400 height=340 src="https://github.com/user-attachments/assets/bd439d67-6537-42ec-8a45-5b4390b821f1"/>
</div>


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
<br>


# Introduction

MAV or Model Activation Visualiser

## Getting started  

#### If `uv` is installed:  

```sh
uv run --with openmav mav
```

or 

```sh
uv run --with git+https://github.com/attentionmech/mav mav --model gpt2 --prompt "hello mello"
```  

#### Without `uv`:

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
   from openmav import MAV

   MAV("gpt2", "Hello")
   ```

#### Locally from scratch

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


You can replace `gpt2` with other Hugging Face models for example:  
- `meta-llama/Llama-3.2-1B`  
- `HuggingFaceTB/SmolLM-135M` 
- `gpt2-medium`
- `gpt2-large`


## UI

<img width="1098" alt="Screenshot" src="https://github.com/user-attachments/assets/0fc919c9-42a5-49d6-8471-2463668799c9" />

## Examples

#### running MAV with a training loop with a custom model (not pretrained one)

`uv run examples/vis_train_loop.py`


## Demos

- [interactive mode](https://x.com/attentionmech/status/1905732784314081511)
- [limit chars](https://x.com/attentionmech/status/1905760510445850709)
- [sample with temperature](https://x.com/attentionmech/status/1905886861245259857)

Note: explore it using the command line help as well, since many sampling params are exposed.

