```
+===========================================================================+
|          _____                    _____                    _____          |
|         /\    \                  /\    \                  /\    \         |
|        /::\____\                /::\    \                /::\____\        |
|       /::::|   |               /::::\    \              /:::/    /        |
|      /:::::|   |              /::::::\    \            /:::/    /         |
|     /::::::|   |             /:::/\:::\    \          /:::/    /          |
|    /:::/|::|   |            /:::/__\:::\    \        /:::/____/           |
|   /:::/ |::|   |           /::::\   \:::\    \       |::|    |            |
|  /:::/  |::|___|______    /::::::\   \:::\    \      |::|    |     _____  |
| /:::/   |::::::::\    \  /:::/\:::\   \:::\    \     |::|    |    /\    \ |
|/:::/    |:::::::::\____\/:::/  \:::\   \:::\____\    |::|    |   /::\____\|
|\::/    / ~~~~~/:::/    /\::/    \:::\  /:::/    /    |::|    |  /:::/    /|
| \/____/      /:::/    /  \/____/ \:::\/:::/    /     |::|    | /:::/    / |
|             /:::/    /            \::::::/    /      |::|____|/:::/    /  |
|            /:::/    /              \::::/    /       |:::::::::::/    /   |
|           /:::/    /               /:::/    /        \::::::::::/____/    |
|          /:::/    /               /:::/    /          ~~~~~~~~~~          |
|         /:::/    /               /:::/    /                               |
|        /:::/    /               /:::/    /                                |
|        \::/    /                \::/    /                                 |
|         \/____/                  \/____/                                  |
+===========================================================================+
```

<img width="1008" alt="Screenshot" src="https://github.com/user-attachments/assets/b565c7c9-bddb-4051-84d7-d279e7bc562b" />


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
   pip install git+https://github.com/attentionmech/mav
   ```  
3. Run:  
   ```sh
   mav --model gpt2 --prompt "hello mello"
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

## Demos

- [interactive mode](https://x.com/attentionmech/status/1905732784314081511)
- [limit chars](https://x.com/attentionmech/status/1905760510445850709)
- [scale factor](https://x.com/attentionmech/status/1905756260370165786)

## Explanation

At every point in prediction, multiple next tokens are possible, each with a different confidence level. The tokens and the numbers near them represent these probabilities.  

#### Layer-wise Activations 

Activations are numerical values representing the forward pass through the network during inference. Each layer (or block) in GPT-style models typically consists of:  
1. An MLP sub-block  
2. An attention sub-block  

For the MLP sub-block, we plot the L2 norm of activations per layer. Other metrics like average or max exist but don’t provide as much intuitive insight.  

#### Attention Sub-block  

For the attention sub-block, we measure entropy. In transformer architectures, attention determines how tokens influence one another. The entropy value gives a rough indication of how widely the attention is spread:  
- Low entropy → Sharp token-to-token relationships  
- High entropy → A broader, more diffused attention span  

These are just intuitive explanations—it's best to study these concepts from multiple sources to build a solid understanding.

## Contributing

IMP NOTE: The design is not good for scaling it right now to multiple backends, and stuff which i am planning.. so your pull requests will have to wait for sometime

please raise PRs to dev branch. 
