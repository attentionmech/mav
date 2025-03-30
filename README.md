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

<img width="1098" alt="Screenshot" src="https://github.com/user-attachments/assets/0fc919c9-42a5-49d6-8471-2463668799c9" />

Model Activations Visualiser

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

   MAV("gpt", "Hello")
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

## Demos

- [interactive mode](https://x.com/attentionmech/status/1905732784314081511)
- [limit chars](https://x.com/attentionmech/status/1905760510445850709)
- [sample with temperature](https://x.com/attentionmech/status/1905886861245259857)

Note: explore it using the command line help as well, since many sampling params are exposed.

