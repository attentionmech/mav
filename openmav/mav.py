# @author: attentionmech

import argparse
import warnings

from openmav.backends.model_backend_transformers import TransformersBackend
from openmav.view.console_view import ConsoleMAV

warnings.filterwarnings("ignore")


def MAV(
    model: str,
    prompt: str,
    max_new_tokens: int = 200,
    aggregation: str = "l2",
    refresh_rate: float = 0.1,
    interactive: bool = False,
    device: str = "cpu",
    scale: str = "linear",
    limit_chars: int = 250,
    temp: float = 0.0,
    top_k: int = 50,
    top_p: float = 1.0,
    min_p: float = 0.0,
    repetition_penalty: float = 1.0,
    backend: str = "transformers",
    seed: int = 42,
    model_obj=None,  # pass model object compatible to your backend
    tokenizer_obj=None,  # pass tokenizer object compatible to your backend
    selected_panels=None,
    num_grid_rows=1,
    max_bar_length=50,
):

    if model is None:
        print("model name cannot be empty.")
        return

    if prompt is None or len(prompt) == 0:
        print("Prompt cannot be empty.")
        return

    if backend == "transformers":
        backend = TransformersBackend(
            model_name=model,
            device=device,
            seed=seed,
            model_obj=model_obj,
            tokenizer_obj=tokenizer_obj,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    manager = ConsoleMAV(
        backend=backend,
        refresh_rate=refresh_rate,
        interactive=interactive,
        limit_chars=limit_chars,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        min_p=min_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        aggregation=aggregation,
        scale=scale,
        max_bar_length=max_bar_length,
        num_grid_rows=num_grid_rows,
        selected_panels=selected_panels,
    )

    manager.ui_loop(prompt)


def main():
    parser = argparse.ArgumentParser(description="Model Activation Visualizer")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Hugging Face model name (default: gpt2)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Once upon a timeline ",
        help="Initial prompt for text generation",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=200, help="Number of tokens to generate"
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        choices=["l2", "max_abs"],
        default="l2",
        help="Aggregation method (l2, max_abs)",
    )
    parser.add_argument(
        "--refresh-rate",
        type=float,
        default=0.1,
        help="Refresh rate for visualization",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable interactive mode (press Enter to continue)",
        default=False,
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run the model on (cpu, cuda, mps).",
    )

    parser.add_argument(
        "--scale",
        type=str,
        choices=["linear", "log", "minmax"],
        default="linear",
        help="Scaling method for visualization (linear, log, minmax).",
    )

    parser.add_argument(
        "--limit-chars",
        type=int,
        default=250,
        help="Limit the number of tokens for visualization.",
    )

    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature (higher values = more randomness, default: 0.0)",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="top-k sampling (set to 0 to disable, default: 50)",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="top-p (nucleus) filtering (set to 1.0 to disable, default: 1.0)",
    )

    parser.add_argument(
        "--min-p",
        type=float,
        default=0.0,
        help="min_p sampling (default: 0.0)",
    )

    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Penalty for repeated words (default: 1.0, higher values discourage repetition)",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="transformers",
        choices=["transformers"],
        help="Backend to use for model provider (currently on transformers)",
    )

    # random seed
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--max-bar-length",
        type=int,
        default=50,
        help="UI bar max length counted in square characters")

    parser.add_argument(
        "--selected-panels",
        type=str,
        nargs="+",
        default=[
            "top_predictions",
            "output_distribution",
            "generated_text",
            "mlp_activations",
            "attention_entropy",
        ],
        help="List of selected panels. Default: top_predictions, output_distribution, "
         "generated_text, mlp_activations, attention_entropy."
    )

    parser.add_argument(
        "--num-grid-rows",
        type=int,
        default=2,
    )

    args = parser.parse_args()

    args_dict = vars(args)

    MAV(**args_dict)


if __name__ == "__main__":
    main()
