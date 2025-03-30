import torch
import numpy as np


class DataConverter:

    @staticmethod
    def compute_entropy(attn_matrix):
        """Compute entropy of attention distributions per layer."""
        entropy = -torch.sum(attn_matrix * torch.log(attn_matrix + 1e-9), dim=-1)
        return entropy.mean(dim=-1).cpu().numpy()

    @staticmethod
    def process_mlp_activations(hidden_states, aggregation="l2"):
        """
        Process MLP activations based on specified aggregation method.

        Args:
            hidden_states (torch.Tensor): Hidden states from the model
            aggregation (str): Aggregation method to use

        Returns:
            numpy.ndarray: Processed MLP activations
        """
        activations = torch.stack([layer[:, -1, :] for layer in hidden_states])

        if aggregation == "l2":
            return torch.norm(activations, p=2, dim=-1).cpu().numpy()
        elif aggregation == "max_abs":
            return activations.abs().max(dim=-1).values.cpu().numpy()
        else:
            raise ValueError("Invalid aggregation method. Choose from: l2, max_abs.")

    @staticmethod
    def process_entropy(attentions):
        """
        Compute entropy values for attention matrices.

        Args:
            attentions (list): Attention matrices from the model

        Returns:
            numpy.ndarray: Entropy values for each layer
        """
        return np.array(
            [
                DataConverter.compute_entropy(attn[:, :, -1, :].cpu())
                for attn in attentions
            ]
        )

    @staticmethod
    def normalize_activations(activations, scale_type="linear", max_bar_length=20):
        """
        Normalize activations for visualization.

        Args:
            activations (numpy.ndarray): Raw activation values
            max_bar_length (int): Maximum length of visualization bar

        Returns:
            numpy.ndarray: Normalized activations
        """
        return DataConverter.apply_scaling(activations, scale_type, max_bar_length)

    @staticmethod
    def normalize_entropy(entropy_values, scale_type="linear", max_bar_length=20):
        """
        Normalize entropy values for visualization.

        Args:
            entropy_values (numpy.ndarray): Raw entropy values
            max_bar_length (int): Maximum length of visualization bar

        Returns:
            numpy.ndarray: Normalized entropy values
        """
        return DataConverter.apply_scaling(entropy_values, scale_type, max_bar_length)

    @staticmethod
    def apply_scaling(values, scale_type="linear", max_bar_length=20):
        """
        Apply scaling transformation to values for better visualization.

        Args:
            values (numpy.ndarray): Input values to be scaled.
            scale_type (str): Scaling method - 'linear', 'log', or 'minmax'.
            max_bar_length (int): Maximum length for visualization bars.

        Returns:
            numpy.ndarray: Scaled values.
        """
        values = np.array(values)  # Ensure input is a NumPy array

        if scale_type == "log":
            values = np.log1p(np.abs(values))  # log(1 + x) to handle zero values safely
        elif scale_type == "minmax":
            min_val, max_val = np.min(values), np.max(values)
            if max_val - min_val > 1e-9:  # Prevent division by zero
                values = (values - min_val) / (max_val - min_val)
            else:
                values = np.zeros_like(
                    values
                )  # If all values are the same, return zeros

        if np.max(values) > 0:
            return (values / np.max(values)) * max_bar_length  # Scale to max_bar_length
        return values
