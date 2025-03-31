from openmav.api.measurements import ModelMeasurements
from openmav.converters.data_converter import DataConverter


class StateProcessor:
    def __init__(self, backend, aggregation="l2", scale="linear", max_bar_length=20):
        self.data_converter = DataConverter()
        self.backend = backend
        self.aggregation = aggregation
        self.scale = scale
        self.max_bar_length = max_bar_length

    def next(
        self,
        generated_ids,
        next_token_id,
        hidden_states,
        attentions,
        logits,
        next_token_probs,
        top_ids,
        top_probs,
        backend,
    ):
        mlp_activations = self.data_converter.process_mlp_activations(
            hidden_states, self.aggregation
        )
        entropy_values = self.data_converter.process_entropy(attentions)

        mlp_normalized = self.data_converter.normalize_activations(
            mlp_activations, scale_type=self.scale, max_bar_length=self.max_bar_length
        )
        entropy_normalized = self.data_converter.normalize_entropy(
            entropy_values, scale_type=self.scale, max_bar_length=self.max_bar_length
        )

        generated_text = backend.decode(
            generated_ids[:-1],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        predicted_char = backend.decode(
            [next_token_id], clean_up_tokenization_spaces=True
        )

        decoded_tokens = [
            (
                self.backend.decode(
                    [token_id], clean_up_tokenization_spaces=True
                ).strip()[
                    :10
                ]  # TODO: this should happen in view layer
                or " "
            )
            for token_id in top_ids.tolist()
        ]

        return self._convert_to_model_measurements(
            {
                "mlp_activations": mlp_activations,
                "mlp_normalized": mlp_normalized,
                "entropy_values": entropy_values,
                "entropy_normalized": entropy_normalized,
                "generated_text": generated_text,
                "predicted_char": predicted_char,
                "next_token_probs": next_token_probs,
                "top_ids": top_ids,
                "top_probs": top_probs,
                "logits": logits,
                "decoded_tokens": decoded_tokens,
            }
        )

    def _convert_to_model_measurements(self, data_dict) -> ModelMeasurements:
        return ModelMeasurements(
            mlp_activations=data_dict["mlp_activations"],
            mlp_normalized=data_dict["mlp_normalized"],
            entropy_values=data_dict["entropy_values"],
            entropy_normalized=data_dict["entropy_normalized"],
            generated_text=data_dict["generated_text"],
            predicted_char=data_dict["predicted_char"],
            next_token_probs=data_dict["next_token_probs"],
            top_ids=data_dict["top_ids"],
            top_probs=data_dict["top_probs"],
            logits=data_dict["logits"],
            decoded_tokens=data_dict["decoded_tokens"],
        )
