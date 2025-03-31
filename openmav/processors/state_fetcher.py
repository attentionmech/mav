import torch

from openmav.processors.state_processor import StateProcessor

# TOOD: move params to config


class StateFetcher:
    """
    Handles token generation and data processing.
    """

    def __init__(
        self,
        backend,
        max_new_tokens=100,
        aggregation="l2",
        scale="linear",
        max_bar_length=20,
    ):
        self.max_new_tokens = max_new_tokens
        self.state_processor = StateProcessor(
            backend, aggregation=aggregation, scale=scale, max_bar_length=max_bar_length
        )
        self.backend = backend

    def fetch_next(
        self,
        prompt,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
    ):
        """
        Generates tokens and yields processed data.
        """
        inputs = self.backend.tokenize(prompt)
        generated_ids = inputs.tolist()[0]

        for _ in range(self.max_new_tokens):
            outputs = self.backend.generate(
                generated_ids,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )
            logits = outputs["logits"]
            hidden_states = outputs["hidden_states"]
            attentions = outputs["attentions"]

            next_token_probs = torch.softmax(logits[:, -1, :], dim=-1).squeeze()
            top_probs, top_ids = torch.topk(next_token_probs, 20)
            next_token_id = torch.multinomial(next_token_probs, num_samples=1).item()
            generated_ids.append(next_token_id)

            measurement_data = self.state_processor.next(
                generated_ids,
                next_token_id,
                hidden_states,
                attentions,
                logits,
                next_token_probs,
                top_ids,
                top_probs,
                self.backend,
            )

            yield measurement_data  # Yield processed data for visualization
