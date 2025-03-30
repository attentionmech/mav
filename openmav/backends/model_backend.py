class ModelBackend:
    def __init__(self, model_name, model_obj=None, tokenizer_obj=None, device="cpu"):
        pass

    def initialize(self):
        raise NotImplementedError("Subclasses must implement initialize()")

    def generate(
        self,
        input_ids,
        temperature=1.0,
        top_k=50,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
    ):
        raise NotImplementedError("Subclasses must implement generate()")

    def tokenize(self, text):
        raise NotImplementedError("Subclasses must implement tokenize()")

    def decode(self, token_ids, **kwargs):
        raise NotImplementedError("Subclasses must implement decode()")
