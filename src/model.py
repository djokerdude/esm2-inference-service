import torch
from transformers import EsmModel, EsmTokenizer

# Maps a friendly name to HuggingFace model ID and its hidden dimension.
# Small is the default — fast on CPU, good for development and tests.
# Swap to "large" for production-quality embeddings without changing call sites.
MODEL_REGISTRY = {
    "small":  ("facebook/esm2_t6_8M_UR50D",    320),   # 6 layers,  8M params
    "medium": ("facebook/esm2_t12_35M_UR50D",   480),   # 12 layers, 35M params
    "large":  ("facebook/esm2_t33_650M_UR50D", 1280),   # 33 layers, 650M params
}

# Standard 20 amino acids + ambiguous codes ESM2 handles gracefully.
VALID_AA = set("ACDEFGHIKLMNPQRSTVWYBJOUXZ")
MAX_SEQUENCE_LENGTH = 1022  # ESM2 positional embeddings cap at 1024; 2 slots reserved for BOS/EOS tokens


class ESM2Model:
    def __init__(self, model_size: str = "small", device: str | None = None):
        if model_size not in MODEL_REGISTRY:
            raise ValueError(f"model_size must be one of {list(MODEL_REGISTRY)}, got '{model_size}'")

        model_id, self.hidden_dim = MODEL_REGISTRY[model_size]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = EsmTokenizer.from_pretrained(model_id)
        self.model = EsmModel.from_pretrained(model_id, add_pooling_layer=False).to(self.device)
        self.model.eval()

    def tokenize(self, sequences: list[str]) -> dict:
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH + 2,  # +2 for BOS/EOS
        ).to(self.device)

    def embed(self, sequences: list[str], layer: int = -1) -> torch.Tensor:
        """
        Returns mean-pooled embeddings of shape [batch, hidden_dim].
        layer=-1 selects the final transformer layer (recommended default).
        """
        _validate_sequences(sequences)
        inputs = self.tokenize(sequences)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states is a tuple of (num_layers + 1) tensors, each [batch, seq_len, hidden_dim].
        # Index 0 is the embedding layer; index -1 is the final transformer layer.
        hidden = outputs.hidden_states[layer]

        # Mean pool over sequence positions, masking out padding tokens so they
        # don't dilute the embedding — same principle as ignoring padding in matmul.
        mask = inputs["attention_mask"].unsqueeze(-1).float()  # [batch, seq_len, 1]
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # [batch, hidden_dim]

        return embeddings


def _validate_sequences(sequences: list[str]) -> None:
    if not sequences:
        raise ValueError("sequences list must not be empty")
    for i, seq in enumerate(sequences):
        if not seq:
            raise ValueError(f"sequence at index {i} is empty")
        invalid = set(seq.upper()) - VALID_AA
        if invalid:
            raise ValueError(f"sequence at index {i} contains invalid characters: {invalid}")
        if len(seq) > MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"sequence at index {i} has length {len(seq)}, "
                f"exceeds max {MAX_SEQUENCE_LENGTH}"
            )
