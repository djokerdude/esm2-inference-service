from src.model import ESM2Model, _validate_sequences

# Number of sequences processed per forward pass.
# Larger = faster throughput but more VRAM. 8 is safe for CPU and small GPU.
DEFAULT_BATCH_SIZE = 8


class InferenceEngine:
    def __init__(
        self,
        model_size: str = "small",
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str | None = None,
    ):
        self.model = ESM2Model(model_size=model_size, device=device)
        self.batch_size = batch_size

    def embed(self, sequences: list[str], layer: int = -1) -> list[list[float]]:
        """
        Embeds an arbitrary number of sequences.

        Validates all inputs upfront, then chunks into batches sized to fit in VRAM.
        Returns Python floats (not tensors) so results are directly JSON-serializable.
        """
        _validate_sequences(sequences)

        results = []
        for batch in _chunk(sequences, self.batch_size):
            embeddings = self.model.embed(batch, layer=layer)
            results.extend(embeddings.cpu().tolist())
        return results


def _chunk(lst: list, size: int):
    for i in range(0, len(lst), size):
        yield lst[i : i + size]
