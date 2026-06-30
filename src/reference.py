import json
from pathlib import Path
from dataclasses import dataclass

from src.inference import InferenceEngine

DATA_PATH = Path(__file__).parent.parent / "data" / "reference_proteins.json"


@dataclass
class ReferenceProtein:
    uniprot_id: str
    name: str
    gene: str
    organism: str
    sequence: str
    function: str
    go_terms: list[dict]


def load_reference_proteins() -> list[ReferenceProtein]:
    with open(DATA_PATH) as f:
        records = json.load(f)
    return [ReferenceProtein(**r) for r in records]


def build_reference_embeddings(
    engine: InferenceEngine,
) -> tuple[list[list[float]], list[ReferenceProtein]]:
    """
    Embeds all reference proteins in a single batched pass.
    Returns (embeddings, proteins) in the same order — index i in embeddings
    corresponds to index i in proteins.
    """
    proteins = load_reference_proteins()
    sequences = [p.sequence for p in proteins]
    embeddings = engine.embed(sequences)
    return embeddings, proteins
