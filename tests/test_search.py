import pytest
from src.search import ProteinSearchIndex
from src.reference import ReferenceProtein


def _protein(uid, name="test"):
    return ReferenceProtein(
        uniprot_id=uid, name=name, gene="X",
        organism="Test", sequence="MKTLL",
        function="test function", go_terms=[],
    )


def _index(n: int) -> tuple[ProteinSearchIndex, list[ReferenceProtein]]:
    """Build an index of n proteins with orthogonal unit-vector embeddings."""
    proteins = [_protein(f"P{i:05d}", f"Protein {i}") for i in range(n)]
    # Each embedding is a unit vector along a different dimension.
    # After FAISS L2 normalization they remain orthogonal.
    embeddings = []
    for i in range(n):
        vec = [0.0] * 320
        vec[i] = 1.0
        embeddings.append(vec)
    return ProteinSearchIndex(embeddings, proteins), proteins


def test_query_returns_k_results():
    index, _ = _index(5)
    query = [0.0] * 320
    query[0] = 1.0
    assert len(index.query(query, k=3)) == 3


def test_top_result_is_nearest_neighbor():
    index, proteins = _index(5)
    # Query exactly matches protein 2 (unit vector along dim 2)
    query = [0.0] * 320
    query[2] = 1.0
    results = index.query(query, k=5)
    assert results[0]["uniprot_id"] == "P00002"


def test_results_sorted_by_similarity_descending():
    index, _ = _index(5)
    query = [0.0] * 320
    query[0] = 1.0
    results = index.query(query, k=5)
    similarities = [r["similarity"] for r in results]
    assert similarities == sorted(similarities, reverse=True)


def test_k_larger_than_index_does_not_raise():
    index, _ = _index(3)
    query = [0.0] * 320
    query[0] = 1.0
    results = index.query(query, k=10)
    assert len(results) <= 3


def test_result_contains_required_fields():
    index, _ = _index(2)
    query = [0.0] * 320
    query[0] = 1.0
    result = index.query(query, k=1)[0]
    for field in ("uniprot_id", "name", "gene", "organism", "function", "go_terms", "similarity"):
        assert field in result


def test_similarity_bounded_between_zero_and_one():
    index, _ = _index(5)
    query = [0.0] * 320
    query[0] = 1.0
    for result in index.query(query, k=5):
        assert 0.0 <= result["similarity"] <= 1.0
