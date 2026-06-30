from src.pipeline import aggregate_go_terms


def _term(go_id, name="test term", category="biological_process"):
    return {"id": go_id, "name": name, "category": category}


def _result(similarity, *go_ids):
    return {"similarity": similarity, "go_terms": [_term(gid) for gid in go_ids]}


def test_empty_results_returns_empty():
    assert aggregate_go_terms([]) == []


def test_single_result_single_term_confidence_is_one():
    results = [_result(0.9, "GO:0001")]
    terms = aggregate_go_terms(results)
    assert len(terms) == 1
    assert terms[0]["confidence"] == 1.0


def test_term_shared_by_all_neighbors_scores_one():
    results = [
        _result(0.9, "GO:0001", "GO:0002"),
        _result(0.8, "GO:0001", "GO:0003"),
        _result(0.7, "GO:0001"),
    ]
    terms = {t["id"]: t for t in aggregate_go_terms(results)}
    assert terms["GO:0001"]["confidence"] == 1.0


def test_shared_term_scores_higher_than_unique():
    results = [
        _result(0.9, "GO:shared", "GO:only_a"),
        _result(0.8, "GO:shared", "GO:only_b"),
    ]
    terms = {t["id"]: t for t in aggregate_go_terms(results)}
    assert terms["GO:shared"]["confidence"] > terms["GO:only_a"]["confidence"]
    assert terms["GO:shared"]["confidence"] > terms["GO:only_b"]["confidence"]


def test_higher_similarity_neighbor_contributes_more_weight():
    # GO:high carried by 0.95 neighbor; GO:low carried by 0.5 neighbor
    results = [
        _result(0.95, "GO:high"),
        _result(0.50, "GO:low"),
    ]
    terms = {t["id"]: t for t in aggregate_go_terms(results)}
    assert terms["GO:high"]["confidence"] > terms["GO:low"]["confidence"]


def test_results_sorted_by_confidence_descending():
    results = [
        _result(0.9, "GO:0001", "GO:0002"),
        _result(0.8, "GO:0001"),
    ]
    terms = aggregate_go_terms(results)
    confidences = [t["confidence"] for t in terms]
    assert confidences == sorted(confidences, reverse=True)


def test_confidence_bounded_between_zero_and_one():
    results = [_result(0.9, "GO:0001", "GO:0002"), _result(0.7, "GO:0003")]
    for term in aggregate_go_terms(results):
        assert 0.0 <= term["confidence"] <= 1.0


def test_output_contains_required_fields():
    results = [_result(0.9, "GO:0001")]
    term = aggregate_go_terms(results)[0]
    assert "id" in term
    assert "name" in term
    assert "category" in term
    assert "confidence" in term
