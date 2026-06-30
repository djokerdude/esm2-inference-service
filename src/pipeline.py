def aggregate_go_terms(search_results: list[dict]) -> list[dict]:
    """
    Produce weighted GO term predictions from nearest-neighbor search results.

    Each GO term's confidence is the fraction of total similarity mass contributed
    by proteins that carry that term. A term shared by three high-similarity neighbors
    scores higher than one carried by a single distant neighbor.

    Example:
        neighbor A (similarity 0.96) has GO:0020037
        neighbor B (similarity 0.91) has GO:0020037 and GO:0015671
        total similarity = 0.96 + 0.91 = 1.87

        GO:0020037 weight = 0.96 + 0.91 = 1.87  → confidence = 1.87 / 1.87 = 1.00
        GO:0015671 weight = 0.91               → confidence = 0.91 / 1.87 = 0.49
    """
    total_similarity = sum(r["similarity"] for r in search_results)
    if total_similarity == 0:
        return []

    term_weights: dict[str, dict] = {}

    for result in search_results:
        weight = result["similarity"]
        for term in result["go_terms"]:
            tid = term["id"]
            if tid not in term_weights:
                term_weights[tid] = {
                    "id": tid,
                    "name": term["name"],
                    "category": term["category"],
                    "_weight": 0.0,
                }
            term_weights[tid]["_weight"] += weight

    results = []
    for term in term_weights.values():
        results.append({
            "id": term["id"],
            "name": term["name"],
            "category": term["category"],
            "confidence": round(term["_weight"] / total_similarity, 4),
        })

    return sorted(results, key=lambda t: t["confidence"], reverse=True)
