import numpy as np
import faiss

from src.reference import ReferenceProtein


class ProteinSearchIndex:
    def __init__(
        self,
        embeddings: list[list[float]],
        proteins: list[ReferenceProtein],
    ):
        self.proteins = proteins
        matrix = np.array(embeddings, dtype=np.float32)

        # Normalize so that inner product == cosine similarity.
        # Cosine similarity is standard for embedding comparison — it measures
        # directional similarity, not magnitude, which is what we want.
        faiss.normalize_L2(matrix)

        # IndexFlatIP: exact inner-product search (no approximation).
        # Right choice for ~100 proteins — approximation only pays off at 1M+.
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
    ) -> list[dict]:
        """
        Find the k most similar reference proteins to the query embedding.
        Returns a list of result dicts sorted by descending similarity.
        """
        vec = np.array([embedding], dtype=np.float32)
        faiss.normalize_L2(vec)

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            p = self.proteins[idx]
            results.append({
                "uniprot_id": p.uniprot_id,
                "name": p.name,
                "gene": p.gene,
                "organism": p.organism,
                "function": p.function,
                "go_terms": p.go_terms,
                "similarity": round(float(score), 4),
            })

        return results
