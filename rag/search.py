import numpy as np
from processing.embeddings import get_image_embedding, get_text_embedding


def cosine_similarity(vec1, vec2) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_ensembled_text_embedding(query: str) -> list[float]:
    """
    Generates multiple phrasings of the same query and averages their
    embeddings. This is a documented CLIP technique (prompt ensembling)
    that reduces sensitivity to exact wording and improves zero-shot
    accuracy over a single raw query embedding.
    """
    templates = [
        f"a photo of {query}",
        f"a close-up photo of {query}",
        f"a picture showing {query}",
        f"{query}",
    ]

    embeddings = [np.array(get_text_embedding(t)) for t in templates]
    averaged = np.mean(embeddings, axis=0)
    averaged = averaged / np.linalg.norm(averaged)  # re-normalize after averaging

    return averaged.tolist()


def build_photo_index(image_paths: list[str]) -> dict:
    """
    Precomputes and stores an embedding for every photo in the library.
    In production this would be a vector database (pgvector/FAISS) — for
    now, an in-memory dict demonstrates the same core logic clearly.
    """
    return {path: get_image_embedding(path) for path in image_paths}


def search_photos(query: str, photo_index: dict, top_k: int = 5, debug: bool = False) -> list[dict]:
    """
    Takes a free-text query, embeds it (using prompt ensembling), and
    returns the top_k most semantically similar photos, ranked by
    relevance rather than filtered by a fixed threshold.
    """
    query_embedding = get_ensembled_text_embedding(query)

    scored_results = []
    for path, photo_embedding in photo_index.items():
        score = cosine_similarity(query_embedding, photo_embedding)
        scored_results.append({"path": path, "score": round(score, 4)})

    scored_results.sort(key=lambda x: x["score"], reverse=True)

    if debug:
        print(f"\n[DEBUG] Full ranked list for query: '{query}'")
        for r in scored_results:
            print(f"  {r['path']}: {r['score']}")

    return scored_results[:top_k]


if __name__ == "__main__":
    from ingestion import ingest_photos

    photos = ingest_photos("local", "./sample_photos")
    paths = [p["path_or_url"] for p in photos]

    print("--- Building photo index (embedding all photos once) ---")
    photo_index = build_photo_index(paths)
    print(f"Indexed {len(photo_index)} photos")

    test_queries = ["my dog photos", "my travel photos", "my cat photos", "pasta"]

    for query in test_queries:
        print(f"\n=== Search: '{query}' (top 3) ===")
        results = search_photos(query, photo_index, top_k=3, debug=True)
        print(f"\nTop results for '{query}':")
        for r in results:
            print(f"  {r['path']}  (score: {r['score']})")