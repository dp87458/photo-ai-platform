import numpy as np
from processing.embeddings import get_image_embedding, get_text_embedding


# Our category definitions — phrased as natural captions, not bare keywords,
# because CLIP was trained on real image captions and matches this style
# far more accurately than single-word labels.
CATEGORIES = {
    "document": "a photo of a document or paper with text",
    "prescription": "a photo of a medical prescription",
    "receipt": "a photo of a shopping receipt or bill",
    "people": "a photo of one or more people",
    "travel": "a photo taken while traveling, showing landmarks or scenery",
    "pet": "a photo of a pet, dog, or cat",
    "food": "a photo of food or a meal",
    "screenshot": "a screenshot of a phone or computer screen",
    "other": "a random photo that doesn't fit any specific category",
}


def cosine_similarity(vec1, vec2) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def build_category_embeddings() -> dict:
    """
    Precompute embeddings for every category ONCE, not per photo —
    these never change, so recomputing them for every image would be
    wasted work. This is called once at startup.
    """
    return {label: get_text_embedding(description) for label, description in CATEGORIES.items()}


def categorize_photo(image_path: str, category_embeddings: dict) -> dict:
    """
    Compares one photo against every category and returns the best match
    plus a confidence score, using RELATIVE ranking (highest score wins)
    rather than a fixed absolute threshold — CLIP's similarity scores are
    naturally compressed, so relative comparison is the reliable approach.
    """
    image_embedding = get_image_embedding(image_path)

    scores = {
        label: cosine_similarity(image_embedding, cat_embedding)
        for label, cat_embedding in category_embeddings.items()
    }

    best_category = max(scores, key=scores.get)
    best_score = scores[best_category]

    return {
        "path": image_path,
        "category": best_category,
        "confidence": round(best_score, 4),
        "all_scores": {k: round(v, 4) for k, v in scores.items()},
    }


if __name__ == "__main__":
    from ingestion import ingest_photos

    photos = ingest_photos("local", "./sample_photos")
    paths = [p["path_or_url"] for p in photos]

    category_embeddings = build_category_embeddings()  # compute once

    print("--- Categorization Results ---")
    for path in paths:
        result = categorize_photo(path, category_embeddings)
        print(f"\n{result['path']}")
        print(f"  → Category: {result['category']} (confidence: {result['confidence']})")
        print(f"  → All scores: {result['all_scores']}")