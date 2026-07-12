import pytest
from processing.dedup import get_md5_hash, hash_similarity_score, compute_all_hashes
from processing.categorization import categorize_photo, build_category_embeddings
from rag.search import cosine_similarity


def test_md5_hash_consistency():
    """Same file should always produce the same MD5 hash."""
    hash1 = get_md5_hash("./sample_photos/download (1).jpeg")
    hash2 = get_md5_hash("./sample_photos/download (1).jpeg")
    assert hash1 == hash2


def test_cosine_similarity_identical_vectors():
    """Identical vectors should have similarity of 1.0."""
    vec = [0.1, 0.2, 0.3, 0.4]
    similarity = cosine_similarity(vec, vec)
    assert abs(similarity - 1.0) < 0.001


def test_cosine_similarity_range():
    """Cosine similarity should always be between -1 and 1."""
    vec1 = [0.5, -0.2, 0.8, 0.1]
    vec2 = [0.1, 0.9, -0.3, 0.4]
    similarity = cosine_similarity(vec1, vec2)
    assert -1.0 <= similarity <= 1.0


def test_hash_similarity_identical_images():
    """Same image compared to itself should score maximum similarity."""
    hashes = compute_all_hashes("./sample_photos/download (1).jpeg")
    score = hash_similarity_score(hashes, hashes)
    assert score == 1.0


def test_categorization_returns_valid_category():
    """Categorization should always return one of the defined categories."""
    category_embeddings = build_category_embeddings()
    result = categorize_photo("./sample_photos/download (1).jpeg", category_embeddings)
    assert result["category"] in category_embeddings.keys()
    assert 0 <= result["confidence"] <= 1