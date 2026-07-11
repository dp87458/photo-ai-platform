import hashlib
import imagehash
import numpy as np
from PIL import Image
from collections import defaultdict
from itertools import combinations

from processing.embeddings import get_image_embedding


# ---------- LAYER 1: Exact duplicates (byte-identical files) ----------

def get_md5_hash(file_path: str) -> str:
    """Fingerprint based on exact file bytes. Same hash = byte-identical file."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_exact_duplicates(image_paths: list[str]) -> dict:
    hash_to_paths = defaultdict(list)
    for path in image_paths:
        hash_to_paths[get_md5_hash(path)].append(path)
    return {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}


# ---------- LAYER 2: Structural similarity (perceptual hashing) ----------

def compute_all_hashes(image_path: str) -> dict:
    """
    Four different hash algorithms, each sensitive to different kinds of
    visual change. Combining them gives a more reliable structural
    similarity signal than any single hash alone.
    """
    image = Image.open(image_path)
    return {
        "phash": imagehash.phash(image),
        "dhash": imagehash.dhash(image),
        "ahash": imagehash.average_hash(image),
        "whash": imagehash.whash(image),
    }


def hash_similarity_score(hashes1: dict, hashes2: dict) -> float:
    max_bits = 64
    phash_sim = 1 - (hashes1["phash"] - hashes2["phash"]) / max_bits
    dhash_sim = 1 - (hashes1["dhash"] - hashes2["dhash"]) / max_bits
    ahash_sim = 1 - (hashes1["ahash"] - hashes2["ahash"]) / max_bits
    whash_sim = 1 - (hashes1["whash"] - hashes2["whash"]) / max_bits
    return round(0.3 * phash_sim + 0.25 * dhash_sim + 0.2 * ahash_sim + 0.25 * whash_sim, 4)


# ---------- LAYER 3: Semantic similarity (CLIP) ----------

def cosine_similarity(vec1, vec2) -> float:
    v1, v2 = np.array(vec1), np.array(vec2)
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ---------- COMBINED PIPELINE ----------

def find_all_duplicates(
    image_paths: list[str],
    clip_threshold: float = 0.85,
    hash_threshold: float = 0.30,
    debug: bool = True,
) -> list[dict]:
    """
    A pair is flagged as duplicate only if BOTH conditions hold:
    - CLIP similarity is high (same semantic content — same specific photo,
      not just same category like 'a dog')
    - Hash similarity is at least moderate (some structural/pixel overlap,
      confirming it's a rotated/resized/recompressed version of the SAME
      photo, not just two different photos of the same subject)

    This AND logic is what separates:
    - True duplicate (rotated/zoomed copy): high CLIP + moderate-high hash
    - Two different dogs/cats/people: high CLIP + LOW hash (correctly rejected)
    - Unrelated photos: low CLIP + low hash (correctly rejected)
    """
    path_to_hashes = {p: compute_all_hashes(p) for p in image_paths}
    path_to_embedding = {p: get_image_embedding(p) for p in image_paths}

    results = []
    for (p1, h1), (p2, h2) in combinations(path_to_hashes.items(), 2):
        hash_score = hash_similarity_score(h1, h2)
        clip_score = cosine_similarity(path_to_embedding[p1], path_to_embedding[p2])

        if debug:
            print(f"{p1} <-> {p2} | hash={hash_score:.4f} clip={clip_score:.4f}")

        is_duplicate = clip_score >= clip_threshold and hash_score >= hash_threshold

        if is_duplicate:
            results.append({
                "photo1": p1,
                "photo2": p2,
                "hash_score": round(hash_score, 4),
                "clip_score": round(clip_score, 4),
            })

    return results


if __name__ == "__main__":
    from ingestion import ingest_photos

    photos = ingest_photos("local", "./sample_photos")
    paths = [p["path_or_url"] for p in photos]

    print("--- Exact Duplicates (MD5) ---")
    exact = find_exact_duplicates(paths)
    if exact:
        for h, group in exact.items():
            print(f"Hash {h[:8]}...: {group}")
    else:
        print("None found")

    print("\n--- DEBUG: all pairwise scores (hash + clip) ---")
    near = find_all_duplicates(paths, clip_threshold=0.85, hash_threshold=0.30, debug=True)

    print("\n--- Flagged Near-Duplicates ---")
    if near:
        for r in near:
            print(f"{r['photo1']} <-> {r['photo2']} | hash={r['hash_score']} clip={r['clip_score']}")
    else:
        print("None found")