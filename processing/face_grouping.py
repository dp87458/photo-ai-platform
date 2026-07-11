from deepface import DeepFace
import numpy as np
from sklearn.cluster import DBSCAN
from collections import defaultdict

def detect_faces_in_photo(image_path: str, min_confidence: float = 0.90) -> list[dict]:
    """
    Detects faces and filters out low-confidence detections (false
    positives from documents, textures, logos, etc.) using MTCNN's
    built-in confidence score for each detection.
    """
    faces = []
    try:
        results = DeepFace.represent(
            img_path=image_path,
            model_name="Facenet",
            enforce_detection=False,
            detector_backend="mtcnn",
        )

        for face_data in results:
            confidence = face_data.get("face_confidence", 0)
            if confidence < min_confidence:
                print(f"  Skipped low-confidence detection ({confidence:.3f}) in {image_path}")
                continue

            faces.append({
                "photo_path": image_path,
                "encoding": np.array(face_data["embedding"]),
                "confidence": confidence,
            })

    except Exception as e:
        print(f"No face detected or error in {image_path}: {e}")

    return faces


def collect_all_faces(image_paths: list[str]) -> list[dict]:
    all_faces = []
    for path in image_paths:
        faces = detect_faces_in_photo(path)
        all_faces.extend(faces)
        print(f"{path}: found {len(faces)} face(s)")
    return all_faces


def cluster_faces(all_faces: list[dict], eps: float = 10.0, min_samples: int = 1) -> dict:
    """
    Groups faces by identity using DBSCAN.
    NOTE: Facenet embeddings have a different scale than face_recognition's,
    so eps needs re-tuning empirically (start around 8-12, verify with debug output).
    """
    if not all_faces:
        return {}

    encodings = np.array([face["encoding"] for face in all_faces])

    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    labels = clustering.fit_predict(encodings)

    clusters = defaultdict(list)
    for face, label in zip(all_faces, labels):
        clusters[int(label)].append(face)

    return dict(clusters)


def summarize_clusters(clusters: dict):
    for cluster_id, faces in clusters.items():
        label = f"Person {cluster_id}" if cluster_id != -1 else "Unmatched/Noise"
        print(f"\n{label} ({len(faces)} face(s)):")
        for f in faces:
            print(f"  - {f['photo_path']}")


if __name__ == "__main__":
    from ingestion import ingest_photos

    photos = ingest_photos("local", "./sample_photos")
    paths = [p["path_or_url"] for p in photos]

    print("--- Detecting faces ---")
    all_faces = collect_all_faces(paths)
    print(f"\nTotal faces found: {len(all_faces)}")

    if all_faces:
        # DEBUG: print pairwise distances first, like we did for dedup,
        # so we tune eps based on real numbers, not guesses
        print("\n--- DEBUG: pairwise distances between faces ---")
        for i in range(len(all_faces)):
            for j in range(i + 1, len(all_faces)):
                dist = np.linalg.norm(all_faces[i]["encoding"] - all_faces[j]["encoding"])
                print(f"{all_faces[i]['photo_path']} <-> {all_faces[j]['photo_path']}: distance = {dist:.2f}")

        print("\n--- Clustering by identity ---")
        clusters = cluster_faces(all_faces, eps=10.0, min_samples=1)
        summarize_clusters(clusters)