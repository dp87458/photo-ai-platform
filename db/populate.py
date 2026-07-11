from ingestion import ingest_photos
from processing.metadata_extractor import extract_metadata
from processing.embeddings import get_image_embedding
from processing.categorization import build_category_embeddings, categorize_photo
from processing.dedup import find_all_duplicates
from processing.face_grouping import collect_all_faces, cluster_faces
from processing.ocr import extract_text_with_confidence
from db.postgres_setup import insert_photo, setup_database
from db.mongo_setup import save_faces_for_photo

def run_full_pipeline(folder_path: str):
    print("Setting up database schema...")
    setup_database()

    print("\nIngesting photos...")
    photos = ingest_photos("local", folder_path)
    paths = [p["path_or_url"] for p in photos]

    print("\nRunning dedup detection...")
    duplicates = find_all_duplicates(paths, debug=False)
    dedup_map = {}
    for i, dup in enumerate(duplicates):
        group_id = f"group_{i}"
        dedup_map[dup["photo1"]] = group_id
        dedup_map[dup["photo2"]] = group_id

    print("\nBuilding category embeddings...")
    category_embeddings = build_category_embeddings()

    print("\nRunning face detection + clustering...")
    all_faces = collect_all_faces(paths)
    face_clusters = cluster_faces(all_faces, eps=10.0, min_samples=1)

    # build a lookup: photo_path -> list of cluster IDs it contains
    photo_to_clusters = {}
    for cluster_id, faces in face_clusters.items():
        for face in faces:
            photo_to_clusters.setdefault(face["photo_path"], []).append(cluster_id)

    print("\nProcessing each photo (metadata, embedding, category, OCR)...")
    for path in paths:
        metadata = extract_metadata(path)
        embedding = get_image_embedding(path)
        category_result = categorize_photo(path, category_embeddings)

        ocr_text, ocr_conf = "", 0
        if category_result["category"] in ("document", "receipt", "prescription"):
            ocr_result = extract_text_with_confidence(path)
            ocr_text = ocr_result["text"]
            ocr_conf = ocr_result["avg_confidence"]

        insert_photo({
            "path": path,
            "source": "local",
            "timestamp": metadata["timestamp"],
            "latitude": metadata["latitude"],
            "longitude": metadata["longitude"],
            "camera_make": metadata["camera_make"],
            "camera_model": metadata["camera_model"],
            "category": category_result["category"],
            "category_confidence": category_result["confidence"],
            "embedding": str(embedding),
            "dedup_group_id": dedup_map.get(path),
            "ocr_text": ocr_text,
            "ocr_confidence": ocr_conf,
        })

        save_faces_for_photo(path, photo_to_clusters.get(path, []))
        print(f"  Saved: {path}")

    print("\nPipeline complete. All photos processed and stored.")


if __name__ == "__main__":
    run_full_pipeline("./sample_photos")