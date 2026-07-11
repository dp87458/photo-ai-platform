import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URL)
db = mongo_client["photos_db"]
faces_collection = db["photo_faces"]


def save_faces_for_photo(photo_path: str, face_cluster_ids: list[int]):
    """
    Stores which face-cluster IDs (people) appear in a given photo.
    This is naturally variable-length (0 faces, 1 face, 5 faces) —
    exactly the kind of data MongoDB handles more naturally than a
    fixed-column SQL table would.
    """
    faces_collection.update_one(
        {"photo_path": photo_path},
        {"$set": {"photo_path": photo_path, "face_cluster_ids": face_cluster_ids}},
        upsert=True,  # create if doesn't exist, update if it does
    )


def get_photos_for_person(cluster_id: int) -> list[str]:
    """Given a person's cluster ID, find every photo they appear in."""
    results = faces_collection.find({"face_cluster_ids": cluster_id})
    return [doc["photo_path"] for doc in results]


if __name__ == "__main__":
    # quick test
    save_faces_for_photo("sample_photos/download (3).jpeg", [1])
    save_faces_for_photo("sample_photos/download (4).jpeg", [2])
    print(get_photos_for_person(1))