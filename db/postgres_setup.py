import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:yourpassword@localhost:5432/photos_db")
engine = create_engine(DATABASE_URL)


def setup_database():
    """
    Creates the pgvector extension and our main photos table.
    Run this once to initialize the database schema.
    """
    with engine.connect() as conn:
        # enable pgvector extension — adds vector similarity search to Postgres
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS photos (
                id SERIAL PRIMARY KEY,
                path TEXT UNIQUE NOT NULL,
                source TEXT,
                timestamp TIMESTAMP,
                latitude FLOAT,
                longitude FLOAT,
                camera_make TEXT,
                camera_model TEXT,
                category TEXT,
                category_confidence FLOAT,
                embedding VECTOR(512),
                dedup_group_id TEXT,
                ocr_text TEXT,
                ocr_confidence FLOAT
            );
        """))

        # index for fast vector similarity search — without this, searching
        # 100,000 embeddings would be slow; this builds a smart lookup structure
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS photos_embedding_idx
            ON photos USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))

        conn.commit()

    print("Database schema created successfully")


def insert_photo(photo_data: dict):
    """
    Inserts or updates a photo's full metadata + embedding into Postgres.
    Uses UPSERT (INSERT ... ON CONFLICT) so re-running ingestion on the
    same photo updates it instead of creating duplicates.
    """
    with engine.connect() as conn:
        conn.execute(text("""
            INSERT INTO photos (path, source, timestamp, latitude, longitude,
                                 camera_make, camera_model, category,
                                 category_confidence, embedding, dedup_group_id,
                                 ocr_text, ocr_confidence)
            VALUES (:path, :source, :timestamp, :latitude, :longitude,
                    :camera_make, :camera_model, :category,
                    :category_confidence, :embedding, :dedup_group_id,
                    :ocr_text, :ocr_confidence)
            ON CONFLICT (path) DO UPDATE SET
                category = EXCLUDED.category,
                category_confidence = EXCLUDED.category_confidence,
                embedding = EXCLUDED.embedding,
                dedup_group_id = EXCLUDED.dedup_group_id,
                ocr_text = EXCLUDED.ocr_text,
                ocr_confidence = EXCLUDED.ocr_confidence;
        """), photo_data)
        conn.commit()


def vector_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Runs similarity search DIRECTLY in Postgres using pgvector — this
    replaces the in-memory dict from Step 8's search.py with a real,
    persistent, scalable vector search.
    The <=> operator computes cosine distance natively in SQL.
    """
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT path, category, 1 - (embedding <=> :query_vec) AS similarity
            FROM photos
            ORDER BY embedding <=> :query_vec
            LIMIT :top_k;
        """), {"query_vec": str(query_embedding), "top_k": top_k})

        return [{"path": row[0], "category": row[1], "similarity": round(row[2], 4)} for row in result]


if __name__ == "__main__":
    setup_database()