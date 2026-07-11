from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import text
from db.postgres_setup import engine, vector_search
from db.mongo_setup import get_photos_for_person, faces_collection
from processing.embeddings import get_text_embedding
from rag.generate import generate_rag_response
from rag.search import build_photo_index

app = FastAPI(
    title="AI Photo Management Platform",
    description="RAG-powered photo search, categorization, dedup, and face grouping",
    version="1.0.0",
)


# ---------- Health check ----------

@app.get("/")
def root():
    return {"status": "running", "message": "AI Photo Platform API"}


# ---------- Search (natural language, RAG-powered) ----------

@app.get("/search")
def search(query: str = Query(..., description="Natural language search query"), top_k: int = 5):
    """
    Full RAG search: retrieves relevant photos using CLIP + pgvector,
    then generates a natural-language summary via LLM.
    """
    query_embedding = get_text_embedding(query)
    results = vector_search(query_embedding, top_k=top_k)

    if not results:
        return {"query": query, "answer": "No matching photos found.", "results": []}

    # Reuse the RAG generation logic, but built from Postgres results directly
    from rag.generate import build_context_from_results, client

    context = "\n".join(
        f"{i+1}. {r['path']} (category: {r['category']}, similarity: {r['similarity']})"
        for i, r in enumerate(results)
    )

    prompt = f"""You are a helpful photo assistant. A user searched: "{query}"
Matching photos found:
{context}
Write a short, friendly 1-2 sentence summary. Only reference photos listed above."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )

    return {
        "query": query,
        "answer": response.choices[0].message.content,
        "results": results,
    }


# ---------- Categories ----------

@app.get("/categories/{category_name}")
def get_by_category(category_name: str):
    """Browse all photos in a specific category (document, pet, travel, etc.)."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT path, category_confidence FROM photos WHERE category = :cat"),
            {"cat": category_name},
        )
        photos = [{"path": row[0], "confidence": row[1]} for row in result]

    if not photos:
        raise HTTPException(status_code=404, detail=f"No photos found in category '{category_name}'")

    return {"category": category_name, "count": len(photos), "photos": photos}


# ---------- Duplicates ----------

@app.get("/duplicates")
def get_duplicates():
    """List all detected duplicate groups."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT path, dedup_group_id FROM photos WHERE dedup_group_id IS NOT NULL")
        )
        rows = result.fetchall()

    groups = {}
    for path, group_id in rows:
        groups.setdefault(group_id, []).append(path)

    return {"duplicate_groups": groups, "total_groups": len(groups)}


# ---------- Faces / People ----------

@app.get("/faces/{cluster_id}")
def get_photos_by_person(cluster_id: int):
    """Get all photos containing a specific person (face cluster)."""
    photos = get_photos_for_person(cluster_id)
    if not photos:
        raise HTTPException(status_code=404, detail=f"No photos found for person cluster {cluster_id}")
    return {"cluster_id": cluster_id, "photo_count": len(photos), "photos": photos}


@app.get("/faces")
def list_all_people():
    """List all detected person clusters and how many photos each has."""
    all_docs = list(faces_collection.find({}))
    cluster_counts = {}
    for doc in all_docs:
        for cluster_id in doc.get("face_cluster_ids", []):
            cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1

    return {"people": [{"cluster_id": k, "photo_count": v} for k, v in cluster_counts.items()]}


# ---------- Documents / OCR ----------

@app.get("/documents")
def get_documents_with_text():
    """List all document-type photos with their extracted OCR text."""
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT path, category, ocr_text, ocr_confidence
                FROM photos
                WHERE ocr_text IS NOT NULL AND ocr_text != ''
            """)
        )
        docs = [
            {"path": row[0], "category": row[1], "text": row[2], "confidence": row[3]}
            for row in result
        ]

    return {"count": len(docs), "documents": docs}


# ---------- All photos (basic listing) ----------

@app.get("/photos")
def list_all_photos(limit: int = 50):
    """List all photos with their metadata."""
    with engine.connect() as conn:
        result = conn.execute(
            text("SELECT path, category, timestamp, latitude, longitude FROM photos LIMIT :limit"),
            {"limit": limit},
        )
        photos = [
            {"path": row[0], "category": row[1], "timestamp": str(row[2]), "lat": row[3], "lon": row[4]}
            for row in result
        ]

    return {"count": len(photos), "photos": photos}