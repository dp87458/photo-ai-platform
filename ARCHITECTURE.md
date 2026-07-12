System Architecture — AI Photo Management Platform

1. Overview

This system ingests photos from local storage (and optionally Google Photos), processes them through a multi-stage AI pipeline — deduplication, zero-shot categorization, face grouping, and OCR — persists all results in PostgreSQL and MongoDB, and exposes everything through a FastAPI layer with a RAG-powered natural language search endpoint.

The architecture deliberately uses different AI techniques for different sub-problems rather than a single model for everything: perceptual hashing for structural duplicate detection, CLIP for semantic understanding (categorization and search), and a dedicated face-embedding model for identity clustering. This is explained in detail in DESIGN_DECISIONS.md.

2. High-Level System Diagram

┌───────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                │
│              Browser / Postman / Swagger UI (/docs)                     │
└──────────────────────────────────┬──────────────────────────────────────┘
                                     │ REST API (HTTP/JSON)
┌────────────────────────────────────▼──────────────────────────────────────┐
│                          API LAYER — FastAPI                              │
│                                                                             │
│   GET /search?query=...          GET /categories/{category_name}          │
│   GET /duplicates                GET /faces                                │
│   GET /faces/{cluster_id}        GET /documents                            │
│   GET /photos                                                              │
└───┬───────────┬────────────┬───────────────┬───────────────┬─────────────┘
    │           │            │               │               │
┌───▼───────┐ ┌─▼──────────┐ ┌▼─────────────┐ ┌▼─────────────┐ ┌▼────────────┐
│ INGESTION │ │ DEDUP      │ │ CATEGORIZ-   │ │ FACE          │ │ RAG SEARCH   │
│           │ │ ENGINE     │ │ ATION        │ │ GROUPING       │ │ ENGINE       │
│ Local     │ │            │ │              │ │                │ │              │
│ folder    │ │ MD5 exact  │ │ CLIP         │ │ DeepFace       │ │ CLIP text    │
│ scan +    │ │ match +    │ │ zero-shot:   │ │ (Facenet)      │ │ encoder →    │
│ Google    │ │ multi-hash │ │ image vs.    │ │ embeddings +   │ │ pgvector     │
│ Photos    │ │ (phash/    │ │ category     │ │ DBSCAN         │ │ similarity   │
│ API       │ │ dhash/     │ │ text cosine  │ │ clustering      │ │ search →     │
│ (OAuth)   │ │ ahash/     │ │ similarity   │ │ (auto-detects   │ │ LLM (Groq /  │
│           │ │ whash) +   │ │              │ │ person count)   │ │ Claude)      │
│           │ │ CLIP AND-  │ │              │ │                │ │ generates    │
│           │ │ logic      │ │              │ │                │ │ grounded     │
│           │ │            │ │              │ │                │ │ answer       │
└─────┬─────┘ └──────┬─────┘ └──────┬───────┘ └───────┬────────┘ └──────┬───────┘
      │              │              │                  │                 │
      └──────────────┴──────────────┴──────────────────┴─────────────────┘
                                     │
                    ┌────────────────▼─────────────────┐
                    │           STORAGE LAYER            │
                    │                                     │
                    │  PostgreSQL + pgvector               │
                    │  • photo metadata (timestamp, GPS,   │
                    │    camera info, category, dedup id)  │
                    │  • CLIP embeddings (512-dim vectors) │
                    │  • OCR extracted text + confidence   │
                    │                                       │
                    │  MongoDB                              │
                    │  • face_cluster_ids per photo         │
                    │    (variable-length array — natural   │
                    │    fit for documents over rigid rows) │
                    └────────────────────────────────────────┘

  External Services:
    • Groq API (Llama 3.3)         — RAG answer generation
    • Google Photos Library API     — optional ingestion source
    • Tesseract OCR engine          — document text extraction

3. Data Flow -- Photo Ingestion Pipeline

Photo added (local folder / Google Photos)
                    │
                    ▼
   ① Metadata extraction (EXIF: timestamp, GPS, camera make/model)
                    │
                    ▼
   ② CLIP image embedding generated (512-dim vector, ViT-B-32)
                    │
                    ▼
   ③ Deduplication check
        • MD5 hash → exact byte-identical match
        • Multi-hash (phash/dhash/ahash/whash) → structural similarity
        • CLIP cosine similarity → semantic similarity
        • Flagged as duplicate only if BOTH hash AND CLIP score are high
          (prevents "two different dogs" false positives from CLIP alone)
                    │
                    ▼
   ④ Zero-shot categorization
        • Photo embedding compared against category text embeddings
          ("a photo of a receipt", "a photo of a pet", etc.)
        • Highest-scoring category wins (relative ranking, not fixed
          threshold — CLIP's absolute similarity scores are compressed)
                    │
                    ▼
   ⑤ IF category ∈ {document, receipt, prescription}:
        → Tesseract OCR extracts text + per-word confidence score
                    │
                    ▼
   ⑥ Face detection + embedding (DeepFace / Facenet, MTCNN detector,
      confidence-filtered to reject false positives e.g. on documents)
                    │
                    ▼
   ⑦ Face clustering (DBSCAN across all faces in the library — groups
      same-person photos without knowing the number of people upfront)
                    │
                    ▼
   ⑧ Persistence
        • PostgreSQL: metadata + embedding + category + OCR text
        • MongoDB: face_cluster_ids array for this photo

4. Data Flow -- RAG Search Pipeline

User query: "show me my dog photos"
                    │
                    ▼
   ① Query embedded via CLIP text encoder
        • Prompt ensembling: multiple phrasings of the query generated
          and averaged into one embedding, improving robustness over
          a single raw phrasing (documented CLIP technique)
                    │
                    ▼
   ② RETRIEVAL — pgvector cosine similarity search
        • Query embedding compared against all stored photo embeddings
          directly in Postgres via the <=> operator
        • Top-K most relevant photos returned, ranked by similarity
                    │
                    ▼
   ③ AUGMENTATION
        • Retrieved results (filenames, categories, similarity scores)
          formatted into a structured text context block
                    │
                    ▼
   ④ GENERATION
        • Context injected into an LLM prompt with an explicit grounding
          instruction: "only reference photos listed above, do not invent
          results" — this is the core anti-hallucination safeguard
        • LLM (Llama 3.3 via Groq) generates a natural-language summary
                    │
                    ▼
   Final response: natural-language sentence + ranked list of real,
   retrieved matching photos

5. Deployment Architecture
Local / Deployment (active,demoed)

┌─────────────────────────────────────────────────────────┐
│                    docker-compose (local)                  │
│                                                              │
│   ┌────────────┐    ┌──────────────┐    ┌──────────────┐  │
│   │ photo-api  │────│  postgres    │    │  mongo        │  │
│   │ (FastAPI)  │    │  (pgvector)  │    │               │  │
│   │ :8080      │    │  :5432        │    │  :27017       │  │
│   └────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│   Networked via Docker's internal DNS — services reach      │
│   each other by service name (e.g. "postgres"), not         │
│   "localhost"                                                │
└─────────────────────────────────────────────────────────────┘

Production Path (designed,documented -- see k8s / and Jenkinsfile)

Developer pushes to GitHub
              │
              ▼
  Jenkins pipeline triggers:
    checkout → lint → run pytest suite → build Docker image
              │
              ▼
  Image pushed to AWS ECR (container registry)
              │
              ▼
  kubectl rolling update applied to Kubernetes cluster
              │
              ▼
  ┌─────────────────────────────────────────────┐
  │            Kubernetes Cluster (EKS)           │
  │                                                 │
  │   Deployment: photo-api (3 replicas)           │
  │     • LivenessProbe on / (health check)        │
  │     • Resource requests/limits per pod         │
  │     • Secrets mounted for DB URL / API keys    │
  │                                                 │
  │   Service: LoadBalancer → distributes traffic  │
  │   across the 3 replicas                        │
  │                                                 │
  │   RDS (managed Postgres) / managed MongoDB      │
  │   S3 for photo file storage                     │
  └─────────────────────────────────────────────────┘

6. Technology Choices Summary

### AI / Machine Learning

| Layer | Technology | Why |
|---|---|---|
| Image & text embeddings | CLIP (open_clip, ViT-B-32) | Zero-shot capability — images and text share one embedding space, enabling both categorization and natural language search with no training required |
| Deduplication | MD5 + multi-hash (phash, dhash, ahash, whash) + CLIP | Hashing catches structural duplicates like resizing or recompression. CLIP catches rotated or zoomed copies. Combined with AND logic (not OR) to prevent CLIP's category-level similarity from producing false positives on different subjects of the same type |
| Face recognition | DeepFace (Facenet backbone) | Installs cleanly via pip on Windows, with no C++ build toolchain required, unlike dlib-based alternatives |
| Face detector | MTCNN | Returns a confidence score per detection, allowing low-confidence false positives (e.g. on document images) to be filtered out |
| Identity clustering | DBSCAN | Automatically discovers the number of distinct people without needing that number specified upfront, and correctly isolates noisy or ambiguous detections instead of forcing them into the wrong group |
| OCR | Tesseract | Open-source and well-documented. Gated behind categorization so it only runs on document-like photos, saving compute |

### Backend & Data

| Layer | Technology | Why |
|---|---|---|
| API framework | FastAPI | Async-native, automatic OpenAPI/Swagger documentation, built-in request validation via Python type hints |
| Structured storage + vectors | PostgreSQL + pgvector | One database handles both relational metadata and vector similarity search, avoiding the operational overhead of running a separate vector database |
| Flexible storage | MongoDB | Naturally represents variable-length per-photo data such as face cluster membership, which would need awkward join tables in a strict relational schema |
| LLM (RAG generation) | Groq (Llama 3.3) | Free-tier access with fast inference. The architecture is provider-agnostic and can swap to Claude or OpenAI with minimal code changes |

### Deployment & Infrastructure

| Layer | Technology | Why |
|---|---|---|
| Containerization | Docker + docker-compose | Direct assignment requirement, and gives a reproducible environment across local and cloud deployment |
| Orchestration (designed) | Kubernetes | Self-healing via liveness probes, horizontal scaling via replicas, and zero-downtime rolling deployments |
| CI/CD (designed) | Jenkins | Automates lint, test, build, push, and deploy on every commit, preventing untested code from reaching production |


7. Known Limitations & Future Improvements

--Dedup and search currently use brute-force pairwise/linear comparison. Fine for a demo-scale dataset; at true 100,000-photo scale this would need approximate nearest-neighbor indexing (e.g., HNSW, already partially available via pgvector's ivfflat index) and hash-bucketing (a simplified LSH approach) to avoid O(n²) comparison cost.

--Google Photos integration is implemented but not demoed live, due to OAuth consent-screen setup time — local folder ingestion was prioritized to maximize coverage of AI/ML requirements within the timeline.

--Fine-tuning CLIP's classifier head (via LoRA) on domain-specific categories — e.g., distinguishing prescriptions from receipts more precisely — was scoped out due to time, but would improve categorization accuracy given a small labeled dataset.

--Kubernetes and Jenkins configurations are written and included but not live-deployed or executed against a real cluster, given the project timeline; the containerized application was validated end-to-end locally via docker-compose instead.

--Async background processing (Celery + Redis) for ingestion at scale is architecturally planned but not wired into the current synchronous pipeline, to keep debugging simpler within the available time.