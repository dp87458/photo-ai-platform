# Design Decisions

## Why CLIP for categorization/search but NOT for deduplication
CLIP embeddings capture semantic meaning (what category something belongs to),
not pixel-level identity. Early testing showed CLIP alone flagged two different
cats or two different people as "duplicates" because it correctly recognized
them as the same category, not the same specific photo. Deduplication instead
uses a combination of MD5 (exact matches) and multi-hash structural similarity
(phash/dhash/ahash/whash), with CLIP semantic similarity used as an additional
AND-condition (not OR) specifically to catch rotated/zoomed copies of the same
photo while still rejecting different-but-similar-category images.

## Why thresholds were empirically tuned, not guessed
For both deduplication and face clustering, initial thresholds were set based
on documentation defaults, then corrected by printing raw pairwise similarity
scores against real test photos and identifying the actual separation point
between true matches and false positives — rather than assuming a fixed
"reasonable-sounding" number.

## Why DeepFace (Facenet) instead of dlib-based face_recognition
dlib requires compiling from source on Windows, which needs Visual Studio
Build Tools and frequently fails without a properly configured C++ toolchain.
DeepFace provides the same core technique (CNN-based face embeddings) via a
pure pip install, with no compilation step — a pragmatic substitution with
no loss of approach validity.

## Why PostgreSQL + pgvector instead of a separate vector database
Rather than introducing a dedicated vector database (Pinecone, Chroma) as a
separate service, pgvector extends Postgres itself to support vector
similarity search. This reduces infrastructure complexity — one database
instead of two — while still providing production-grade ANN search via
the ivfflat index.

## Why MongoDB for face-cluster data specifically
Face-cluster membership per photo is inherently variable-length (0, 1, or
many people per photo). Modeling this in a fixed-column relational table
would require an awkward join table. MongoDB's document model naturally
represents this as a single array field per photo.

## Why zero-shot CLIP categorization instead of a custom-trained classifier
Training a custom image classifier requires labeled data and compute neither
available nor necessary here. CLIP's zero-shot approach — comparing image
embeddings against natural-language category descriptions — achieves
reasonable accuracy with zero training, and categories can be added or
changed at runtime by simply editing text descriptions.

## Why RAG instead of returning raw search results
Raw vector search returns a ranked list of file paths and similarity scores —
useful but not user-friendly. The RAG layer (retrieval + LLM generation)
converts this into natural-language summaries while explicitly constraining
the LLM's prompt to only reference retrieved results, preventing
hallucination of photos that don't exist in the actual search results.

## Known limitations / future improvements
- Dedup and search currently use brute-force pairwise comparison (O(n²)),
  fine for a demo dataset but would need optimization (e.g., approximate
  nearest neighbor indexing, LSH bucketing) at true 100,000-photo scale.
- Google Photos integration is implemented but not demoed live, due to
  OAuth setup time constraints — local folder ingestion was prioritized.
- Fine-tuning CLIP's classifier head (via LoRA) on domain-specific categories
  (e.g., distinguishing receipts from prescriptions more precisely) was
  scoped out due to time, but would improve categorization accuracy given
  a small labeled dataset.
- Kubernetes and Jenkins configurations are written and included but not
  live-deployed/run, given the project timeline — validated locally via
  docker-compose instead.