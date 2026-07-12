# AI Photo Management Platform

An AI-powered photo management system that connects to local storage (and Google Photos), automatically deduplicates, categorizes, and organizes photos using computer vision and LLM-powered natural language search (RAG).

## Features
- **Deduplication**: exact (MD5) + near-duplicate detection (multi-hash + CLIP semantic similarity)
- **Zero-shot categorization**: documents, receipts, prescriptions, pets, travel, people, food (via CLIP)
- **Face grouping**: automatic person clustering across photos (DeepFace + DBSCAN)
- **OCR**: text extraction from documents/receipts (Tesseract)
- **Natural language search (RAG)**: semantic search powered by CLIP + pgvector + LLM-generated summaries
- **REST API**: FastAPI with auto-generated docs

## Architecture
See `architecture.png` / `ARCHITECTURE.md` for the full diagram and explanation.

## Tech Stack
- **AI/ML**: CLIP (open_clip), DeepFace (Facenet), Tesseract OCR, scikit-learn (DBSCAN)
- **Backend**: FastAPI, Python 3.11
- **Databases**: PostgreSQL + pgvector (structured data + embeddings), MongoDB (face clusters)
- **LLM**: Groq (Llama 3.3) for RAG generation
- **Deployment**: Docker, docker-compose, Kubernetes manifests, Jenkins CI/CD

## Setup

### Prerequisites
- Docker Desktop
- Python 3.11+ (for local development outside Docker)

### Environment Variables
Create a `.env` file in the project root:
\`\`\`
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/photos_db
MONGO_URL=mongodb://localhost:27017
GROQ_API_KEY=your_groq_key
ANTHROPIC_API_KEY=your_anthropic_key
\`\`\`

### Run with Docker (recommended)
\`\`\`bash
docker-compose up --build
\`\`\`
API will be available at `http://localhost:8080`
Interactive docs: `http://localhost:8080/docs`

### Populate with sample data
\`\`\`bash
python -m db.populate
\`\`\`

## API Endpoints
| Endpoint | Description |
|---|---|
| `GET /search?query=...` | Natural language photo search (RAG) |
| `GET /categories/{category_name}` | Browse photos by category |
| `GET /duplicates` | List detected duplicate groups |
| `GET /faces` | List all detected people |
| `GET /faces/{cluster_id}` | Photos of a specific person |
| `GET /documents` | Documents with extracted OCR text |
| `GET /photos` | List all photos |

## Testing
\`\`\`bash
pytest tests/
\`\`\`

## Deployment
- **Local/Cloud (Docker)**: `docker-compose up`
- **Kubernetes**: manifests in `k8s/` — see `ARCHITECTURE.md` for deployment notes
- **CI/CD**: `Jenkinsfile` defines the build/test/deploy pipeline

## Design Decisions
See `DESIGN_DECISIONS.md` for detailed reasoning behind technology choices.