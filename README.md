# TMDB Movie Recommender

A full-stack, production-style **movie recommendation system** that combines **content-based machine learning** (TF-IDF + cosine similarity) with **live metadata from The Movie Database (TMDB) API**. Users search for films, browse curated feeds, and receive two complementary recommendation streams: *similar movies by content* and *more movies in the same genre*.

**Live demo:** [https://movie-rec-7nz8.onrender.com](https://movie-rec-7nz8.onrender.com) (API)  
**Repository:** [github.com/venkat-sai-ganesh/movie-rec](https://github.com/venkat-sai-ganesh/movie-rec)

---

## At a Glance (For Recruiters)

| Dimension | Summary |
|-----------|---------|
| **Project type** | End-to-end ML-powered web application |
| **Domain** | Recommender systems / information retrieval |
| **ML approach** | Content-based filtering (TF-IDF vectorization + cosine similarity) |
| **Dataset scale** | ~45,500 movies, 50,000-dimensional sparse feature space |
| **Backend** | FastAPI (async Python REST API) |
| **Frontend** | Streamlit (interactive UI) |
| **External integration** | TMDB REST API v3 (search, discover, metadata, posters) |
| **Deployment** | Render (Python 3.11.9) |
| **Key skills demonstrated** | NLP feature engineering, sparse linear algebra, API design, async I/O, model serialization, full-stack integration |

This is not a notebook-only prototype. It is a **deployed, API-first system** with a trained model loaded at startup, typed request/response schemas, error handling, and a consumer-facing UI.

---

## Problem Statement

Given a movie a user is interested in, the system should:

1. **Retrieve rich metadata** (title, overview, genres, posters, ratings).
2. **Recommend similar movies** based on textual content (plot, tagline, genres) — not just popularity.
3. **Recommend genre-aligned movies** from TMDB's live catalog for discovery beyond the local dataset.
4. **Present everything** in a fast, browsable UI with search autocomplete and poster grids.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         STREAMLIT FRONTEND (app.py)                     │
│  Home Feed │ Keyword Search │ Autocomplete │ Movie Details │ Posters   │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ HTTP (requests)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      FASTAPI BACKEND (main.py)                          │
│                                                                         │
│  ┌──────────────────┐    ┌─────────────────────────────────────────┐  │
│  │  REST Endpoints  │    │         Recommendation Engine             │  │
│  │  /health         │    │  ┌─────────────────────────────────────┐  │  │
│  │  /home           │    │  │ TF-IDF Content-Based (local ML)     │  │  │
│  │  /tmdb/search    │───▶│  │  • tfidf_matrix @ query_vector      │  │  │
│  │  /movie/id/{id}  │    │  │  • Cosine similarity ranking        │  │  │
│  │  /recommend/*    │    │  └─────────────────────────────────────┘  │  │
│  │  /movie/search   │    │  ┌─────────────────────────────────────┐  │  │
│  └──────────────────┘    │  │ Genre-Based (TMDB Discover API)     │  │  │
│                          │  │  • Filter by primary genre          │  │  │
│                          │  │  • Sort by popularity               │  │  │
│                          │  └─────────────────────────────────────┘  │  │
│                          └─────────────────────────────────────────┘  │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Startup: Load serialized ML artifacts (pickle)                  │  │
│  │  df.pkl │ indices.pkl │ tfidf.pkl │ tfidf_matrix.pkl             │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ async HTTP (httpx)
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    TMDB API v3 (api.themoviedb.org)                     │
│  /search/movie │ /movie/{id} │ /discover/movie │ /trending/movie/day   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Request Flow: Movie Details + Recommendations

```
User selects movie
       │
       ▼
Streamlit → GET /movie/id/{tmdb_id}          → TMDB movie details
       │
       ▼
Streamlit → GET /movie/search?query={title}    → Bundle endpoint
       │
       ├── TMDB search (best match)
       ├── TF-IDF recs (local cosine similarity on tags)
       │      └── For each rec title → TMDB search (attach poster)
       └── Genre recs (TMDB discover by first genre)
       │
       ▼
UI renders: Details + "Similar Movies (TF-IDF)" + "More Like This (Genre)"
```

---

## Machine Learning Pipeline

### 1. Data Source

| Asset | Description |
|-------|-------------|
| `movies_metadata.csv` | Raw TMDB/Kaggle dataset (~45,571 rows, 24 columns) |
| `df.pkl` | Preprocessed DataFrame (45,447 movies × 7 columns) |

**Raw columns used:** `title`, `overview`, `genres`, `tagline`, `vote_average`, `popularity`

**Preprocessed columns in `df.pkl`:**

| Column | Role |
|--------|------|
| `title` | Movie identifier for lookup |
| `overview` | Plot summary (source text) |
| `genres` | Genre labels (source text) |
| `tagline` | Marketing tagline (source text) |
| `vote_average` | User rating (metadata) |
| `popularity` | Popularity score (metadata) |
| `tags` | **Combined, cleaned feature string** for TF-IDF |

### 2. Feature Engineering

The `tags` column is a single concatenated text field built from:

- **Overview** (plot description)
- **Genres** (e.g., Animation, Comedy, Family)
- **Tagline** (short promotional text)

Text is lowercased and tokenized so that semantically related movies share vocabulary (e.g., *"enchanted board game"*, *"magical world"*, *"family comedy"*).

**Example — Toy Story:**

```
led woody andys toy live happily room andys birthday brings buzz lightyear
onto scene ... animation comedy family
```

### 3. Vectorization — TF-IDF

| Parameter | Value |
|-----------|-------|
| Algorithm | `sklearn.feature_extraction.text.TfidfVectorizer` |
| Vocabulary size | 50,000 features |
| Output matrix | Sparse CSR matrix `(45,447 × 50,000)` |
| Serialized as | `tfidf.pkl` (vectorizer), `tfidf_matrix.pkl` (document-term matrix) |

**TF-IDF (Term Frequency–Inverse Document Frequency)** down-weights common words across the corpus and up-weights terms that distinguish one movie from another. This is a classic, interpretable baseline for content-based recommendation and information retrieval.

### 4. Similarity & Ranking — Cosine Similarity

For a query movie at index `idx`:

```python
query_vector = tfidf_matrix[idx]
scores = (tfidf_matrix @ query_vector.T).toarray().ravel()
ranked_indices = np.argsort(-scores)  # descending
```

Because TF-IDF vectors are L2-normalized by default in scikit-learn, the dot product **equals cosine similarity**. The query movie itself is excluded; the top-N neighbors are returned as `(title, score)` pairs.

| Property | Detail |
|----------|--------|
| Complexity | O(n × d) per query via sparse matrix multiply (efficient for 45K × 50K sparse) |
| Interpretability | Higher score = more shared weighted terms in plot/genre/tagline |
| Cold start (local) | Requires title to exist in local `df.pkl` / `indices.pkl` |

### 5. Hybrid Recommendation Strategy

The system uses **two complementary strategies**:

| Strategy | Type | Signal | Source |
|----------|------|--------|--------|
| **TF-IDF similarity** | Content-based ML | Textual overlap in plot, genres, tagline | Local trained model |
| **Genre discover** | Metadata / rule-based | Shared primary genre, sorted by popularity | TMDB live API |

This hybrid design gives both **semantic similarity** (content) and **catalog breadth** (TMDB's full, up-to-date library with posters and ratings).

### 6. Model Artifacts

| File | Size | Contents |
|------|------|----------|
| `df.pkl` | ~28 MB | Preprocessed movie DataFrame |
| `indices.pkl` | ~1.4 MB | `title → row index` mapping (pandas Series) |
| `tfidf.pkl` | ~1.9 MB | Fitted `TfidfVectorizer` |
| `tfidf_matrix.pkl` | ~18 MB | Sparse document-term matrix |

Artifacts are loaded once at FastAPI startup (`@app.on_event("startup")`) and kept in memory for low-latency inference.

---

## Tech Stack

### Backend

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.11.9 | Runtime |
| **FastAPI** | 0.111.0 | Async REST API framework |
| **Uvicorn** | 0.30.1 | ASGI server |
| **Pydantic** | (via FastAPI) | Request/response validation & OpenAPI schemas |
| **httpx** | 0.27.0 | Async HTTP client for TMDB API |
| **python-dotenv** | 1.0.1 | Environment variable management |

### Machine Learning & Data

| Tool | Version | Purpose |
|------|---------|---------|
| **scikit-learn** | 1.5.1 | `TfidfVectorizer`, model serialization |
| **NumPy** | 2.1.0 | Similarity scoring, ranking |
| **SciPy** | 1.13.1 | Sparse matrix operations (CSR) |
| **pandas** | 2.2.2 | DataFrame handling, index mapping |
| **joblib** | 1.4.2 | ML ecosystem compatibility |

### Frontend

| Tool | Version | Purpose |
|------|---------|---------|
| **Streamlit** | 1.36.0 | Interactive web UI |
| **requests** | (dependency) | Sync HTTP calls to FastAPI backend |

### External Services

| Service | Purpose |
|---------|---------|
| **TMDB API v3** | Movie search, details, trending/popular feeds, genre discovery, poster URLs |
| **Render** | Cloud deployment (`runtime.txt`: `python-3.11.9`) |

---

## API Reference

Base URL (production): `https://movie-rec-7nz8.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/home?category=&limit=` | Home feed cards (trending, popular, top_rated, upcoming, now_playing) |
| `GET` | `/tmdb/search?query=&page=` | Raw TMDB keyword search (autocomplete + grid) |
| `GET` | `/movie/id/{tmdb_id}` | Full movie details from TMDB |
| `GET` | `/recommend/tfidf?title=&top_n=` | TF-IDF content-based recommendations only |
| `GET` | `/recommend/genre?tmdb_id=&limit=` | Genre-based recommendations via TMDB discover |
| `GET` | `/movie/search?query=&tfidf_top_n=&genre_limit=` | **Bundle:** details + TF-IDF recs + genre recs |

### Response Models (Pydantic)

- `TMDBMovieCard` — poster grid item (id, title, poster, release date, rating)
- `TMDBMovieDetails` — full detail view (overview, backdrop, genres)
- `TFIDFRecItem` — local rec with similarity score + optional TMDB card
- `SearchBundleResponse` — combined payload for the details page

Interactive docs: `{BASE_URL}/docs` (Swagger UI auto-generated by FastAPI)

---

## Frontend (Streamlit)

**File:** `app.py`

### Views

| View | Features |
|------|----------|
| **Home** | Category selector (trending, popular, top_rated, now_playing, upcoming), configurable poster grid |
| **Search** | Keyword input → TMDB autocomplete dropdown + word-matched result grid |
| **Details** | Poster, overview, genres, backdrop; TF-IDF and genre recommendation carousels |

### UX Details

- Session state + URL query params (`?view=details&id=`) for lightweight routing
- `@st.cache_data(ttl=30)` on API calls for autocomplete responsiveness
- Graceful fallbacks: if bundle endpoint fails, falls back to genre-only recommendations
- Responsive poster grid (4–8 columns, user-configurable)

---

## Project Structure

```
TMDBMovies/
├── main.py                 # FastAPI backend — ML inference + TMDB integration
├── app.py                  # Streamlit frontend — consumes FastAPI
├── requirements.txt        # Pinned Python dependencies
├── runtime.txt             # Render Python version (3.11.9)
├── .python-version         # Local pyenv version
├── .env                    # TMDB_API_KEY (not committed)
├── .gitignore
│
├── movies_metadata.csv     # Raw source dataset (~45.5K movies)
│
├── df.pkl                  # Preprocessed movie DataFrame
├── indices.pkl             # Title → index lookup
├── tfidf.pkl               # Fitted TfidfVectorizer
└── tfidf_matrix.pkl        # Sparse TF-IDF document matrix
```

---

## Environment Setup

### Prerequisites

- Python 3.11.9
- TMDB API key ([register free at themoviedb.org](https://www.themoviedb.org/settings/api))

### 1. Clone & install

```bash
git clone https://github.com/venkat-sai-ganesh/movie-rec.git
cd movie-rec
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

Create `.env` in the project root:

```env
TMDB_API_KEY=your_tmdb_api_key_here
```

### 3. Run the API

```bash
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Verify: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)  
API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 4. Run the frontend

In `app.py`, set `API_BASE` to your backend URL for local development:

```python
API_BASE = "http://127.0.0.1:8000"
```

Then:

```bash
streamlit run app.py
```

Open: [http://localhost:8501](http://localhost:8501)

---

## Engineering Highlights

### API Design

- **Separation of concerns:** ML inference and TMDB I/O live in the API layer; Streamlit is a thin client.
- **Typed contracts:** Pydantic models enforce consistent JSON across all endpoints.
- **Resilient TMDB integration:** Network errors map to HTTP 502; missing posters never crash recommendation endpoints.
- **Bundle endpoint:** `/movie/search` aggregates three data sources in one round-trip for the details page.

### ML / Inference

- **Precomputed matrix:** TF-IDF vectors are materialized at training time; inference is a single sparse dot product — no re-vectorization at request time.
- **Normalized title lookup:** Case-insensitive title → index map built at startup.
- **Fallback chain:** Bundle endpoint tries TMDB title first, then user query, then returns empty TF-IDF list rather than failing the whole response.

### Performance Considerations

- Sparse CSR storage (~18 MB vs. dense ~18 GB for full matrix)
- In-memory artifact loading avoids disk I/O per request
- Async TMDB calls via `httpx.AsyncClient`
- Streamlit API response caching (30s TTL) for search autocomplete

---

## Algorithms & Concepts Used

| Concept | Application in this project |
|---------|----------------------------|
| **TF-IDF** | Text vectorization of movie content |
| **Cosine similarity** | Ranking movies by content overlap |
| **Content-based filtering** | Recommendations from item features, not user history |
| **Sparse linear algebra** | Efficient similarity over 45K × 50K matrix |
| **Hybrid recommender** | ML content similarity + metadata genre filtering |
| **Information retrieval** | Keyword search, ranking, top-K retrieval |
| **Feature concatenation** | Multi-field text fusion (overview + genres + tagline) |
| **Model serialization** | Pickle artifacts for production serving |

---

## Limitations & Future Improvements

| Area | Current state | Possible enhancement |
|------|---------------|---------------------|
| Training pipeline | Artifacts pre-built; no training script in repo | Add reproducible `train.py` notebook/script |
| User personalization | None (item-item only) | Collaborative filtering or embedding-based models |
| Semantic understanding | Bag-of-words TF-IDF | Sentence-BERT / OpenAI embeddings for deeper semantics |
| Cold start | Local recs require title in dataset | Embedding fallback or TMDB-only content recs |
| Evaluation | No offline metrics in repo | Precision@K, MAP, diversity metrics on held-out set |
| Caching | Minimal | Redis for TMDB responses and hot recommendation queries |

---

## What This Project Demonstrates

- Building a **recommender system** from raw metadata to deployed inference
- Applying **classical NLP + IR techniques** (TF-IDF, cosine similarity) at production scale
- Designing a **clean REST API** that separates ML serving from UI concerns
- Integrating **third-party APIs** (TMDB) with robust error handling
- **Serializing and serving** scikit-learn models in a live web service
- Delivering an **end-to-end user experience** — browse, search, detail, recommend

---

## Author

**Venkat Sai Ganesh**  
GitHub: [venkat-sai-ganesh](https://github.com/venkat-sai-ganesh)

---

## License

This project uses the TMDB API under [TMDB's terms of use](https://www.themoviedb.org/documentation/api/terms-of-use). Dataset derived from publicly available TMDB/Kaggle movie metadata.
