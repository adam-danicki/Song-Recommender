# Song Recommender API

Song Recommender is a FastAPI + PostgreSQL backend for searching a song catalog and generating playlists from user selected seed tracks.

**Search → Select up to 20 songs → Get 20 similar recommendations**

It uses the Million Song Dataset to build a content based recommender with feature scaling, optional PCA embeddings, and cosine similarity kNN retrieval. The system is designed for fast local development with Docker and reproducible ML artifacts saved to disk.

----------------------------------------------------------------------------------------

## What this project does

- Extracts song metadata and audio features from the Million Song Dataset
- Stores songs and feature vectors in PostgreSQL
- Builds a content based retrieval index using StandardScaler, optional PCA, and kNN cosine similarity
- Exposes an API to search songs and generate recommendations from selected seeds
- Supports multi seed recommendations by averaging seed embeddings into a user taste vector
- Saves model artifacts to disk so recommendations work without retraining

----------------------------------------------------------------------------------------

## Tech stack

- Python 3.12
- FastAPI (REST API + OpenAPI/Swagger UI)
- SQLAlchemy
- PostgreSQL
- scikit learn (StandardScaler, PCA, NearestNeighbors)
- NumPy and pandas
- Docker / Docker Compose

----------------------------------------------------------------------------------------

## Project Structure

```text
.
├── src/
│   ├── api.py                 # FastAPI endpoints for /search and /recommend
│   ├── db.py                  # SQLAlchemy engine setup using DATABASE_URL
│   ├── recommender.py         # Loads ML artifacts and returns top K recommendations
│   ├── ingest.py              # Runs schema.sql and ingests tracks.csv + features.csv into Postgres
│   ├── build_index.py         # Fits scaler, optional PCA, and kNN then saves artifacts to /models
│   └── extract.py             # Extracts MSD .h5 files into data/tracks.csv and data/features.csv
├── data/
│   ├── msd/                   # Million Song Subset directory
│   ├── tracks.csv             # Song metadata (track_id, title, artist, year, release, tags)
│   └── features.csv           # Numeric features used to build similarity vectors
├── models/
│   ├── scaler.pkl             # Fitted StandardScaler
│   ├── pca.pkl                # Fitted PCA model (optional)
│   ├── knn.pkl                # Fitted NearestNeighbors retriever (cosine)
│   ├── track_ids.npy          # Index position -> track_id mapping
│   ├── embeddings.npy         # Catalog embeddings (after transforms)
│   └── config.json            # Pipeline metadata + feature order
├── schema.sql                 # DB schema for songs and song_features
├── docker-compose.yml         # Local stack: API + Postgres
├── Dockerfile                 # API container build definition
├── requirements.txt           # Python dependencies
├── .env                       # Local environment variables (DO NOT commit)
└── README.md                  # Project documentation

----------------------------------------------------------------------------------------

## Data model

### Entities

- **Song**
  - `track_id`, `title`, `artist`, `year`, `release`
  - optional derived fields like `genre` and tag arrays from the dataset
- **SongFeature**
  - belongs to a Song
  - stores the numeric feature vector used for similarity search

### Relationships

- Song has one SongFeature
- SongFeature references Song by `track_id`

### Integrity rules

- `songs.track_id` is the primary key and uniquely identifies a song
- `song_features.track_id` is the primary key and also a foreign key to `songs.track_id`
- Cascading deletes remove features when a song is removed

----------------------------------------------------------------------------------------

## Feature vector

The recommender uses a fixed feature order stored in `models/config.json` so training time and recommendation time stay consistent.

Example feature order:
- duration
- tempo
- loudness
- key
- mode
- time_signature
- danceability
- energy

Feature scaling is learned using StandardScaler and applied consistently during retrieval.

----------------------------------------------------------------------------------------

## Recommendation approach

### Offline step build_index.py

- Loads all feature vectors from `song_features`
- Fits a StandardScaler across the full catalog
- Optionally fits PCA to produce compact embeddings
- Normalizes embeddings for cosine similarity behavior
- Fits a NearestNeighbors retriever using cosine distance
- Saves artifacts to `models/` so the API can recommend without retraining

Artifacts saved to `models/`:
- `scaler.pkl`
- `pca.pkl` optional
- `knn.pkl`
- `track_ids.npy`
- `embeddings.npy`
- `config.json`

### Online step recommender.py

- User selects 1 to 20 `track_ids`
- Fetches seed vectors from Postgres
- Applies the same scaler and PCA pipeline as the offline index
- Averages seed embeddings into a single taste vector
- Queries kNN for nearest songs by cosine distance
- Filters out seed songs and caps repeated artists
- Returns up to `k` results with metadata from `songs`

----------------------------------------------------------------------------------------

## API overview

OpenAPI Swagger UI:
- `http://localhost:8000/docs`

Core endpoints you can use right away:

### Health
- `GET /health` returns a basic status response

### Search
- `GET /search?q=jack&limit=20`  
Returns matching songs from the catalog and includes `track_id` so the frontend can select songs.

### Recommend
- `POST /recommend`

Request body:
- `track_ids` list of seed track ids, 1 to 20
- `k` number of recommendations to return, default 20
- `per_artist_cap` max number of recommendations per artist, default 2

Example request body:
```json
{
  "track_ids": ["TRBIAHA128F42A4C9B", "TRBIACJ128F93087A9"],
  "k": 20,
  "per_artist_cap": 2
}

----------------------------------------------------------------------------------------

## Running with Docker

### 1) Download dependencies local

- Python 3.12+ installed and available

```bash
pip install -r requirements.txt

### 2)Build and start services Docker

From the repo root:
```bash
docker compose up -d --build

API is available at:
- http://localhost:8000/docs

----------------------------------------------------------------------------------------

## Local environment variables

You will typically run extract.py, ingest.py, and build_index.py from your host machine. For that, .env should point to localhost.

Example .env:
```bash
DATABASE_URL=postgresql+psycopg2://songrec:songrec_pw@localhost:5432/songrec

Inside Docker Compose the API uses db as the hostname, and that is configured in docker-compose.yml.

----------------------------------------------------------------------------------------

## Building the catalog and index manually

These steps are run manually when you refresh data or reset the DB.

### Step A Extract MSD to CSV

Reads MSD .h5 files and writes CSVs into data/.
```bash
python src/extract.py



