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
Data model
Tables

songs

track_id primary key

title not null

artist not null

year nullable

release nullable

genre derived from top terms or tags

artist_terms_top text array

artist_mbtags_top text array

song_features

track_id primary key references songs(track_id) on delete cascade

features double precision array used for similarity search

Feature vector

The recommender uses a fixed feature order stored in models/config.json.

duration

tempo

loudness

key

mode

time_signature

danceability

energy

Feature scaling is learned using StandardScaler and applied consistently at recommendation time.

Recommendation approach
Offline step (build_index.py)

Loads all feature vectors from song_features

Fits a StandardScaler across the dataset

Optionally fits PCA to build compact embeddings

L2 normalizes embeddings for cosine similarity

Fits a NearestNeighbors retriever using cosine distance

Saves artifacts to models/ so the API can recommend without retraining

Online step (recommender.py)

User selects 1 to 20 track_ids

Fetches seed vectors from Postgres

Applies the same scaler and PCA as the index

Averages seed embeddings into a user taste vector

Queries kNN for nearest songs

Filters out seed songs and caps repeated artists

Returns the top 20 results with metadata from songs

API Overview

OpenAPI (Swagger UI)

http://localhost:8000/docs

Core endpoints you can use right away

Health

GET /health basic health check

Search

GET /search?q=jack&limit=20

Returns matching songs from the catalog

Includes track_id so the frontend can select songs

Recommend

POST /recommend

track_ids list of selected songs (1 to 20)

k number of recommendations, default 20

per_artist_cap max number of recommendations per artist, default 2

Example request body

{
  "track_ids": ["TRBIAHA128F42A4C9B", "TRBIACJ128F93087A9"],
  "k": 20,
  "per_artist_cap": 2
}
Running with Docker
1) Download dependencies (local)

Python 3.12+ installed and available
To install Python dependencies

pip install -r requirements.txt

2) Build + start services (Docker)

From the repo root

docker compose up -d --build

API is available at

http://localhost:8000/docs

Local environment variables

You will typically run extract.py, ingest.py, and build_index.py from your host machine. For that, .env should point to localhost.

Example .env

DATABASE_URL=postgresql+psycopg2://songrec:songrec_pw@localhost:5432/songrec

Inside Docker Compose the API uses db as the hostname, and that is set in docker-compose.yml.

Building the catalog and index manually

These steps are run manually when you refresh data or reset the DB.

Step A) Extract MSD to CSV

Reads MSD .h5 files and writes CSVs into data/.

python src/extract.py

Outputs

data/tracks.csv

data/features.csv

Step B) Ingest CSV into Postgres

Runs schema.sql to create tables if needed, then inserts and upserts data.

python src/ingest.py
Step C) Build ML index artifacts

Fits scaler, optional PCA, and kNN retrieval then saves artifacts into models/.

python src/build_index.py

After this, recommendations work immediately using the saved artifacts.

Quick verification

Check DB row counts

docker compose exec -T db psql -U songrec -d songrec -c "SELECT COUNT(*) FROM songs; SELECT COUNT(*) FROM song_features;"

Test search

curl "http://localhost:8000/search?q=jack&limit=5"

Test recommend

curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d "{\"track_ids\":[\"TRBIAHA128F42A4C9B\"],\"k\":20,\"per_artist_cap\":2}"