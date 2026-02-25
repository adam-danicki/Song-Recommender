from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text

from src.db import ENGINE
from src.recommender import recommend

app  = FastAPI(title="Song Recommender API")

# Allow CORS for local development (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request model for /recommend endpoint
class RecommendRequest(BaseModel):
    track_ids: List[str]
    k: Optional[int] = 20
    per_artist_cap: Optional[int] = 2

# Response model for /health endpoint
@app.get("/health")
def health():
    return {"status": "ok"}

"""
Endpoint to recommend similar tracks based on seed track_ids.
Request body should be JSON with keys:
    - track_ids: list of seed track_ids (required)
    - k: number of recommendations to return (optional, default=20)
    - per_artist_cap: max number of recommendations per artist (optional, default=2)
Response is a list of recommended tracks with metadata.
"""
def search(q: str, limit: int = 20):
    q = (q or "").strip()
    if not q:
        return []
    
    # case-insensitive partial match on title or artist
    pattern = f"%{q.lower()}%"

    with ENGINE.connect() as conn:
        rows = conn.execute(
            text("""
                 SELECT track_id, title, artist, year, release, genre
                 FROM songs
                 Where lower(title) Like :pattern OR lower(artist) LIKE :pattern
                 ORDER BY artist, title
                 LIMIT :limit
            """),
            {"pattern": pattern, "limit": limit},
        ).fetchall()
    
    return [
        {
            "track_id": str(tid),
            "title": title,
            "artist": artist,
            "year": year,
            "release": release,
            "genre": genre,
        }
        for tid, title, artist, year, release, genre in rows
    ]

"""
Endpoint to recommend similar tracks based on seed track_ids.
Request body should be JSON with keys:
    - track_ids: list of seed track_ids (required)
    - k: number of recommendations to return (optional, default=20)
    - per_artist_cap: max number of recommendations per artist (optional, default=2)
Response is a list of recommended tracks with metadata.
"""
@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    if not req.track_ids:
        raise HTTPException(status_code=400, detail="track_ids is required and cannot be empty")
    try:
        recs = recommend(
            req.track_ids,
            k = req.k,
            per_artist_cap = req.per_artist_cap,
        )
        return {"seed_count": min(len(req.track_ids), 20), "results": recs}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
