from pathlib import Path

import numpy as np
import joblib
from sqlalchemy import text
from sklearn.preprocessing import normalize

from src.db import ENGINE

ROOT  = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

ARTIFACTS = None

"""
Load scaler, PCA, KNN, and track_ids artifacts from disk.
"""
def load_artifacts():
    global ARTIFACTS
    if ARTIFACTS is not None:
        return ARTIFACTS
    
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")

    pca_path = MODELS_DIR / "pca.pkl"
    pca = joblib.load(pca_path) if pca_path.exists() else None

    knn = joblib.load(MODELS_DIR / "knn.pkl")
    index_track_ids = np.load(MODELS_DIR / "track_ids.npy", allow_pickle=True)

    ARTIFACTS = (scaler, pca, knn, index_track_ids)

    return ARTIFACTS

"""
Given a list of track_ids, fetch their feature vectors from Postgres 
and apply the same transformations as the index (scaling, PCA).
"""
def fetch_vectors(track_ids):
    ids = [str(t).strip() for t in track_ids if str(t).strip()]
    if not ids:
        return {}
    
    with ENGINE.connect() as conn:
        rows = conn.execute(
            text("""
                 SELECT track_id, features
                 FROM song_features
                 WHERE track_id = ANY(CAST(:ids AS TEXT[]))
                 """),
            {"ids": ids},
        ).fetchall()

    out = {}
    for tid, features in rows:
        if tid is None or features is None:
            continue
        out[str(tid)] = np.array(features, dtype=np.float64)
    return out

"""
Given a list of track_ids, fetch their metadata from Postgres.
"""
def fetch_metadata(track_ids):
    ids = [str(t).strip() for t in track_ids if str(t).strip()]
    if not ids:
        return []

    with ENGINE.connect() as conn:
        rows = conn.execute(
            text("""
                 SELECT track_id, title, artist, year, release, genre
                 FROM songs
                 WHERE track_id = ANY(CAST(:ids AS TEXT[]))
                """),
            {"ids": ids},
        ).fetchall()

    by_id = {
        str(tid): {
            "track_id": str(tid),
            "title": title,
            "artist": artist,
            "year": year,
            "release": release,
            "genre": genre,
        }
        for tid, title, artist, year, release, genre in rows
    }

    return [by_id[tid] for tid in ids if tid in by_id]

"""
Given a track_id, return the top k most similar tracks based on the KNN index.
"""
def embed(X, scaler, pca):
    X_scaled = scaler.transform(X)
    X_emb = pca.transform(X_scaled) if pca is not None else X_scaled
    return normalize(X_emb, norm="l2")

"""
Recommend similar tracks given a list of seed track_ids.
    - seed_track_ids: list of track_ids to base recommendations on
    - k: number of recommendations to return
    - per_artist_cap: max number of recommendations per artist (None for no cap)
Returns a list of dicts with keys: track_id, title, artist, year, release, genre
"""
def recommend(seed_track_ids, k=20, per_artist_cap=2):
    if not seed_track_ids:
        raise ValueError("At least one seed track_id is required")
    
    seed_track_ids = [str(t).strip() for t in seed_track_ids if str(t).strip()]

    scaler, pca, knn, index_track_ids = load_artifacts()

    seed_map = fetch_vectors(seed_track_ids)
    found = list(seed_map.keys())
    if not found:
        raise ValueError("None of the provided seed track_ids exist in song_features")
    
    X_seed = np.vstack([seed_map[tid] for tid in found])
    E_seed = embed(X_seed, scaler, pca)

    user_vec = E_seed.mean(axis=0, keepdims=True)
    user_vec = normalize(user_vec, norm="l2")

    buffer = max(200, k * 15)
    n_neighbors = min(len(index_track_ids), k + buffer)

    distances, indices = knn.kneighbors(user_vec, n_neighbors=n_neighbors)
    candidate_ids = [str(index_track_ids[i]) for i in indices[0]]

    seed_set = set(found)
    candidate_ids = [tid for tid in candidate_ids if tid not in seed_set]

    candidates = fetch_metadata(candidate_ids)

    if per_artist_cap is None:
        return candidates[:k]

    results = []
    artist_counts = {}
    for s in candidates:
        a = s["artist"]
        artist_counts[a] = artist_counts.get(a, 0) + 1
        if artist_counts[a] <= per_artist_cap:
            results.append(s)
        if len(results) >= k:
            break

    return results

"""
Quick test
"""
# if __name__ == "__main__":
#     recs = recommend(["TRBIAHA128F42A4C9B"], k=20)
#     for r in recs[:10]:
#         print(r["artist"], "-", r["title"])
        
                 



    
    