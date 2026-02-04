import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import joblib

load_dotenv()

# ---------- Config ----------
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set")

ENGINE = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

ROOT  = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

USE_PCA = True
PCA_COMPONENTS = 64
KNN_METRIC = 'cosine'
KNN_ALGORITHM = 'brute'


"""
Pull (track_id, features[]) from Postgres.
Assumes song_features.features is DOUBLE PRECISION[].
"""
def load_vectors():
    with ENGINE.connect() as conn:
        rows = conn.execute(text("SELECT track_id, features FROM song_features"))
    
    track_ids = []
    X = []

    for tid, features in rows:
        if tid is None or features is None:
            continue
        track_ids.append(str(tid))
        X.append(np.array(features, dtype=np.float64))
    
    X = np.vstack(X)
    return np.array(track_ids), X

def main():
    track_ids, X = load_vectors()
    n, d = X.shape
    print(f"Loaded {n} vectors of dimension {d}")

    # 1) Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2) PCA
    pca = None
    X_emb = X_scaled

    if USE_PCA:
        n_components = min(PCA_COMPONENTS, X_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        X_emb = pca.fit_transform(X_scaled)
        print(f"PCA: {d} -> {X_emb.shape[1]} dims")
    
    # 3) L2 Normalization so cosine behaves well
    X_emb = normalize(X_emb, norm='l2')

    # 4) Fit kNN retriever
    nn = NearestNeighbors(
        metric=KNN_METRIC,
        algorithm=KNN_ALGORITHM,
    )
    nn.fit(X_emb)

    # 5) Save models
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    if pca is not None:
        joblib.dump(pca, MODELS_DIR / "pca.pkl")
    joblib.dump(nn, MODELS_DIR / "knn.pkl")

    np.save(MODELS_DIR / "track_ids.npy", track_ids)
    np.save(MODELS_DIR / "embeddings.npy", X_emb)

    config = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "n_songs": int(n),
        "raw_dim": int(d),
        "use_pca": bool(USE_PCA),
        "pca_components_requested": int(PCA_COMPONENTS),
        "final_dim": int(X_emb.shape[1]),
        "knn_metric": KNN_METRIC,
        "knn_algorithm": KNN_ALGORITHM,
        "vector_cols_order": [
            "duration", "tempo", "loudness", "key", "mode", "time_signature", "danceability", "energy"
        ],
    }
    (MODELS_DIR / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Saved model artifacts to:", MODELS_DIR)

if __name__ == "__main__":
    main()