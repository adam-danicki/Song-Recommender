import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# ----- Configuration ----- #
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable not set.")

ENGINE = create_engine(DATABASE_URL, future=True, pool_pre_ping=True)

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "src" / "schema.sql"

TRACKS_PATH = ROOT / "data" / "tracks.csv"
FEATURES_PATH = ROOT / "data" / "features.csv"

RESET_TABLES = True
CHUNK_SIZE = 5000

VECTOR_COLS = [
    "duration",
    "tempo",
    "loudness",
    "key",
    "mode",
    "time_signature",
    "danceability",
    "energy",
]


"""
Convert 'a|b|c' -> ['a','b','c'] ; blanks -> []
"""
def pipe_to_list(val):
    if val is None:
        return []
    s = str(val).strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]


"""
Execute schema.sql (idempotent) inside an open transaction.
"""
def execute_sql(conn, schema_file: Path):
    if not schema_file.exists():
        raise RuntimeError(f"Schema file not found at: {schema_file}")
    
    sql = schema_file.read_text(encoding="utf-8")

    statements = [stmt.strip() for stmt in sql.split(";") if stmt.strip()]

    for stmt in statements:
        conn.execute(text(stmt))


def chunker(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i + size]

def main():
    # 1) load CSVs
    tracks = pd.read_csv(TRACKS_PATH)
    features = pd.read_csv(FEATURES_PATH)

    # 2) Basic clean and typing
    tracks["track_id"] = tracks["track_id"].astype(str).str.strip()
    tracks["title"] = tracks["title"].astype(str).str.strip()
    tracks["artist"] = tracks["artist"].astype(str).str.strip()

    # nullable int year
    tracks["year"] = pd.to_numeric(tracks.get("year"), errors="coerce")
    tracks["year"] = tracks["year"].where(tracks["year"].fillna(0) > 0, pd.NA).astype("Int64")
    
    # optional columns
    if "release" not in tracks.columns:
        tracks["release"] = pd.NA
    if "genre" not in tracks.columns:
        tracks["genre"] = pd.NA
    if "artist_terms_top" not in tracks.columns:
        tracks["artist_terms_top"] = ""
    if "artist_mbtags_top" not in tracks.columns:
        tracks["artist_mbtags_top"] = ""

    # Convert pipe-delimited -> Postgres TEXT[]
    tracks["artist_terms_top"] = tracks["artist_terms_top"].apply(pipe_to_list)
    tracks["artist_mbtags_top"] = tracks["artist_mbtags_top"].apply(pipe_to_list)

    features["track_id"] = features["track_id"].astype(str).str.strip()

    # force numeric for vector columns
    for col in VECTOR_COLS:
        features[col] = pd.to_numeric(features.get(col), errors="coerce")

    if "year" in features.columns:
        features["year"] = pd.to_numeric(features["year"], errors="coerce")

    # 3) Merge to keep tracks that have both metadata and vector features
    merged = tracks.merge(features[["track_id"] + VECTOR_COLS], on="track_id", how="inner")
    merged = merged.dropna(subset=["track_id", "title", "artist"])
    merged = merged.dropna(subset=VECTOR_COLS)
    merged["features"] = merged[VECTOR_COLS].astype(float).values.tolist()
    merged = merged.drop_duplicates(subset=["track_id"])

    print(f"tracks.csv rows:   {len(tracks)}")
    print(f"features.csv rows: {len(features)}")
    print(f"merged rows:       {len(merged)} (will be inserted)")

    # 4) Build rows for DB
    song_rows = merged[
        ["track_id", "title", "artist", "year", "release", "genre", "artist_terms_top", "artist_mbtags_top"]
    ].to_dict(orient="records")

    feat_rows = merged[["track_id", "features"]].to_dict(orient="records")

    # 5) Run schema and insert data
    with ENGINE.begin() as conn:
        execute_sql(conn, SCHEMA_PATH)

        if RESET_TABLES:
            conn.execute(text("TRUNCATE songs CASCADE;"))

        # Upsert SONGS
        for chunk in chunker(song_rows, CHUNK_SIZE):
            stmt = text("""
                INSERT INTO songs (
                    track_id, title, artist, year, release, genre, artist_terms_top, artist_mbtags_top
                )
                VALUES (
                    :track_id, :title, :artist, :year, :release, :genre, :artist_terms_top, :artist_mbtags_top
                )
                ON CONFLICT (track_id) DO UPDATE SET
                    title = EXCLUDED.title,
                    artist = EXCLUDED.artist,
                    year = EXCLUDED.year,
                    release = EXCLUDED.release,
                    genre = EXCLUDED.genre,
                    artist_terms_top = EXCLUDED.artist_terms_top,
                    artist_mbtags_top = EXCLUDED.artist_mbtags_top;
            """)
            conn.execute(stmt, chunk)
        
        # Upsert features vector
        for chunk in chunker(feat_rows, CHUNK_SIZE):
            conn.execute(
                text("""
                    INSERT INTO song_features (track_id, features)
                    VALUES (:track_id, :features)
                    ON CONFLICT (track_id) DO UPDATE
                    SET features = EXCLUDED.features;
                """),
                chunk,
            )
        
    print("Data ingestion complete.")

if __name__ == "__main__":
    main()

            