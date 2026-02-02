CREATE TABLE IF NOT EXISTS songs (
    track_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    year INTEGER,
    release TEXT,
    genre TEXT,
    artist_terms_top TEXT[],
    artist_mbtags_top TEXT[]
);

CREATE TABLE IF NOT EXISTS song_features (
    track_id TEXT PRIMARY KEY REFERENCES songs(track_id) ON DELETE CASCADE,
    features DOUBLE PRECISION[] NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_songs_title_lower  ON songs (lower(title));
CREATE INDEX IF NOT EXISTS idx_songs_artist_lower ON songs (lower(artist));

-- optional but great for tag search later:
CREATE INDEX IF NOT EXISTS idx_songs_terms_gin  ON songs USING GIN (artist_terms_top);
CREATE INDEX IF NOT EXISTS idx_songs_mbtags_gin ON songs USING GIN (artist_mbtags_top);
