from pathlib import Path
import csv
import h5py
import time

DATA_ROOT = Path("data/msd/MillionSongSubset")
OUT_TRACKS = Path("data/tracks.csv")
OUT_FEATURES = Path("data/features.csv")

TOPK_TERMS = 5
TOPK_MBTAGS = 5

TRACKS_FIELDS = [
    "track_id",
    "title",
    "artist",
    "year",
    "release",
    "genre",             
    "artist_terms_top",   
    "artist_mbtags_top",
    ]
FEATURE_FIELDS = [
    "track_id",
    "duration",
    "tempo",
    "loudness",
    "key",
    "mode",
    "time_signature",
    "danceability",
    "energy",
    "key_confidence",
    "mode_confidence",
    "time_signature_confidence",
    "end_of_fade_in",
    "start_of_fade_out",
    "song_hotttnesss",
    "artist_hotttnesss",
    "artist_familiarity",
    "year",
]

"""
Convert bytes -> str safely.
"""
def decode_str(byte_string) -> str:
    if byte_string is None:
        return ""
    if isinstance(byte_string, bytes):
        return byte_string.decode('utf-8', errors='ignore').strip()
    return str(byte_string).strip()


"""
Convert numeric-ish values to int safely. If missing/unusable, return "".
"""
def to_int(value):
    try:
        return int(float(value))
    except Exception:
        return ""


"""
Convert numeric values to float safely. If missing/unusable, return "".
"""
def to_float(value):
    try:
        v = float(value)
        if v != v:
            return ""
        return v
    except Exception:
        return ""


"""
Attempt to read songs[0][field]. If anything is missing, return None.
"""
def get_field(f: h5py.File, group: str, field: str):
    try:
        return f[group]["songs"][0][field]
    except Exception:
        return None
    

"""
Read an array dataset from <group>/<name>.
"""
def get_array(f: h5py.File, group: str, name: str):
    try:
        return f[group][name][:]
    except Exception:
        return None


"""
Convert an HDF5 array of strings/bytes into Python list[str].
"""
def decode_str_list(arr):
    if arr is None:
        return []
    out = []
    try:
        for x in arr:
            s = decode_str(x)
            if s:
                out.append(s)
    except Exception:
        return []
    return out


"""
Pick the top-k unique string items by descending numeric "weight".
Used for Echo Nest artist terms:
- items  = metadata/artist_terms
- weights = metadata/artist_terms_weight (or fallback to artist_terms_freq)
"""
def topk_by_weight(items, weights, k: int):
    if items is None or weights is None:
        return []
    pairs = []
    try:
        for it, w in zip(items, weights):
            name = decode_str(it)
            ww = to_float(w)
            if name and ww != "":
                pairs.append((name, ww))
    except Exception:
        return []

    pairs.sort(key=lambda x: x[1], reverse=True)

    result = []
    seen = set()
    for name, _ in pairs:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
        if len(result) >= k:
            break
    return result


"""
Pick the top-k unique string items by descending numeric "count".
Used for MusicBrainz tags:
- items  = musicbrainz/artist_mbtags
- counts = musicbrainz/artist_mbtags_count
"""
def topk_by_count(items, counts, k: int):
    if items is None or counts is None:
        return []
    pairs = []
    try:
        for it, c in zip(items, counts):
            name = decode_str(it)
            cc = to_float(c)
            if name and cc != "":
                pairs.append((name, cc))
    except Exception:
        return []

    # Highest counts first
    pairs.sort(key=lambda x: x[1], reverse=True)

    result = []
    seen = set()
    for name, _ in pairs:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(name)
        if len(result) >= k:
            break
    return result


"""
Join a list of strings into a single pipe-delimited string for CSV storage.
"""
def join_pipe(items):
    return "|".join(items) if items else ""


"""
Read one .h5 file and return:
 - tracks_row dict
 - features_row dict
If track_id is missing, return (None, None) to signal skip.
"""
def extract_one(h5_path: Path):
    with h5py.File(h5_path, "r") as f:
        # Stable ID
        track_id = decode_str(get_field(f, "analysis", "track_id") or "")
        if not track_id:
            return None, None

        # Display metadata
        title = decode_str(get_field(f, "metadata", "title") or "")
        artist = decode_str(get_field(f, "metadata", "artist_name") or "")
        release = decode_str(get_field(f, "metadata", "release") or "")

        year_raw = get_field(f, "musicbrainz", "year")
        if year_raw is None:
            year_raw = get_field(f, "metadata", "year")

        year = to_int(year_raw)

        if year == 0:
            year = ""

        # Tag/term data (genre-ish)
        artist_terms = get_array(f, "metadata", "artist_terms")
        artist_terms_weight = get_array(f, "metadata", "artist_terms_weight")

        mbtags = get_array(f, "musicbrainz", "artist_mbtags")
        mbtags_count = get_array(f, "musicbrainz", "artist_mbtags_count")

        terms_top = topk_by_weight(artist_terms, artist_terms_weight, TOPK_TERMS)
        mbtags_top = topk_by_count(mbtags, mbtags_count, TOPK_MBTAGS)

        # Pick a simple "genre" proxy for now (top EchoNest term else top MB tag)
        genre = terms_top[0] if terms_top else (mbtags_top[0] if mbtags_top else "")

        # Numeric features
        duration = to_float(get_field(f, "analysis", "duration"))
        tempo = to_float(get_field(f, "analysis", "tempo"))
        loudness = to_float(get_field(f, "analysis", "loudness"))

        key = to_int(get_field(f, "analysis", "key"))
        mode = to_int(get_field(f, "analysis", "mode"))
        time_signature = to_int(get_field(f, "analysis", "time_signature"))

        danceability = to_float(get_field(f, "analysis", "danceability"))
        energy = to_float(get_field(f, "analysis", "energy"))

        key_conf = to_float(get_field(f, "analysis", "key_confidence"))
        mode_conf = to_float(get_field(f, "analysis", "mode_confidence"))
        ts_conf = to_float(get_field(f, "analysis", "time_signature_confidence"))

        end_fade_in = to_float(get_field(f, "analysis", "end_of_fade_in"))
        start_fade_out = to_float(get_field(f, "analysis", "start_of_fade_out"))

        song_hot = to_float(get_field(f, "metadata", "song_hotttnesss"))
        artist_hot = to_float(get_field(f, "metadata", "artist_hotttnesss"))
        artist_fam = to_float(get_field(f, "metadata", "artist_familiarity"))

        tracks_row = {
            "track_id": track_id,
            "title": title,
            "artist": artist,
            "year": year,
            "release": release,
            "genre": genre,
            "artist_terms_top": join_pipe(terms_top),
            "artist_mbtags_top": join_pipe(mbtags_top),
        }

        features_row = {
            "track_id": track_id,
            "duration": duration,
            "tempo": tempo,
            "loudness": loudness,
            "key": key,
            "mode": mode,
            "time_signature": time_signature,
            "danceability": danceability,
            "energy": energy,
            "key_confidence": key_conf,
            "mode_confidence": mode_conf,
            "time_signature_confidence": ts_conf,
            "end_of_fade_in": end_fade_in,
            "start_of_fade_out": start_fade_out,
            "song_hotttnesss": song_hot,
            "artist_hotttnesss": artist_hot,
            "artist_familiarity": artist_fam,
            "year": year,
        }

        return tracks_row, features_row


def main():
    OUT_TRACKS.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_ROOT.exists():
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")

    temp_tracks = OUT_TRACKS.with_suffix(".tmp.tracks.csv")
    temp_features = OUT_FEATURES.with_suffix(".tmp.features.csv")

    processed = 0
    skipped = 0
    started = time.time()
    seen_ids = set()

    with temp_tracks.open("w", newline="", encoding="utf-8") as tracks_csvfile, \
         temp_features.open("w", newline="", encoding="utf-8") as features_csvfile:

        tracks_writer = csv.DictWriter(tracks_csvfile, fieldnames=TRACKS_FIELDS)
        features_writer = csv.DictWriter(features_csvfile, fieldnames=FEATURE_FIELDS)
        tracks_writer.writeheader()
        features_writer.writeheader()

        for h5_path in DATA_ROOT.rglob("*.h5"):
            try:
                tracks_row, feats_row = extract_one(h5_path)
                if tracks_row is None:
                    skipped += 1
                    continue

                tid = tracks_row["track_id"]
                if tid in seen_ids:
                    skipped += 1
                    continue
                seen_ids.add(tid)

                tracks_writer.writerow(tracks_row)
                features_writer.writerow(feats_row)
                processed += 1

                if processed % 500 == 0:
                    elapsed = time.time() - started
                    print(f"Processed {processed} tracks | skipped {skipped} | elapsed {elapsed:.1f}s")

            except Exception:
                skipped += 1
                continue

    if OUT_TRACKS.exists():
        OUT_TRACKS.unlink()
    if OUT_FEATURES.exists():
        OUT_FEATURES.unlink()

    temp_tracks.replace(OUT_TRACKS)
    temp_features.replace(OUT_FEATURES)

    elapsed = time.time() - started
    print(f"Done. Wrote {processed} tracks. Skipped {skipped}. Total time {elapsed:.1f}s")
    print(f"Tracks:   {OUT_TRACKS}")
    print(f"Features: {OUT_FEATURES}")


if __name__ == "__main__":
    main()