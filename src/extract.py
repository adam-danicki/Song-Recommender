from pathlib import Path
import csv
import h5py
import time

DATA_ROOT = Path("data/msd/MillionSongSubset")
OUT_TRACKS = Path("data/tracks.csv")
OUT_FEATURES = Path("data/features.csv")

TRACKS_FIELDS = ['track_id', 'title', 'artist', 'year']
FEATURE_FIELDS = ["track_id", "duration", "tempo", "loudness", "key", "mode", "time_signature", "year"]


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
Returns the first non-None field name value.
"""
def get_first_available(f: h5py.File, group: str, fields: list[str]):
    for field in fields:
        value = get_field(f, group, field)
        if value is not None:
            return value
    return None


"""
Read one .h5 file and return:
 - tracks_row dict
 - features_row dict
If track_id is missing, return (None, None) to signal skip.
"""
def extract_one(h5_path: Path):
    with h5py.File(h5_path, 'r') as f:
        # Stable ID
        track_id = decode_str(get_first_available(f, "analysis", ["track_id"]) or "")
        if not track_id:
            return None, None
        
        # Track Info
        title = decode_str(get_first_available(f, "metadata", ["title"]) or "")
        artist = decode_str(get_first_available(f, "metadata", ["artist_name"]) or "")
        year_raw = get_first_available(f, "metadata", ["year"])

        year = to_int(year_raw)

        # Numeric Features
        duration = to_float(get_first_available(f, "analysis", ["duration"]))
        tempo = to_float(get_first_available(f, "analysis", ["tempo"]))
        loudness = to_float(get_first_available(f, "analysis", ["loudness"]))
        key = to_float(get_first_available(f, "analysis", ["key"]))
        mode = to_float(get_first_available(f, "analysis", ["mode"]))
        time_signature = to_float(get_first_available(f, "analysis", ["time_signature"]))

        tracks_row = {
            "track_id": track_id,
            "title": title,
            "artist": artist,
            "year": year,
        }

        features_row = {
            "track_id": track_id,
            "duration": duration,
            "tempo": tempo,
            "loudness": loudness,
            "key": key,
            "mode": mode,
            "time_signature": time_signature,
            "year": year,
        }

        return tracks_row, features_row
    
def main():
    # Make sure output directory exists
    OUT_TRACKS.parent.mkdir(parents=True, exist_ok=True)

    if not DATA_ROOT.exists():
        raise RuntimeError(f"DATA_ROOT not found: {DATA_ROOT}")
    
    # Write to temp files first
    temp_tracks = OUT_TRACKS.with_suffix(".tmp.tracks.csv")
    temp_features = OUT_FEATURES.with_suffix(".tmp.features.csv")

    processed = 0
    skipped = 0
    started = time.time()

    seen_ids = set()

    with temp_tracks.open("w", newline='', encoding='utf-8') as tracks_csvfile, \
         temp_features.open("w", newline='', encoding='utf-8') as features_csvfile:
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
    
    # Delete old outputs if they exist, then rename temp -> final
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
        

        
    



