from pathlib import Path
import json
import numpy as np
import pandas as pd

# Maximum number of interactions retained per user
# if a user has more than 200 interactions, keep the most recent 200.

MAX_LEN = 200

root = Path(__file__).resolve().parents[1]
raw = root / "data" / "raw" / "ml-1m"
out = root / "data" / "processed"

out.mkdir(parents=True, exist_ok=True)

# Load raw ratings
# ratings rows: [num_ratings, 4]

ratings = pd.read_csv(
    raw / "ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

# Load movie metadata.
# movies rows: [num_movies, 3]

movies = pd.read_csv(
    raw / "movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"],
)

# Build genre vocabulary from strings like "Action|Comedy".
# Example genre_vocab: ["Action", "Adventure", ...]

genre_vocab = sorted({g for s in movies.genres for g in s.split("|")})

# Map genre string to integer index.
g2i = {g: i for i, g in enumerate(genre_vocab)}

# Build movie_id -> multi-hot genre vector.
# Each vector shape: [G]
# G = number of genres

movie_genres = {}

for r in movies.itertuples(index=False):
    v = np.zeros(len(genre_vocab), dtype=np.float32)
    for g in r.genres.split("|"):
        v[g2i[g]] = 1.0
    movie_genres[r.movie_id] = v

# sort all interactions by user, then by time.
# this gives chronological order within each user.
ratings = ratings.sort_values(by=["user_id", "timestamp"])

records = []

# Each group g is one user's full interaction history
for user_id, g in ratings.groupby("user_id"):
    # keep most recent MAX_LEN interactions
    g = g.tail(MAX_LEN)

    # shape: [T_i]
    # T_i <= MAX_LEN and can differ per user
    movie_id = g.movie_id.to_numpy(np.int64)
    rating = g.rating.to_numpy(np.float32)
    timestamp = g.timestamp.to_numpy(np.int64)

    # time gap in hours between consecutive events
    # shape [T_i]
    # First event has no previous event, so gap = 0,
    gaps = np.zeros(len(g), dtype=np.float32)
    if len(g) > 1:
        gaps[1:] = np.diff(timestamp) / 3600.0

    # stack genre vectors for all movies in the user's sequence
    # Shape: [T_i, G]
    genres = np.stack([movie_genres[int(m)] for m in movie_id]).astype(np.float32)
    
    records.append({
        "user_id": int(user_id),
        "movie_id": movie_id,            #[T_i]
        "rating": rating,                #[T_i]
        "timestamp": timestamp,          #[T_i]
        "gaps": gaps,                    #[T_i]
        "genres": genres,                #[T_i, G]
    })

# Save variable-length Python objects.
# Object array is used because each user can have a different T_i.
np.save(out / "sequences.npy", np.array(records, dtype=object), allow_pickle=True)

metadata = {
    "max_len": MAX_LEN,
    "num_users": len(records),
    "num_genres": len(genre_vocab),
    "genre_vocab": genre_vocab,
    # +1 allows movie_id to be used directly as an embedding index.
    # Index 0 is reserved for padding.
    "num_movies": int(ratings.movie_id.max()) + 1,
}

with open(out / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

lengths = np.array([len(r["movie_id"]) for r in records])
print("users:", len(records))
print("length min/median/p90/max:", lengths.min(), np.median(lengths), np.quantile(lengths, .9), lengths.max())
print(metadata)