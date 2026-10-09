
from pathlib import Path
import pandas as pd

# Repo root and raw MovieLens directory.
root = Path(__file__).resolve().parents[1]
d = root / "data" / "raw" / "ml-1m"

# ratings shape: [num_ratings, 4]
# columns: user_id, movie_id, rating, timestamp
ratings = pd.read_csv(
    d / "ratings.dat",
    sep="::",
    engine="python",
    names=["user_id", "movie_id", "rating", "timestamp"],
)

# movies shape: [num_movies, 3]
# genres is a string like "Comedy|Romance".
movies = pd.read_csv(
    d / "movies.dat",
    sep="::",
    engine="python",
    encoding="latin-1",
    names=["movie_id", "title", "genres"],
)

# users shape: [num_users, 5]
users = pd.read_csv(
    d / "users.dat",
    sep="::",
    engine="python",
    names=["user_id", "gender", "age", "occupation", "zip_code"],
)

print("ratings:", ratings.shape)
print("movies :", movies.shape)
print("users  :", users.shape)
print("unique users :", ratings.user_id.nunique())
print("unique movies:", ratings.movie_id.nunique())

# Convert Unix timestamps to human-readable timestamps for inspection.
ts = pd.to_datetime(ratings.timestamp, unit="s")
print("time range:", ts.min(), "->", ts.max())

# Count how many interactions each user has.
lengths = ratings.groupby("user_id").size()
print(lengths.quantile([0, .25, .5, .75, .9, .99, 1.0]))
