from pathlib import Path
import json
import sys

import torch
from torch.utils.data import DataLoader

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.data.movielens import MovieLensSequenceDataset, collate_sequences
from src.models.event_encoder import EventEncoder

processed = root / "data" / "processed"

# Load metadata needed to build the model.
with open(processed / "metadata.json") as f:
    meta = json.load(f)

# Dataset length = number of users.
dataset = MovieLensSequenceDataset(processed)

# DataLoader creates batches of user sequences.
# batch size = 8
# collate_sequences pads variable-length squences to [B, T]
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_sequences,
)

# Pull one real batch.
batch = next(iter(loader))

# Portable device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")



# Build model.
# meta["num_movies"] controls embedding table size.
# meta["num_genres"] controls genre projection input size.
# d_model=64 means event_repr last dimension is 64.
d_model = 64
model = EventEncoder(meta["num_movies"], meta["num_genres"], d_model).to(device)

# Move batch tensors used by model to device.
movie = batch["movie_id"].to(device)              # [B, T]
rating = batch["rating"].to(device)               # [B, T]
gap = batch["time_gap_hours"].to(device)          # [B, T]
genres = batch["genres"].to(device)               # [B, T, G]


# Forward pass without gradient tracking.
with torch.no_grad():
    z = model(movie, rating, gap, genres)          # [B, T, 64]

print("device:", device)

# Print all batch shapes.
for k, v in batch.items():
    print(k, tuple(v.shape))

print("event_repr", tuple(z.shape))
print("finite:", torch.isfinite(z).all().item())
print("mean/std:", z.mean().item(), z.std().item())


with torch.no_grad():
    # Full event representation using all fields.
    full = model(movie, rating, gap, genres)                         # [B, T, D]

    # Remove genre information by replacing genres with zeros.
    no_genre = model(movie, rating, gap, torch.zeros_like(genres))   # [B, T, D]

    # Remove time-gap information by replacing gap with zeros.
    no_time = model(movie, rating, torch.zeros_like(gap), genres)    # [B, T, D]

# Compute root mean squared difference between representations.
# If these are near zero, the feature path may not be connected.
print("delta_no_genre:", (full - no_genre).pow(2).mean().sqrt().item())
print("delta_no_time :", (full - no_time).pow(2).mean().sqrt().item())