from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset 

class MovieLensSequenceDataset(Dataset):
    def __init__(self, processed_dir):
        p = Path(processed_dir)

        # records is an object array of length num_users
        # Each element is a dict of one user.
        self.records = np.load(p / "sequences.npy", allow_pickle=True)

        # metadata contains num_movies, num_genres, max_len, etc.
        with open(p / "metadata.json", "r") as f:
            self.metadata = json.load(f)

    def __len__(self):
        # number of user sequences
        return len(self.records)
    
    def __getitem__(self, idx):
        # one user record.
        r = self.records[idx]

        # return variable-length tensors
        # let this user's length be T_i.
        return {
            "user_id": torch.tensor(r["user_id"], dtype=torch.long),
            "movie_id": torch.tensor(r["movie_id"], dtype=torch.long),
            "rating": torch.tensor(r["rating"], dtype=torch.float32),
            "time_gap_hours": torch.tensor(r["gaps"], dtype=torch.float32),
            "genres": torch.tensor(r["genres"], dtype=torch.float32),
        }

def collate_sequences(batch):
    # batch is a list of examples from __getitem__.
    # Length of batch = B.
    B = len(batch)

    # T is the longest sequence length in this batch.
    # This can change from batch to batch.
    T = max(x["movie_id"].shape[0] for x in batch)

    # G is number of genres.
    G = batch[0]["genres"].shape[-1]

    # Allocate padded tensors.
    # Padding value for movie_id is 0, matching padding_idx=0 in the embedding.
    movie_id = torch.zeros(B, T, dtype=torch.long)      # [B, T]
    rating = torch.zeros(B, T)                          # [B, T]
    time_gap = torch.zeros(B, T)                        # [B, T]
    genres = torch.zeros(B, T, G)                       # [B, T, G]

    # valid_mask tells us which positions are real events.
    # True  = real event
    # False = padding position
    valid_mask = torch.zeros(B, T, dtype=torch.bool)    # [B, T]

    # user_id has one scalar per sequence.
    user_id = torch.stack([x["user_id"] for x in batch]) # [B]

    for i, x in enumerate(batch):
        # Actual length for user i.
        n = x["movie_id"].shape[0]

        # Copy the real sequence into the first n positions.
        movie_id[i, :n] = x["movie_id"]
        rating[i, :n] = x["rating"]
        time_gap[i, :n] = x["time_gap_hours"]
        genres[i, :n] = x["genres"]

        # Mark real positions.
        valid_mask[i, :n] = True

    return {
        "user_id": user_id,                 # [B]
        "movie_id": movie_id,               # [B, T]
        "rating": rating,                   # [B, T]
        "time_gap_hours": time_gap,         # [B, T]
        "genres": genres,                   # [B, T, G]
        "valid_mask": valid_mask,           # [B, T]
    }