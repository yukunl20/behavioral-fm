import torch
from torch import nn

class EventEncoder(nn.Module):
    def __init__(self, num_movies, num_genres, d_model=64):
        super().__init__()

        # Split d_model across four feature groups.
        # If d_model=64, then d=16.
        d = d_model // 4

        # movie_id is categorical, so use an embedding table.
        # Input shape:  [B, T]
        # Output shape: [B, T, d]
        # padding_idx=0 makes movie embedding at index 0 stay zero.
        self.movie_emb = nn.Embedding(num_movies, d, padding_idx=0)

        # genres is multi-hot [B, T, G].
        # Linear projects from G genre indicators to d dimensions.
        # Output shape: [B, T, d]
        self.genre_proj = nn.Linear(num_genres, d)

        # rating is scalar [B, T].
        # Unsqueeze to [B, T, 1], normalize by 5, then project to [B, T, d].
        self.rating_proj = nn.Sequential(nn.Linear(1, d), nn.GELU())

        # time_gap_hours is scalar [B, T].
        # log1p compresses heavy-tailed gaps before projection.
        # Output shape: [B, T, d]
        self.time_proj = nn.Sequential(nn.Linear(1, d), nn.GELU())

        # Concatenate four [B, T, d] tensors into [B, T, 4d].
        # Then fuse into [B, T, d_model].
        self.fuse = nn.Sequential(
            nn.Linear(d * 4, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

    def forward(self, movie_id, rating, time_gap_hours, genres):
        # movie_id: [B, T] -> movie: [B, T, d]
        movie = self.movie_emb(movie_id)

        # genres: [B, T, G] -> genre: [B, T, d]
        genre = self.genre_proj(genres)

        # rating: [B, T] -> [B, T, 1] -> [B, T, d]
        rating = self.rating_proj((rating / 5.0).unsqueeze(-1))

        # time_gap_hours: [B, T] -> [B, T, 1] -> [B, T, d]
        time = self.time_proj(torch.log1p(time_gap_hours).unsqueeze(-1))

        # Concatenate on the feature dimension.
        # [B, T, d] x 4 -> [B, T, 4d]
        x = torch.cat([movie, genre, rating, time], dim=-1)

        # Fuse into event representation.
        # [B, T, 4d] -> [B, T, D]
        return self.fuse(x)