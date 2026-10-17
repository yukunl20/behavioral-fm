import torch
from torch import nn

from src.objectives.corruption import MASKED


class EventEncoder(nn.Module):
    """
    status_mode controls how the encoder represents masked / missing fields:
      ignore       - ignore field_status; the model only sees the corrupted raw value.
      status_embed - use projection(value) + learned status embedding.
      mask_token   - for MASKED fields, replace the field representation with a learned
                     field-specific mask token.

    Experiment mapping:
      zero   -> ignore       # zero-filled mask; encoder is not told it was masked
      status -> status_embed # zero-fill plus explicit status signal
      token  -> mask_token   # MAE/BERT-style learned mask token
      vime   -> ignore       # donor corruption; status must not reveal the corruption label
    """

    def __init__(
        self,
        num_movies,
        num_genres,
        d_model=64,
        status_mode="ignore",
    ):

        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by four")

        if status_mode not in {"ignore", "status_embed", "mask_token"}:
            raise ValueError(f"unknown status_mode: {status_mode}")

        self.status_mode = status_mode
        self.d_model = d_model              # Store event embedding width D
        self.d_field = d_model // 4         # Each of four field groups gets D/4 dimensions

        d = self.d_field
        self.movie_emb = nn.Embedding(      # Categorical embedding table for movie_id
            num_movies,                     # Number of rows in embedding table
            d,                              # Embedding width per movie
            padding_idx=0                   # movie_id = 0 is padding; embedding stays as zero
        )

        self.genre_proj = nn.Linear(        # Linear projection for multi-hot genre vector
            num_genres,                     # Input dimension G
            d                               # Output dimension d
        )

        self.rating_proj = nn.Sequential(   # Small projection network for scalar rating
            nn.Linear(1, d),                # Map one scalar to d dimensions
            nn.GELU()                       # Nonlinear activation
        )

        self.time_proj = nn.Sequential(     # Small projection network for scalar time gap
            nn.Linear(1, d),                # Map one scalar to d dimensions
            nn.GELU()                       # Nonlinear activation
        )

        if status_mode in {"status_embed", "mask_token"}:
            self.status_emd = nn.Embedding(  # Learn embeddings for observation status codes
                4,                           # observed / natural missing / unavailable / masked
                d,                           # Status vector has the same width as one field representation
                padding_idx=0                # OBSERVED=0 uses a fixed zero embedding
            )

        if status_mode == "mask_token":
            self.mask_tokens = nn.Parameter(      # Register mask tokens as trainable parameters
                torch.randn(4, d) * 0.02          # One small random token per field group
            )

        self.fuse = nn.Sequential(          # Fusion network after concatenating field vectors
            nn.Linear(d * 4, d_model),      # Combine four field vectors [4d] into [D]
            nn.LayerNorm(d_model),          # Normalize event embedding for stability
            nn.GELU()                       # Add nonlinearity
        )

    def _status_transform(self, x, status, field_index):  # Apply status logic to one projected field
        if self.status_mode == "ignore":
            return x

        if self.status_mode == "status_embed":
            return x + self.status_emd(status)

        # mask_token: still encode non-masked statuses additively, replace MASKED with token
        x = x + self.status_emd(status)
        is_masked = (status == MASKED).unsqueeze(-1)            # [B, T, 1]
        token = self.mask_tokens[field_index].view(1, 1, -1)    # This field's learned mask token
        return torch.where(is_masked, token, x)


    def forward(                      # Forward pass: raw fields -> event embeddings
        self,
        movie_id,                     # [B, T]
        rating,                       # [B, T]
        time_gap_hours,               # [B, T]
        genres,                       # [B, T, G]
        field_status=None,            # [B, T, 4] status code for the four field groups
        valid_mask=None,              # [B, T] padding mask
    ):

        B, T = movie_id.shape

        if field_status is None:
            field_status = torch.zeros(    # If no status provided, assume every field is observed
                B, T, 4,
                dtype=torch.long,
                device=movie_id.device
            )

        movie = self.movie_emb(movie_id)  # [B, T] -> [B, T, d]
        genre = self.genre_proj(genres)   # [B, T, G] -> [B, T, d]
        rating_repr = self.rating_proj(   # [B, T] -> [B, T, d]
            (rating/5.0).unsqueeze(-1)
        )
        time_repr = self.time_proj(       # [B, T] -> [B, T, d]
            torch.log1p(time_gap_hours.clamp_min(0.0)).unsqueeze(-1)
        )

        movie = self._status_transform(movie, field_status[..., 0], 0)
        rating_repr = self._status_transform(
            rating_repr, field_status[..., 1], 1
        )
        time_repr = self._status_transform(
            time_repr, field_status[..., 2], 2
        )
        genre = self._status_transform(genre, field_status[..., 3], 3)

        event = self.fuse(
            torch.cat([movie, rating_repr, time_repr, genre], dim=-1)
        )

        if valid_mask is not None:
            event = event * valid_mask.unsqueeze(-1)
        return event


class MaskedEventModel(nn.Module):
    def __init__(
        self,
        num_movies,
        num_genres,
        d_model=64,
        status_mode="ignore",
    ):
        super().__init__()
        self.encoder = EventEncoder(
            num_movies=num_movies,
            num_genres=num_genres,
            d_model=d_model,
            status_mode=status_mode,
        )
        self.movie_head = nn.Linear(d_model, num_movies)
        self.rating_head = nn.Linear(d_model, 1)
        self.time_head = nn.Linear(d_model, 1)
        self.genre_head = nn.Linear(d_model, num_genres)
        self.mask_head = nn.Linear(d_model, 4)

    def forward(self, values, field_status):
        z = self.encoder(
            movie_id=values["movie_id"],
            rating=values["rating"],
            time_gap_hours=values["time_gap_hours"],
            genres=values["genres"],
            field_status=field_status,
            valid_mask=values["valid_mask"],
        )
        return {
            "z": z,
            "movie_logits": self.movie_head(z),
            "rating_pred": self.rating_head(z).squeeze(-1),
            "time_pred": self.time_head(z).squeeze(-1),
            "genre_logits": self.genre_head(z),
            "mask_logits": self.mask_head(z),
        }
