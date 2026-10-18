import torch
from torch import nn

from src.objectives.corruption import MASKED


class EventEncoder(nn.Module):
    """
    status_mode controls how the encoder represents masked / missing fields:
      ignore       - ignore field_status; the model only sees the corrupted raw value.
      status_embed - use projection(value) + learned status embedding.
      mask_token   - for MASKED fields only, replace the field representation with a learned
                     mask token. Tokens are one per field group (movie/rating/time/genres),
                     not one per status code. Non-MASKED fields keep their projected
                     original value (status emb is added; OBSERVED adds zero).

    Experiment mapping (CLI --mode -> status_mode + corruption):
      zero   -> ignore       + zero-fill   # MAE: encoder sees zeros, no mask signal
      status -> status_embed + zero-fill   # MAE: zeros plus explicit status embedding
      token  -> mask_token   + zero-fill   # MAE: MASKED fields use a learned mask token
      vime   -> ignore       + donor fill  # donor values; status hidden so label is not leaked
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
            # One learnable token per field group (not per status). Only used when status==MASKED.
            self.mask_tokens = nn.Parameter(
                torch.randn(4, d) * 0.02
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

        # mask_token: non-MASKED keep projected original value; MASKED -> this field's token.
        # status_emb is still added for non-MASKED; OBSERVED (padding_idx=0) adds a zero vector.
        x = x + self.status_emd(status)                         # [B, T, d]
        is_masked = (status == MASKED).unsqueeze(-1)            # status [B, T] -> [B, T, 1]
        token = self.mask_tokens[field_index].view(1, 1, -1)    # [d] -> [1, 1, d]; field-group token
        return torch.where(is_masked, token, x)                 # broadcast -> [B, T, d]


    def forward(                      # Forward pass: raw fields -> event embeddings
        self,
        movie_id,                     # [B, T]
        rating,                       # [B, T]
        time_gap_hours,               # [B, T]
        genres,                       # [B, T, G]
        field_status=None,            # [B, T, 4] int64; see comments below
        is_real_event=None,           # [B, T] bool; True=real event, False=sequence padding
    ):
        # field_status: how to treat each field of each event (from corruption, or all 0).
        # Shape [B, T, 4]: B=batch, T=time, last dim = field groups:
        #   [..., 0] movie, [..., 1] rating, [..., 2] time_gap, [..., 3] genres
        # Each entry is a status code:
        #   0 OBSERVED        - value as given; status emb is zero (padding_idx=0)
        #   1 NATURAL_MISSING - missing in real data (not pretrain corruption)
        #   2 UNAVAILABLE     - field not available for this event
        #   3 MASKED          - deliberately corrupted for pretraining (MAE); triggers mask_token
        # Example: only rating MAE-masked -> field_status[b, t] = [0, 3, 0, 0]
        # MAE zero_corrupt sets MASKED where corruption_mask is True; VIME keeps all OBSERVED.
        # Used below as field_status[..., i] with shape [B, T] in _status_transform.
        #
        # Distinguish from corruption_mask [B,T,4] (which fields were corrupted for pretrain)
        # and is_real_event [B,T] (which timesteps are real vs padding).

        B, T = movie_id.shape

        if field_status is None:
            field_status = torch.zeros(    # Default: all OBSERVED (0)
                B, T, 4,
                dtype=torch.long,
                device=movie_id.device
            )

        movie = self.movie_emb(movie_id)  # [B, T] -> [B, T, d]
        genre = self.genre_proj(genres)   # [B, T, G] -> [B, T, d]
        rating_repr = self.rating_proj(   # [B, T] -> [B, T, 1] via unsqueeze -> [B, T, d]
            (rating/5.0).unsqueeze(-1)
        )
        time_repr = self.time_proj(       # [B, T] -> [B, T, 1] via unsqueeze -> [B, T, d]
            torch.log1p(time_gap_hours.clamp_min(0.0)).unsqueeze(-1)
        )

        movie = self._status_transform(movie, field_status[..., 0], 0)       # status [B, T]
        rating_repr = self._status_transform(
            rating_repr, field_status[..., 1], 1                            # status [B, T]
        )
        time_repr = self._status_transform(
            time_repr, field_status[..., 2], 2                              # status [B, T]
        )
        genre = self._status_transform(genre, field_status[..., 3], 3)      # status [B, T]

        event = self.fuse(
            torch.cat([movie, rating_repr, time_repr, genre], dim=-1)
        )

        if is_real_event is not None:
            event = event * is_real_event.unsqueeze(-1)  # zero out padded timesteps
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
            is_real_event=values["is_real_event"],
        )
        return {
            "z": z,
            "movie_logits": self.movie_head(z),
            "rating_pred": self.rating_head(z).squeeze(-1),
            "time_pred": self.time_head(z).squeeze(-1),
            "genre_logits": self.genre_head(z),
            "mask_logits": self.mask_head(z),
        }
