import torch
from torch import nn

from src.models.event_encoder import EventEncoder 

def causal_attention_mask(length, device):
    # Transformer causal mask: True means query i can attend to key j
    return torch.tril(
        torch.ones(length, length, dtype=torch.bool, device=device), #[T, T]
        diagonal = 0, 
    )

class BehavioralSequenceModel(nn.Module):
    def __init__(
        self,
        num_movies,
        num_genres,
        status_mode,
        d_model=64,      # Sequence embedding width
        nhead=4,         # Attention heads
        num_layers=2,
        dropout=0.1
    ):

        super().__init__()

        # create event-level encoder
        self.event_encoder = EventEncoder(
            num_movies = num_movies,
            num_genres = num_genres,
            d_model = d_model,
            status_mode = status_mode,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,    # Positionwise feed-forward expansion
            dropout=dropout,
            batch_first=True,             # Input use [B, T, D]
            norm_first=True,              # Pre-normalization architecture
            activation="gelu"
        )

        self.sequence = nn.TransformerEncoder(layer, num_layers) # Original adds no explicit positional encoding
        self.genre_head = nn.Linear(d_model, num_genres) # Independent genre logits for BCE
        self.item_head = nn.Linear(d_model, num_movies)  # Categorical item logits for CE


    def encode_events(self, values, field_status):
        return self.event_encoder(
            movie_id=values["movie_id"], # [B,T] integer IDs.
            rating=values["rating"], # [B,T] scalar feedback.
            time_gap_hours=values["time_gap_hours"], # [B,T] time features must be cutoff-safe.
            genres=values["genres"], # [B,T,G] multi-hot movie metadata.
            field_status=field_status, # [B,T,4] encoder-visible observation state.
            valid_mask=values["is_real_event"], # real event or padding; True = real event, False = padding
        )


    def forward(self, values, field_status, direction):
        valid = values["is_real_event"].bool() # [B,T] real/visible event positions; excludes padding.
        event = self.encode_events(values, field_status) # Project fields before sequence mixing.
        can_attend = None # No temporal edge restriction for bidirectional attention.
        if direction == "causal":
            can_attend = causal_attention_mask(event.size(1), event.device) # True = allowed.
        elif direction != "bidirectional":
            raise ValueError(f"unknown direction={direction}") # Fail rather than silently choosing a mask.

        hidden = self.sequence(
            event,
            mask=None if can_attend is None else ~can_attend, # API adapter: Transformer expects True = blocked.
            # Key-padding mask: True means ignore this key.
            src_key_padding_mask=~valid, # [B,T] remove invalid keys from every query's context.
        )
        return hidden * valid.unsqueeze(-1) # Zero invalid query outputs; this is not a substitute for key masking.


