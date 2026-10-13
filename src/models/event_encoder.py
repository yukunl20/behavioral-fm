import torch
from torch import nn

class EventEncoder(nn.Module):
    """
    status_mode controls how the encoder represents masked / missing fields:
      none  - ignore field_status; the model only sees the corrupted raw value.
      add   - use projection(value) + learned status embedding.
      token - for SYNTHETIC_MASK fields, replace the whole field representation with a learned field-specific mask token.

    Experiment mapping:
      zero   -> none   # zero-filled mask, but encoder is not told it was masked
      status -> add    # zero-filled mask plus explicit status signal
      token  -> token  # MAE/BERT-style learned mask token
      vime   -> none   # donor corruption; status must not reveal the corruption label
    """

    def __init__(
        self, 
        num_movies, 
        num_genres, 
        d_model=64,
        status_mode = "none",
        ):

        super().__init__()
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by four")
        
        if status_mode not in {"none", "add", "token"}:
            raise ValueError(f"unknown status_mode: {status_mode}")
        
        self.status_mode = status_mode
        self.d_model = d_model              # Store event embedding width D
        self.d_field = d_model // 4         # Each of four field groups gets D/4 dimensions

        d = self.d_field
        self.movie_emb = nn.Embedding(      # Categorical embedding table for movie_id
            num_movies,                     # Number of rows in embedding table
            d,                              # Embedding width per movie
            padding_idx = 0                 # movie_id = 0 is padding; embedding stays as zero
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

        if status_mode in {"add", "token"}:  # Only modes add/token need a status embedding table
            self.status_emd = nn.Embedding(  # Learn embeddings for observation status codes
                4,                           # Four status codes: observed/natural/unavailable/synthetic
                d,                           # Status vector has the same width as one field representation
                padding_idx = 0              # OBSERVED=0 uses a fixed zero embedding
            )

        if status_mode == "token":                # Token mode needs learned mask tokens
            self.mask_tokens = nn.Parameter(      # Register mask tokens as trainable parameters
                torch.randn(4, d) * 0.02          # One small random token per field group
            )

        self.fuse = nn.Sequential(          # Fusion network after concatenating field vectors
            nn.Linear(d * 4, d_model),      # Combine four field vectors [4d] into [D]
            nn.LayerNorm(d_model),          # Normalize event embedding for stability
            nn.GELU()                       # Add nonlinearity
        )

    def _status_transform(self, x, status, field_index):  # Apply status logic to one projected field
        if self.status_mode == "none":
            return x
        
        if self.status_mode == "add":
            return x + self.status_emd(status)
        
        x = x + self.status_emd(status)                         # Token mode, still encode non-synthetic statuses additively
        is_synthetic = (status == 3).unsqueeze(-1)              # [B, T, 1], True where status is SYNTHETIC_MASK
        token = self.mask_tokens[field_index].view(1, 1, -1)    # Select this field's learned mask token.
        return torch.where(is_synthetic, token, x)              # Replace synthetic masked positions by token; keep others unchanged.


    def forward(                      # Forward pass: raw fields -> event embeddings
        self,                         
        movie_id,                     # [B, T]
        rating,                       # [B, T]
        time_gap_hours,               # [B, T]
        genres,                       # [B, T, G]
        field_status=None,            # [B, T, 4] status code for the four field groups for different experiment
        valid_mask=None,              # [B, T] padding mask
    ):

        B, T = movie_id.shape           

        if field_status is None: 
            field_status = torch.zeros(    # If not status provided, assume every field is observed
                B, T, 4,
                dtype = torch.long,
                device = movie_id.device
            )
        
        movie = self.movie_emb(movie_id)  # [B, T] -> [B, T, d]
        genre = self.genre_proj(genres)   # [B, T, G] -> [B, T, d]
        rating_repr = self.rating_proj(   # [B, T] -> [B, T, d]
            (rating/5.0).unsqueeze(-1)
        )
        time_repr = self.time_proj(       # [B, T] -> [B, T, d]
            torch.log1p(time_gap_hours.clamp_min(0.0)).unsqueeze(-1)   
        )
        
        movie = self._status_transform(movie, field_status[..., 0], 0) # Apply status handling to movie field.
        rating_repr = self._status_transform(                          # Apply status handling to rating field.
            rating_repr, field_status[..., 1], 1
        )
        time_repr = self._status_transform(                            # Apply status handling to time-gap field.
            time_repr, field_status[..., 2], 2
        )
        genre = self._status_transform(genre, field_status[..., 3], 3) # Apply status handling to genre field.

        event = self.fuse(                       # Fuse all field representations.
            torch.cat([movie, rating_repr, time_repr, genre], dim=-1) # Concatenate four [B,T,d] tensors into [B,T,4d].
        )

        if valid_mask is not None:                    
            event = event * valid_mask.unsqueeze(-1)  # Zero out padded event embeddings so pooling cannot use them.
        return event                                  # Return event embeddings [B, T, D]


class MaskedEventModel(nn.Module):               
    def __init__(                                
        self,                                    
        num_movies,                              
        num_genres,                              
        d_model=64,                              
        status_mode="none",                      
    ):
        super().__init__()                       
        self.encoder = EventEncoder(             
            num_movies=num_movies,              
            num_genres=num_genres,               
            d_model=d_model,                    
            status_mode=status_mode,             
        )
        self.movie_head = nn.Linear(d_model, num_movies)  # Predict movie_id logits from event embedding.
        self.rating_head = nn.Linear(d_model, 1)          # Predict rating scalar from event embedding.
        self.time_head = nn.Linear(d_model, 1)            # Predict log time-gap scalar from event embedding.
        self.genre_head = nn.Linear(d_model, num_genres)  # Predict multi-label genre logits from event embedding.
        self.mask_head = nn.Linear(d_model, 4)            # Predict which fields were corrupted for VIME.

    def forward(self, values, field_status):      # Forward pass for corrupted batch.
        z = self.encoder(                         # Encode the corrupted event fields.
            movie_id=values["movie_id"],          # Pass movie ids [B,T].
            rating=values["rating"],              # Pass ratings [B,T].
            time_gap_hours=values["time_gap_hours"], # Pass time gaps [B,T].
            genres=values["genres"],              # Pass genres [B,T,G].
            field_status=field_status,            # Pass model-visible field statuses [B,T,4].
            valid_mask=values["valid_mask"],      # Pass padding mask [B,T].
        )
        return {                                  # Return shared representation plus all head outputs.
            "z": z,                              # [B,T,D] event representation.
            "movie_logits": self.movie_head(z),  # [B,T,num_movies] movie reconstruction logits.
            "rating_pred": self.rating_head(z).squeeze(-1), # [B,T] rating predictions.
            "time_pred": self.time_head(z).squeeze(-1),     # [B,T] time predictions.
            "genre_logits": self.genre_head(z),  # [B,T,G] genre reconstruction logits.
            "mask_logits": self.mask_head(z),    # [B,T,4] VIME corruption-detection logits.
        }








