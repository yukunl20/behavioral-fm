import torch  # Import PyTorch for tensors, random fake data, and tensor assertions.

from src.models.event_encoder import EventEncoder  # Import EventEncoder to test padded output behavior.
from src.objectives.corruption import (            # Import corruption utilities and field indices.
    GENRES,                                      # Field index for genres.
    MASKED,                                      # Status code for deliberately masked fields.
    MOVIE,                                       # Field index for movie_id.
    sample_field_mask,                           # Function that samples synthetic field corruption masks.
    zero_corrupt,                                # Function that zero-fills masked values and sets field_status.
)


def synthetic_batch():                           # Build a tiny fake batch with known padding.
    return {                                     # Return same dictionary structure as real DataLoader batch.
        "movie_id": torch.tensor([[1, 2, 0], [3, 4, 5]]), # [B=2,T=3]; first row position 2 is padding.
        "rating": torch.tensor([[5.0, 3.0, 0.0], [4.0, 2.0, 1.0]]), # [2,3] fake ratings.
        "time_gap_hours": torch.tensor(          # [2,3] fake time gaps.
            [[0.0, 2.0, 0.0], [0.0, 5.0, 9.0]]  # Padding has zero, but valid_mask defines true padding.
        ),
        "genres": torch.randn(2, 3, 6),          # [B=2,T=3,G=6] fake genre vectors.
        "valid_mask": torch.tensor(              # [2,3] boolean real-event mask.
            [[True, True, False], [True, True, True]] # False marks the padded event.
        ),
    }


def test_padding_is_never_corrupted():           # Verify corruption sampler respects valid_mask.
    batch = synthetic_batch()                     # Create fake batch.
    mask = sample_field_mask(batch["valid_mask"], 0.999) # Use high probability to stress the padding rule.
    assert not mask[0, 2].any()                   # Padded event [0,2] must not be corrupted in any field.
    assert torch.equal(mask[..., MOVIE], mask[..., GENRES]) # movie_id and genres must share the same group mask.


def test_zero_corruption_sets_status():          # Verify zero_corrupt writes MASKED status.
    batch = synthetic_batch()                     # Create fake batch.
    mask = sample_field_mask(batch["valid_mask"], 0.999) # Sample corruption mask.
    corrupted = zero_corrupt(batch, mask)         # Apply zero corruption.
    assert (corrupted.field_status[mask] == MASKED).all() # Every corrupted field must be MASKED.
    assert not corrupted.field_status[0, 2].any()    # Padding position should remain all status 0.


def test_encoder_zeros_padded_events():          # Verify EventEncoder output is zero at padded events.
    batch = synthetic_batch()                     # Create fake batch.
    model = EventEncoder(                         # Build a small encoder for test.
        num_movies=10,                            # Fake movie vocabulary size.
        num_genres=6,                             # Genre dimension matches fake genres.
        d_model=32,                               # Small representation width for fast test.
        status_mode="mask_token",                 # Test the most complex status mode.
    )
    status = torch.zeros(2, 3, 4, dtype=torch.long) # [B,T,4] all-observed field status.
    z = model(                                    # Run encoder forward pass.
        batch["movie_id"],                        # [2,3] movie ids.
        batch["rating"],                          # [2,3] ratings.
        batch["time_gap_hours"],                  # [2,3] time gaps.
        batch["genres"],                          # [2,3,6] genres.
        field_status=status,                      # [2,3,4] status codes.
        valid_mask=batch["valid_mask"],           # [2,3] padding mask.
    )
    assert z.shape == (2, 3, 32)                  # Encoder output should be [B,T,D].
    assert torch.equal(z[0, 2], torch.zeros(32))  # Padded event output should be exactly zero.