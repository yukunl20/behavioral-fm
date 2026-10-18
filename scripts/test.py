import torch  # Import PyTorch for tensors, random fake data, and tensor assertions.

from src.models.event_encoder import EventEncoder  # Import EventEncoder to test padded output behavior.
from src.objectives.corruption import (            # Import corruption utilities and field indices.
    GENRES,                                      # Field index for genres.
    MASKED,                                      # Status code for deliberately masked fields.
    MOVIE,                                       # Field index for movie_id.
    donor_corrupt,                               # VIME donor replacement.
    sample_corruption_mask,                      # Sample which fields to corrupt.
    zero_corrupt,                                # Zero-fill corruption + MASKED status.
)


def synthetic_batch():                           # Build a tiny fake batch with known padding.
    return {                                     # Return same dictionary structure as real DataLoader batch.
        "movie_id": torch.tensor([[1, 2, 0], [3, 4, 5]]), # [B=2,T=3]; first row position 2 is padding.
        "rating": torch.tensor([[5.0, 3.0, 0.0], [4.0, 2.0, 1.0]]), # [2,3] fake ratings.
        "time_gap_hours": torch.tensor(          # [2,3] fake time gaps.
            [[0.0, 2.0, 0.0], [0.0, 5.0, 9.0]]  # Padding has zero, but is_real_event defines true padding.
        ),
        "genres": torch.randn(2, 3, 6),          # [B=2,T=3,G=6] fake genre vectors.
        "is_real_event": torch.tensor(              # [2,3] True=real event, False=padding.
            [[True, True, False], [True, True, True]] # False marks the padded event.
        ),
    }


def test_padding_is_never_corrupted():           # Verify corruption sampler respects is_real_event.
    batch = synthetic_batch()                     # Create fake batch.
    corruption_mask = sample_corruption_mask(batch["is_real_event"], 0.999)
    assert not corruption_mask[0, 2].any()        # Padded event [0,2] must not be corrupted in any field.
    assert torch.equal(corruption_mask[..., MOVIE], corruption_mask[..., GENRES])


def test_donor_corrupt_keeps_movie_genre_entity():
    """movie_id and genres share a corruption mask, so donors must be the same event."""
    batch = synthetic_batch()
    # Give each real event a unique (movie_id, genres) pair.
    batch["movie_id"] = torch.tensor([[10, 20, 0], [30, 40, 50]])
    genres = torch.zeros(2, 3, 6)
    genres[0, 0, 0] = 1.0
    genres[0, 1, 1] = 1.0
    genres[1, 0, 2] = 1.0
    genres[1, 1, 3] = 1.0
    genres[1, 2, 4] = 1.0
    batch["genres"] = genres

    pair = {}
    is_real = batch["is_real_event"]
    for b in range(2):
        for t in range(3):
            if is_real[b, t]:
                mid = int(batch["movie_id"][b, t])
                pair[mid] = batch["genres"][b, t].clone()

    corruption_mask = torch.zeros(2, 3, 4, dtype=torch.bool)
    corruption_mask[..., MOVIE] = is_real
    corruption_mask[..., GENRES] = is_real
    corrupted = donor_corrupt(batch, corruption_mask)

    for b in range(2):
        for t in range(3):
            if not is_real[b, t]:
                continue
            mid = int(corrupted.values["movie_id"][b, t])
            assert mid in pair
            assert torch.equal(corrupted.values["genres"][b, t], pair[mid])


def test_zero_corruption_sets_status():          # Verify zero_corrupt writes MASKED status.
    batch = synthetic_batch()                     # Create fake batch.
    corruption_mask = sample_corruption_mask(batch["is_real_event"], 0.999)
    corrupted = zero_corrupt(batch, corruption_mask)
    assert (corrupted.field_status[corruption_mask] == MASKED).all()
    assert not corrupted.field_status[0, 2].any()    # Padding position should remain all status 0.


def test_mlp_decoder_keeps_encoder_z_for_probe():
    from src.models.event_encoder import MaskedEventModel

    batch = synthetic_batch()
    model = MaskedEventModel(
        num_movies=10,
        num_genres=6,
        d_model=32,
        status_mode="ignore",
        decoder="mlp",
        decoder_layers=2,
    )
    status = torch.zeros(2, 3, 4, dtype=torch.long)
    out = model(batch, status)
    assert out["z"].shape == (2, 3, 32)
    assert out["h"].shape == (2, 3, 32)
    assert out["movie_logits"].shape == (2, 3, 10)
    # Probe path must use encoder z, which can differ from decoder h.
    assert not torch.equal(out["z"], out["h"])


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
        is_real_event=batch["is_real_event"],           # [2,3] True=real, False=padding.
    )
    assert z.shape == (2, 3, 32)                  # Encoder output should be [B,T,D].
    assert torch.equal(z[0, 2], torch.zeros(32))  # Padded event output should be exactly zero.