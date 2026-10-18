from dataclasses import dataclass
from typing import Dict
import torch


# Field index
MOVIE = 0          # Field index 0 represents movie_id
RATING = 1         # Field index 1 represent rating
TIME_GAP = 2       # Field index 2 represent time gap hours
GENRES = 3         # Field index 3 represent genres
N_FIELDS = 4       # Each event has four field groups in this dataset

# Status index
OBSERVED = 0             # Status index 0 represents observed event
NATURAL_MISSING = 1      # Status index 1 represents natural missing event
UNAVAILABLE = 2          # Status index 2 represents unavailable event
MASKED = 3               # Status index 3: field was deliberately masked/corrupted for pretraining


@dataclass
class CorruptedBatch:
    values: Dict[str, torch.Tensor]  # corrupted input tensors that will be fed into the model
    corruption_mask: torch.Tensor    # Boolean [B, T, 4]: True where a field was synthetically corrupted
    field_status: torch.Tensor       # Long [B, T, 4]: observation status visible to the encoder


def sample_corruption_mask(
    is_real_event: torch.Tensor,    # [B, T] bool; True = real event, False = sequence padding
    mask_prob: float,               # probability of corrupting each field or field group (0.0 to 1.0)
    group_movie_genres: bool = True,  # whether movie_id and genres are corrupted together
) -> torch.Tensor:
    """
    Sample which fields will be corrupted for this batch.

    is_real_event: [B, T], True only for real (non-padding) events.
    returns: [B, T, 4] boolean corruption_mask; True only for synthetically corrupted fields.
    Padding positions are never corrupted.
    """

    if not 0.0 < mask_prob < 1.0:
        raise ValueError("mask_prob must be between 0 and 1")

    B, T = is_real_event.shape
    corruption_mask = torch.zeros(
        B, T, N_FIELDS,
        dtype=torch.bool,
        device=is_real_event.device,
    )

    if group_movie_genres:
        entity_mask = torch.rand(
            B, T,
            device=is_real_event.device,
        ) < mask_prob
        corruption_mask[..., MOVIE] = entity_mask
        corruption_mask[..., GENRES] = entity_mask

        corruption_mask[..., RATING] = (
            torch.rand(B, T, device=is_real_event.device) < mask_prob
        )
        corruption_mask[..., TIME_GAP] = (
            torch.rand(B, T, device=is_real_event.device) < mask_prob
        )
    else:
        corruption_mask = torch.rand(
            B, T, N_FIELDS,
            device=is_real_event.device,
        ) < mask_prob

    # Never corrupt padding positions.
    corruption_mask &= is_real_event.unsqueeze(-1)
    return corruption_mask


def _base_values(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Return a new dictionary so corruption does not mutate the clean batch.
    return {
        "movie_id": batch["movie_id"].clone(),
        "rating": batch["rating"].clone(),
        "time_gap_hours": batch["time_gap_hours"].clone(),
        "genres": batch["genres"].clone(),
        "is_real_event": batch["is_real_event"],
    }


def zero_corrupt(
    batch: Dict[str, torch.Tensor],
    corruption_mask: torch.Tensor,  # Boolean [B, T, 4] from sample_corruption_mask
) -> CorruptedBatch:
    """
    Corruption used by zero/status/token modes (zero-fill).
    """
    values = _base_values(batch)
    values["movie_id"] = values["movie_id"].masked_fill(
        corruption_mask[..., MOVIE],
        0,
    )

    values["rating"] = values["rating"].masked_fill(
        corruption_mask[..., RATING],
        0.0,
    )

    values["time_gap_hours"] = values["time_gap_hours"].masked_fill(
        corruption_mask[..., TIME_GAP],
        0.0,
    )

    values["genres"] = values["genres"].masked_fill(
        corruption_mask[..., GENRES].unsqueeze(-1),
        0.0,
    )

    status = torch.zeros_like(corruption_mask, dtype=torch.long)
    status[corruption_mask] = MASKED
    return CorruptedBatch(values, corruption_mask, status)


def _real_event_index(is_real_event: torch.Tensor) -> torch.Tensor:
    """Flat indices of real (non-padding) events in a [B, T] layout."""
    return is_real_event.reshape(-1).nonzero(as_tuple=False).squeeze(1)


def _sample_donor_index(real_index: torch.Tensor, device) -> torch.Tensor:
    """
    One random permutation of real_index used as donor sources.
    Same length as real_index; entry i is the donor for the i-th real event
    when that event is selected for replacement.
    """
    if real_index.numel() == 0:
        return real_index
    return real_index[torch.randperm(real_index.numel(), device=device)]


def _replace_with_donors(
    x: torch.Tensor,                 # Field tensor, [B, T] or [B, T, K]
    is_real_event: torch.Tensor,     # [B, T] True = real event (not padding)
    replace_mask: torch.Tensor,      # [B, T] True = positions to replace with a donor
    real_index: torch.Tensor,        # Flat indices of real events
    donor_index: torch.Tensor,       # Donor source for each real-event slot (same perm for grouped fields)
) -> torch.Tensor:
    """
    Helper for VIME-style donor replacement.
    Donors are sampled only from real (non-padding) event positions.
    Pass the same donor_index for fields that must stay entity-consistent
    (movie_id and genres when they share a corruption mask).
    """

    out = x.clone()
    B, T = is_real_event.shape
    trailing_shape = x.shape[2:]

    if real_index.numel() == 0:
        return out

    flat = x.reshape(B * T, *trailing_shape)
    out_flat = out.reshape(B * T, *trailing_shape)

    selected = replace_mask.reshape(-1)[real_index]
    target_index = real_index[selected]
    source_index = donor_index[selected]
    out_flat[target_index] = flat[source_index]

    return out


def donor_corrupt(
    batch: Dict[str, torch.Tensor],
    corruption_mask: torch.Tensor,  # Boolean [B, T, 4] corruption labels
) -> CorruptedBatch:
    """
    VIME-style corruption: use valid donor values from other event rows.
    Do not reveal MASKED status to the encoder (field_status stays OBSERVED).

    movie_id and genres share one donor permutation so a corrupted entity stays
    consistent (same donor event for both fields). rating and time_gap each get
    their own independent donor permutation.
    """
    values = _base_values(batch)
    is_real = batch["is_real_event"]
    real_index = _real_event_index(is_real)

    # Shared donors for the movie+genres entity group.
    entity_donors = _sample_donor_index(real_index, is_real.device)
    values["movie_id"] = _replace_with_donors(
        values["movie_id"],
        is_real,
        corruption_mask[..., MOVIE],
        real_index,
        entity_donors,
    )
    values["genres"] = _replace_with_donors(
        values["genres"],
        is_real,
        corruption_mask[..., GENRES],
        real_index,
        entity_donors,
    )

    rating_donors = _sample_donor_index(real_index, is_real.device)
    values["rating"] = _replace_with_donors(
        values["rating"],
        is_real,
        corruption_mask[..., RATING],
        real_index,
        rating_donors,
    )

    time_donors = _sample_donor_index(real_index, is_real.device)
    values["time_gap_hours"] = _replace_with_donors(
        values["time_gap_hours"],
        is_real,
        corruption_mask[..., TIME_GAP],
        real_index,
        time_donors,
    )

    status = torch.zeros_like(corruption_mask, dtype=torch.long)
    return CorruptedBatch(values, corruption_mask, status)
