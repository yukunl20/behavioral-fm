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
SYNTHETIC_MASK = 3       # Status index 3 represents the field was deliberately masked/corrupted for pretraining


@dataclass
class CorruptedBatch:
    values: Dict[str, torch.Tensor] # corrupted input tensors that will be fed into the model
    field_mask: torch.Tensor        # Boolean [B, T, 4] tensor: True where a field was synthetically corrupted
    field_status: torch.Tensor      # Long [B, T, 4] tensor: observation status visible to the encoder


def sample_field_mask( 
    valid_mask: torch.Tensor,           # [B, T] boolean tensor; True means real event, False means padding
    mask_prob: float,                   # probability of corrupting each field or field group (0.0 to 1.0)
    group_movie_genres: bool = True,    # whether movie_id and genres are corrupted together (True) or not (False)
) -> torch.Tensor:
    """
    Sample which field will be corrupted for this batch
    valid_mask: [B, T], True only for real events
    returns: [B, T, 4] boolean tensor, True only for synthetically corrupted fields
    """

    if not 0.0 < mask_prob < 1.0:    # Reject invalid probabilities such as 0, 1, negative, or >1.
        raise ValueError("mask_prob must be between 0 and 1") # Fail early with a clear error.

    B, T = valid_mask.shape          # Read batch size B and sequence length T from valid_mask.
    mask = torch.zeros(              # Start from an all-False mask instead of sampling all fields first.
        B, T, N_FIELDS,              # Shape [B,T,4], one boolean per event field.
        dtype=torch.bool,            # The mask is boolean: True means corrupted.
        device=valid_mask.device     # Create tensor on the same device as valid_mask.
    )

    if group_movie_genres:           # If true, treat movie_id and genres as one semantic entity group.
        entity_mask = torch.rand(    # Sample one Bernoulli decision per event for the entity group.
            B, T,                    # Shape [B,T], shared by movie_id and genres.
            device=valid_mask.device # Keep device consistent.
        ) < mask_prob                # Marginal probability for movie_id and genres is still mask_prob.
        mask[..., MOVIE] = entity_mask  # Apply same entity mask to movie_id.
        mask[..., GENRES] = entity_mask # Apply same entity mask to genres.

        mask[..., RATING] = (        # Sample rating corruption independently.
            torch.rand(B, T, device=valid_mask.device) < mask_prob
        )
        mask[..., TIME_GAP] = (      # Sample time-gap corruption independently.
            torch.rand(B, T, device=valid_mask.device) < mask_prob
        )
    else:                            # If no grouping is requested...
        mask = torch.rand(           # Sample every field independently.
            B, T, N_FIELDS,          # Shape [B,T,4].
            device=valid_mask.device # Keep device consistent.
        ) < mask_prob                # Each field has marginal probability mask_prob.

    mask &= valid_mask.unsqueeze(-1) # Force all padding positions to False for every field.
    return mask                      # Return [B,T,4] synthetic corruption mask.


def _base_values(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
     # Return a new dictionary so corruption does not mutate the clean batch.
    return {
        "movie_id": batch["movie_id"].clone(),
        "rating": batch["rating"].clone(),
        "time_gap_hours": batch["time_gap_hours"].clone(),
        "genres": batch["genres"].clone(),
        "valid_mask": batch["valid_mask"],
    }


def zero_corrupt(
    batch: Dict[str, torch.Tensor],     # Original clean batch from DataLoader
    field_mask: torch.Tensor,           # Boolean [B, T, 4] mask sampled by sample_field_mask
) -> CorruptedBatch:                    # Return corrupted values, field_mask, and field_status
    """
    Corruption used by zero/status/token modes
    """
    values = _base_values(batch)  # make editable copies of clean input fields
    values["movie_id"] = values["movie_id"].masked_fill(
        field_mask[..., MOVIE],   # select movie_id corruption positions [B, T]
        0                         # Fill with id 0, the padding/movie mask id
    )

    values["rating"] = values["rating"].masked_fill(
        field_mask[..., RATING],
        0.0
    )

    values["time_gap_hours"] = values["time_gap_hours"].masked_fill(
        field_mask[..., TIME_GAP],
        0.0
    )

    values["genres"] = values["genres"].masked_fill(
        field_mask[..., GENRES].unsqueeze(-1),
        0.0
    )

    status = torch.zeros_like(field_mask, dtype=torch.long) # Start with all fields marked OBSERVED=0.
    status[field_mask] = SYNTHETIC_MASK                     # Mark corrupted fields as SYNTHETIC_MASK=3.
    return CorruptedBatch(values, field_mask, status)       # Package model inputs and supervision masks.


def _replace_with_donors(
    x: torch.Tensor,                 # Field tensor, [B, T] or [B, T, k]
    valid_mask: torch.Tensor,        # [B, T] boolean mask for real events
    replace_mask: torch.Tensor       # [B, T] boolean mask saying which positions to replace
) -> torch.Tensor:
    """
    Helper for VIME-style donor replacement
    x: [B, T] or [B, T, K]
    valid_mask: [B, T]
    replace_mask: [B, T]
    Donors are sampled only from valid event positions
    """

    out = x.clone()
    B, T = valid_mask.shape
    trailing_shape = x.shape[2:]      # capture extra dimensions, such as genre dimension [G]

    flat = x.reshape(B*T, *trailing_shape)   # flatten event positions for easier indexing
    out_flat = out.reshape(B*T, *trailing_shape) # flatten the output the same way

    valid_index = valid_mask.reshape(-1).nonzero(as_tuple=False).squeeze(1)
    
    # no valid index to replace
    if valid_index.numel() == 0:
        return out 

    donor_index = valid_index[
        torch.randperm(valid_index.numel(), device=x.device)
    ]

    selected = replace_mask.reshape(-1)[valid_index]
    target_index = valid_index[selected]
    source_index = donor_index[selected]
    out_flat[target_index] = flat[source_index]

    return out

def donor_corrupt(                    # VIME-style corruption using donor replacement.
    batch: Dict[str, torch.Tensor],   # Original clean batch.
    field_mask: torch.Tensor,         # Boolean [B,T,4] corruption labels.
) -> CorruptedBatch:                  # Return corrupted values, hidden field_mask, and visible status.
    """
    VIME-style corruption: use valid donor values from other event rows.
    Do not reveal synthetic mask status to the encoder.
    """
    values = _base_values(batch)      # Clone clean values so donor replacement is non-mutating.
    valid = batch["valid_mask"]       # Reuse real-event mask for safe donor sampling.

    values["movie_id"] = _replace_with_donors( # Replace selected movie ids with valid donor movie ids.
        values["movie_id"], valid, field_mask[..., MOVIE]
    )
    values["rating"] = _replace_with_donors( # Replace selected ratings with valid donor ratings.
        values["rating"], valid, field_mask[..., RATING]
    )
    values["time_gap_hours"] = _replace_with_donors( # Replace selected time gaps with valid donor time gaps.
        values["time_gap_hours"], valid, field_mask[..., TIME_GAP]
    )
    values["genres"] = _replace_with_donors( # Replace selected genre vectors with valid donor genre vectors.
        values["genres"], valid, field_mask[..., GENRES]
    )

    status = torch.zeros_like(field_mask, dtype=torch.long) # VIME hides corruption status; all fields appear observed.
    return CorruptedBatch(values, field_mask, status)       # field_mask is still kept for the auxiliary mask head.


