import torch
import torch.nn.functional as F 

from src.objectives.corruption import (
    MASKED,                 # field_status code 3: "we hid this field on purpose"
    MOVIE,                  # field axis index 0 = movie_id (of 4 fields)
    donor_corrupt,          # replace hidden fields with another event's values (VIME)
    sample_corruption_mask, # random [B,T,4] bool mask; never marks padding
    zero_corrupt,           # set hidden fields to 0 and mark them MASKED
)

def clean_status(batch):
    # batch: collated sequence dict with movie_id/rating/.../is_real_event as [B, T, ...]
    # Returns field_status [B, T, 4] all OBSERVED (0) — no masks, fully clean inputs.
    b, t = batch["is_real_event"].shape
    return torch.zeros(b, t, 4, dtype=torch.long, device=batch["movie_id"].device)

def build_corrupted(batch, experiment, mask_probability):
    corruption_mask = sample_corruption_mask(
        batch["is_real_event"], # Exclude padding from corruption draws.
        mask_prob=mask_probability,
        group_movie_genres=True, # Hide movie and associated genre metadata together.
    )
    if experiment == "vime":
        corrupted = donor_corrupt(batch, corruption_mask) # VIME: swap in donor values instead of zeros.
    else:
        corrupted = zero_corrupt(batch, corruption_mask) # Shared raw corruption for zero/status/token.
    # Contract: status-aware treatments expose MASKED only where intended.
    if experiment in {"status", "token"}:
        assert torch.all(corrupted.field_status[corruption_mask] == MASKED) # Empty selections satisfy this vacuously.
    return corrupted # Return corrupted inputs and separate supervision metadata.

def causal_next_genre_loss(model, batch):
    # Autoregressive next-genre (teacher forcing): one embedding per time → predict next genres.
    # h_0→genres_1, h_1→genres_2, …; h_{T-1} is unused. Only real→real pairs in the loss.
    status = clean_status(batch)  # [B,T,4] all OBSERVED; clean (uncorrupted) inputs
    hidden = model(batch, status, direction="causal")  # [B,T,D]; h_t attends only to ≤ t
    # Shift to [B,T-1,*]: use embeddings 0..T-2; genre_head is Linear(D→G) only (no re-encode)
    logits = model.genre_head(hidden[:, :-1])  # [B,T-1,G]
    target = batch["genres"][:, 1:]  # [B,T-1,G]
    active = batch["is_real_event"][:, :-1] & batch["is_real_event"][:, 1:]  # [B,T-1]
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")  # [B,T-1,G]
    loss = (raw.mean(-1) * active).sum() / active.sum().clamp_min(1)  # mean over G, then valid pairs
    return loss

def masked_item_loss(model, batch, experiment, mask_probability):
    # Masked movie reconstruction (not next-step): corrupt → bidirectional encode → CE on hidden movie_ids.
    corrupted = build_corrupted(batch, experiment, mask_probability)  # zero or donor per experiment
    hidden = model(
        corrupted.values,  # must use corrupted inputs, not the clean batch
        corrupted.field_status,
        direction="bidirectional",  # full context OK for same-t reconstruction
    )  # [B,T,D]
    selected = corrupted.corruption_mask[..., MOVIE] & batch["is_real_event"]  # [B,T] masked movies at real events
    if not selected.any():
        return hidden.sum() * 0.0  # no movie masks this batch; keep graph/device, CE would error on N=0
    logits = model.item_head(hidden[selected])  # [N,num_movies]; multi-class over catalog
    target = batch["movie_id"][selected]  # [N] clean original IDs
    return F.cross_entropy(logits, target)

def strict_prefix(batch, random_cut):
    # Hold out event `cut` as the label; model may only see events 0..cut-1 (no future leakage).
    # Assumes real events are left-contiguous with right padding. Needs length >= 2 (not checked).
    lengths = batch["is_real_event"].sum(1)  # [B] real-event counts
    cuts = []
    for length in lengths.tolist():
        if random_cut:
            # Sample cut in 2..length-1; length==2 → cut=1 (one-event prefix)
            cut = torch.randint(2, max(3, length), (1,)).item()
            cut = min(int(cut), length - 1)
        else:
            cut = length - 1  # last real event is the held-out target
        cuts.append(cut)

    cuts = torch.tensor(cuts, device=lengths.device)  # [B] target index per user
    position = torch.arange(batch["is_real_event"].size(1), device=lengths.device)  # [T]
    prefix_valid = position.unsqueeze(0) < cuts.unsqueeze(1)  # [B,T] True only before cut
    # Shallow copy: tensors still hold the suffix, but is_real_event marks them as pad so attn ignores them
    prefix = dict(batch)
    prefix["is_real_event"] = prefix_valid
    row = torch.arange(len(cuts), device=cuts.device)
    target = batch["genres"][row, cuts]  # [B,G] genres at cut (not in visible prefix)
    return prefix, cuts, target


def strict_prefix_loss(model, batch, random_cut=True):
    # Predict held-out event genres from a strict prefix (future positions treated as padding).
    prefix, cuts, target = strict_prefix(batch, random_cut)
    hidden = model(
        prefix,
        clean_status(prefix),
        direction="bidirectional",  # bi-dir only inside the prefix; suffix already masked out
    )  # [B,T,D]
    row = torch.arange(len(cuts), device=cuts.device)
    pooled = hidden[row, cuts - 1]  # [B,D] last visible step (cut-1), not mean-pool
    logits = model.genre_head(pooled)  # [B,G]
    return F.binary_cross_entropy_with_logits(logits, target)  # multi-label BCE