import torch
from torch import nn
from src.models.event_encoder import EventEncoder
from src.models.temporal_state import (
    EventOnlyState, SingleTimescaleState, DualTimescaleState,
)


# Event encoder + temporal user state + next-genre head for backbone ablations.
class BehavioralStateModel(nn.Module):
    def __init__(self, num_movies, num_genres, backbone, width=64,
                 status_mode="ignore", tau_single_hours=24,
                 tau_fast_hours=2, tau_slow_hours=720,
                 learnable_tau=False, head_seed=7):
        super().__init__()
        # Per-event field embeddings -> z [B, T, W]
        self.event_encoder = EventEncoder(
            num_movies=num_movies, num_genres=num_genres,
            d_model=width, status_mode=status_mode,
        )
        # Pick temporal memory: none / one τ / fast+slow gated mix
        if backbone == "event_only":
            self.temporal = EventOnlyState()
        elif backbone == "single":
            self.temporal = SingleTimescaleState(width, tau_single_hours, learnable_tau)
        elif backbone == "dual":
            self.temporal = DualTimescaleState(
                width, tau_fast_hours, tau_slow_hours, learnable_tau,
            )
        else:
            raise ValueError(backbone)
        # Temporal modules consume different amounts of global RNG on construction.
        # Fork + fixed seed so next_genre_head gets identical init across backbones.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.next_genre_head = nn.Linear(width, num_genres)  # user_z -> genre logits
        self.encoder_frozen = False  # set True after freeze_event_encoder()

    def freeze_event_encoder(self):
        # Stop updating encoder weights (e.g. probe / transfer on frozen z).
        self.encoder_frozen = True
        self.event_encoder.requires_grad_(False)
        self.event_encoder.eval()

    def train(self, mode=True):
        # Override: keep frozen encoder in eval even when the rest of the model trains.
        super().train(mode)
        if self.encoder_frozen:
            self.event_encoder.eval()
        return self

    def forward(self, batch):
        # Encode events, run temporal state, predict next-genre logits from user_z.
        b, t = batch["is_real_event"].shape
        # All fields observed (status 0); shape [B, T, 4] for movie/rating/time/genres
        status = torch.zeros(b, t, 4, dtype=torch.long,
                             device=batch["movie_id"].device)
        event_z = self.event_encoder(
            movie_id=batch["movie_id"], rating=batch["rating"],
            time_gap_hours=batch["time_gap_hours"], genres=batch["genres"],
            field_status=status, is_real_event=batch["is_real_event"],
        )  # [B, T, W]
        state = self.temporal(event_z, batch["time_gap_hours"], batch["is_real_event"])
        # state has user_z (+ optional fast_z/slow_z/gate); attach encoder z and logits
        state["event_z"] = event_z
        state["logits"] = self.next_genre_head(state["user_z"])  # [B, T, num_genres]
        return state
