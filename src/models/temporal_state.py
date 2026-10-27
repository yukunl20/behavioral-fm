import math
import torch
from torch import nn

# Causal exponential memory: blend past state with each real event using e^{-Δt/τ}.
class DecayMemory(nn.Module):
    def __init__(self, tau_hours, learnable_tau=False):
        super().__init__()
        if not math.isfinite(tau_hours) or not 1/60 <= tau_hours <= 24 * 365:
            raise ValueError("tau_hours must be finite and within [1/60, 8760]")
        
        # store the log value keeps the learnable value to be positive when applying exponential to resume it
        value = torch.tensor(float(tau_hours)).log()
        if learnable_tau:
            self.log_tau = nn.Parameter(value)
        else:
            self.register_buffer("log_tau", value)

    @property
    def tau_hours(self):
        return self.log_tau.clamp(math.log(1 / 60), math.log(24 * 365)).exp()
    
    def forward(self, z, time_gap_hours, is_real_event):
        # Caller validates finite real-event inputs and nonnegative real gaps
        b, length, width = z.shape
        
        if length == 0:
            raise ValueError("need at least one time position")
        
        state = z.new_zeros(b, width)  # [B, W] running memory
        seen = torch.zeros(b, 1, dtype=torch.bool, device=z.device)  # [B, 1] any real event yet?

        outputs = []

        for t in range(length):
            real = is_real_event[:, t, None]  # [B, 1] real vs padding
            current = z[:, t, :]  # [B, W] event embedding
            delta = time_gap_hours[:, t, :].clamp_min(1 / 60)  # [B, ...] hours since last real
            ratio = delta / self.tau_hours  # [B, ...] Δt / τ
            retain = torch.exp(-ratio)[:, None]  # [B, 1] keep old state: e^{-Δt/τ}
            arrival = (-torch.expm1(-ratio))[:, None]  # [B, 1] mix in current: 1 - e^{-Δt/τ}
            updated = retain * state + arrival * current  # [B, W] decayed blend
            updated = torch.where(seen, updated, current)  # first real event: no decay
            state = torch.where(real, updated, state)  # padding: leave state unchanged
            outputs.append(torch.where(real, state, torch.zeros_like(state)))  # [B, W]
            seen = seen | real  # [B, 1]
        return torch.stack(outputs, dim=1)  # [B, T, W]
            

# Baseline: no temporal memory; just zero out padding in z.
class EventOnlyState(nn.Module):
    def forward(self, z, time_gap_hours, is_real_event):
        return dict(
            user_z=z * is_real_event.unsqueeze(-1),
            fast_z=None, slow_z=None, gate=None,
        )


# One DecayMemory bank; residual-add memory into z, then LayerNorm + mask padding.
class SingleTimescaleState(nn.Module):
    def __init__(self, width, tau_hours=24, learnable_tau=False):
        super().__init__()
        self.memory = DecayMemory(tau_hours, learnable_tau)
        self.norm = nn.LayerNorm(width)

    def forward(self, z, time_gap_hours, is_real_event):
        memory = self.memory(z, time_gap_hours, is_real_event)
        return dict(
            user_z=self.norm(z + memory) * is_real_event.unsqueeze(-1),
            fast_z=memory, slow_z=None, gate=None,
        )


# Fast + slow DecayMemory banks; gate mixes them (and time gap) before residual into z.
class DualTimescaleState(nn.Module):
    def __init__(self, width, tau_fast_hours=2, tau_slow_hours=720,
                 learnable_tau=False):
        super().__init__()
        self.fast = DecayMemory(tau_fast_hours, learnable_tau)  # short-term bank (~2h)
        self.slow = DecayMemory(tau_slow_hours, learnable_tau)  # long-term bank (~30d)
        # MLP: [z, fast, slow, log1p(Δt)] -> per-dim gate logits; in: 3W+1, out: W
        self.gate = nn.Sequential(
            nn.Linear(3 * width + 1, width),
            nn.GELU(),
            nn.Linear(width, width),
        )
        nn.init.zeros_(self.gate[-1].weight)  # start at even mix (sigmoid(0)=0.5)
        nn.init.zeros_(self.gate[-1].bias)
        self.norm = nn.LayerNorm(width)

    def forward(self, z, time_gap_hours, is_real_event):
        # z, fast, slow: [B, T, W]
        fast = self.fast(z, time_gap_hours, is_real_event)
        slow = self.slow(z, time_gap_hours, is_real_event)
        # log1p compresses skewed gaps so large Δt don't dominate the linear gate; [B,T] -> [B,T,1]
        time = torch.log1p(time_gap_hours.clamp_min(0)).unsqueeze(-1)
        # cat -> [B,T,3W+1]; MLP -> logits [B,T,W]; sigmoid -> soft fast/slow weights in (0,1)
        gate = torch.sigmoid(self.gate(torch.cat([z, fast, slow, time], -1)))
        mixed = gate * fast + (1 - gate) * slow  # [B, T, W] convex blend
        return dict(
            user_z=self.norm(z + mixed) * is_real_event.unsqueeze(-1),  # residual + mask pad
            fast_z=fast, slow_z=slow,
            gate=gate * is_real_event.unsqueeze(-1),  # zero gate on padding
        )