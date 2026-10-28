import torch
from src.models.temporal_state import DecayMemory, DualTimescaleState, SingleTimescaleState


def test_padding_does_not_update_internal_state():
    memory = DecayMemory(2)
    compact = memory(torch.tensor([[[1.], [3.]]]),
                     torch.tensor([[0., 2.]]), torch.tensor([[True, True]]))
    # The last gap remains elapsed time since the preceding REAL event.
    padded = memory(torch.tensor([[[1.], [999.], [3.]]]),
                    torch.tensor([[0., 100., 2.]]), torch.tensor([[True, False, True]]))
    assert padded[0, 1, 0] == 0
    torch.testing.assert_close(padded[:, [0, 2]], compact)


def test_large_gap_moves_fast_closer_to_current_event():
    z = torch.tensor([[[1.], [3.]]])
    gap = torch.tensor([[0., 24.]])
    real = torch.ones(1, 2, dtype=torch.bool)
    fast, slow = DecayMemory(2)(z, gap, real), DecayMemory(720)(z, gap, real)
    assert abs(fast[0, 1, 0] - 3) < abs(slow[0, 1, 0] - 3)


def test_neutral_gate_and_padding_outputs():
    module = DualTimescaleState(4)
    z = torch.randn(2, 3, 4)
    gap = torch.tensor([[0., 1., 24.], [0., 2., 0.]])
    real = torch.tensor([[True, True, True], [True, True, False]])
    out = module(z, gap, real)
    torch.testing.assert_close(out["gate"][real], torch.full_like(out["gate"][real], 0.5))
    assert out["user_z"].shape == z.shape
    for key in ("user_z", "fast_z", "slow_z", "gate"):
        assert torch.equal(out[key][~real], torch.zeros_like(out[key][~real]))


def test_future_mutation_does_not_change_prefix():
    module = DualTimescaleState(4).eval()
    z = torch.randn(1, 4, 4)
    changed = z.clone()
    changed[:, 3] = 99
    gap = torch.tensor([[0., 1., 2., 3.]])
    real = torch.ones(1, 4, dtype=torch.bool)
    torch.testing.assert_close(module(z, gap, real)["user_z"][:, :3],
                               module(changed, gap, real)["user_z"][:, :3])


def test_equal_fixed_timescales_cancel_gate():
    dual = DualTimescaleState(4, 2, 2)
    single = SingleTimescaleState(4, 2)
    single.norm.load_state_dict(dual.norm.state_dict())
    # Make gates non-neutral; output should still equal the single-memory path.
    with torch.no_grad():
        dual.gate[-1].weight.normal_()
        dual.gate[-1].bias.fill_(2)
    z = torch.randn(2, 3, 4)
    gap = torch.tensor([[0., 1., 24.], [0., 2., 48.]])
    real = torch.ones(2, 3, dtype=torch.bool)
    torch.testing.assert_close(dual(z, gap, real)["user_z"], single(z, gap, real)["user_z"])


def test_first_event_initializes_and_tied_timestamp_updates():
    memory = DecayMemory(2)
    z = torch.tensor([[[1.], [3.]]])
    out = memory(z, torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool))
    assert out[0, 0, 0] == 1
    assert 1 < out[0, 1, 0] < 3
