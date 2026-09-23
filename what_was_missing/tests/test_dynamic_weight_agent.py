import os

import torch

from agent.dynamic_weight_agent import (
    REGIME_TARGETS, DynamicWeightAgent, GateHead, agent_loss, alpha_schedule, export_gate,
    kl_divergence, load_gate,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def test_heads_output_simplex():
    torch.manual_seed(0)
    f, c = torch.rand(32, 4), torch.rand(32, 5)
    for head in ("mlp", "gate"):
        w, mix = DynamicWeightAgent(head)(f, c)
        assert w.shape == (32, 4)
        assert torch.allclose(w.sum(-1), torch.ones(32), atol=1e-6)
        assert (w >= 0).all()
        if head == "gate":
            assert torch.allclose(mix.sum(-1), torch.ones(32), atol=1e-6)


def test_gate_starts_at_regime_targets():
    assert torch.allclose(GateHead().regime_weights(), REGIME_TARGETS, atol=1e-6)


def test_alpha_schedule():
    assert alpha_schedule(0) == 1.0
    assert abs(alpha_schedule(25) - 0.75) < 1e-9
    assert alpha_schedule(50) == 0.5 and alpha_schedule(199) == 0.5


def test_kl_zero_at_target_and_rl_skips_crashes():
    t = REGIME_TARGETS[:1].repeat(4, 1)
    assert kl_divergence(t, t).item() < 1e-7
    f = torch.tensor([[1.0, 1, 1, 1]] * 2 + [[0.0, 0, 0, 0]] * 2)
    crashed = torch.tensor([False, False, True, True])
    _, _, rl = agent_loss(t, t, f, 0.5, crashed)
    assert abs(rl.item()) < 1e-6          # only the two AI = 100 rows count, 1 - AI = 0


def test_export_roundtrip(tmp_path):
    head = GateHead()
    path = str(tmp_path / "g.pt")
    export_gate(head, path, {"created_by": "test"})
    m, prov = load_gate(path)
    f, c = torch.rand(3, 4), torch.rand(3, 5)
    assert torch.allclose(m(f, c)[0], head(f, c)[0])
    assert prov["created_by"] == "test"


def test_shipped_checkpoints_carry_provenance():
    for name in ("gate_v2.pt", "gate_anchor_only.pt"):
        _, prov = load_gate(os.path.join(ROOT, "model", name))
        assert prov["not_the_original"] is True
        assert "Claude" in prov["created_by"]
