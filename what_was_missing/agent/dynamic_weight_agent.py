"""
DynamicWeightAgent: context-conditioned weights for the Aggressiveness Index.

RECREATED BY CLAUDE (Anthropic), September 2026. Written from the descriptions
in the VIPP 201A final report (Sections 2.1, 2.2, 3.1, 3.3) and weekly reports
2, 3 and 11. This is not the original project code and was not used to produce
any result in those reports.

Two heads share one interface, forward(features, context) -> (weights, mixture):

  MLPHead   9 inputs (4 kinematic + 5 context) -> 64 -> 64 -> 4, softmax.
            mixture is returned as zeros (no gating).
  GateHead  a 2-layer MLP maps the 5-dim context to a 3-way softmax over the
            highway / urban / weather regimes; the weights are the convex
            combination of three learnable regime embeddings, each kept on
            the simplex through a softmax over logits initialized at the
            regime targets.

Context vector (5): [is_highway, is_urban, density, friction, slip], following
the final report's description of a road-type one-hot (highway / urban) plus
normalized traffic density, friction coefficient and slip ratio. The weekly
report for week 2 mentions an 8-dim input instead; the 9-dim version from the
final report is used here.

Loss (final report, Section 3.1):
  loss = alpha * KL(pred || target) + (1 - alpha) * (-E[1 - AI(state, pred)])
with alpha annealed from 1.0 to 0.5 over the first 50 epochs, then held.

Note on the RL term: taken literally, maximizing 1 - AI has a trivial optimum,
putting all weight on whichever normalized feature is smallest. The KL anchor
at alpha >= 0.5 is what keeps the weights near the regime targets, so drift
away from the targets should be read with that in mind.
"""
import json
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from agent.aggressiveness import (  # noqa: E402  torch-free shared definitions
    AGGRESSIVE_FROM, CONTEXT, FEATURES, REGIMES, category, normalize_features,
)
from agent.aggressiveness import REGIME_TARGETS as _TARGETS

# (3, 4) tensor in REGIMES order
REGIME_TARGETS = torch.tensor([_TARGETS[r] for r in REGIMES])


def ai_score(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Aggressiveness Index in [0, 100]: weighted sum of normalized features."""
    return 100.0 * (features * weights).sum(-1).clamp(0.0, 1.0)


class MLPHead(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATURES) + len(CONTEXT), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, len(FEATURES)),
        )

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        weights = F.softmax(self.net(torch.cat([features, context], -1)), -1)
        mixture = torch.zeros(weights.shape[:-1] + (len(REGIMES),), dtype=weights.dtype)
        return weights, mixture


class GateHead(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(len(CONTEXT), hidden), nn.ReLU(),
            nn.Linear(hidden, len(REGIMES)),
        )
        self.regime_logits = nn.Parameter(REGIME_TARGETS.clone().log())

    @torch.jit.export
    def regime_weights(self) -> torch.Tensor:
        """(3, 4): one weight vector per regime, each on the simplex."""
        return F.softmax(self.regime_logits, -1)

    def forward(self, features: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mixture = F.softmax(self.gate(context), -1)
        weights = mixture @ self.regime_weights()
        return weights, mixture


class DynamicWeightAgent(nn.Module):
    """Wrapper that picks a head and adds the AI computation."""

    def __init__(self, head: str = "gate"):
        super().__init__()
        if head not in ("gate", "mlp"):
            raise ValueError("head must be 'gate' or 'mlp'")
        self.head_name = head
        self.head = GateHead() if head == "gate" else MLPHead()

    def forward(self, features, context):
        return self.head(features, context)

    def score(self, features, context):
        weights, mixture = self.head(features, context)
        return ai_score(features, weights), weights, mixture


def alpha_schedule(epoch: int, anneal_epochs: int = 50, floor: float = 0.5) -> float:
    if epoch >= anneal_epochs:
        return floor
    return 1.0 - (1.0 - floor) * epoch / anneal_epochs


def kl_divergence(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """KL(pred || target), averaged over the batch."""
    return (pred * (pred.clamp_min(eps).log() - target.clamp_min(eps).log())).sum(-1).mean()


def agent_loss(pred_w, target_w, features, alpha, crashed=None):
    """Returns (total, kl, rl). The RL term only uses non-crash samples."""
    kl = kl_divergence(pred_w, target_w)
    ai = ai_score(features, pred_w) / 100.0
    if crashed is not None:
        ai = ai[~crashed]
    rl = -(1.0 - ai).mean() if ai.numel() else torch.zeros(())
    return alpha * kl + (1.0 - alpha) * rl, kl, rl


def export_gate(head: GateHead, path: str, provenance: dict) -> None:
    """Save as TorchScript so the ROS2 node can load it without this module."""
    scripted = torch.jit.script(head.eval())
    torch.jit.save(scripted, path, _extra_files={"provenance.json": json.dumps(provenance, indent=2)})


def load_gate(path: str):
    """Load a TorchScript gate head and its provenance record."""
    extra = {"provenance.json": ""}
    module = torch.jit.load(path, _extra_files=extra)
    module.eval()
    return module, json.loads(extra["provenance.json"] or "{}")
