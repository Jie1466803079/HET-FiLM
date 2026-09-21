"""Temporal Text Trajectory (TTT) — analog of EdgeTrajectoryEncoder for text.

Encodes the SEQUENCE of per-fund text projections (abs_proj_out) across the
support window via a GRU, producing temporally-smoothed text embeddings that
are fed to TCMP's per-layer FiLM instead of the instantaneous abs_proj.

Design contract:
  - Input per snapshot: (N_funds_t, text_dim=128) — already produced by
    ProspectusTextFusion.abs_proj and cached in
    MultiTaskEdgePredictor._abs_proj_per_snapshot.
  - Output per snapshot: (N_funds_t, traj_dim) — GRU hidden at time t.
  - The GRU cell is shared across funds and snapshots.

Asymmetric: fund-side only (stocks have no text). At construction time the
GRU is shared across all funds; per-fund state is maintained in a single
tensor and updated at each snapshot.

If snapshots have varying N_funds (e.g., new funds entering across the window),
the encoder uses the smaller N for the GRU state and zero-pads outputs for any
extra rows in larger snapshots — those positions contribute zero TTT signal.
This matches the existing behavior of EdgeTrajectoryEncoder when registry
fund counts vary.
"""
from typing import List, Optional

import torch
import torch.nn as nn


class TextTrajectoryEncoder(nn.Module):
    """GRU over per-fund text projections across the support window.

    encode(abs_proj_per_snapshot) → list of per-snapshot trajectory tensors,
    each shaped (N_funds_t, traj_dim).
    """

    def __init__(self, text_dim: int = 128, traj_dim: int = 128):
        super().__init__()
        self.text_dim = int(text_dim)
        self.traj_dim = int(traj_dim)
        self.gru_cell = nn.GRUCell(input_size=self.text_dim, hidden_size=self.traj_dim)
        print(f"[TTT] Temporal text trajectory encoder enabled "
              f"(text_dim={self.text_dim}, traj_dim={self.traj_dim}); "
              f"GRU updates per-fund text across support snapshots.")

    def encode(self, abs_proj_per_snapshot: List[Optional[torch.Tensor]]
              ) -> List[Optional[torch.Tensor]]:
        """abs_proj_per_snapshot: list of T tensors, each (N_funds_t, text_dim).
        Returns: list of T tensors, each (N_funds_t, traj_dim). None entries
        in the input are passed through as None.
        """
        if not abs_proj_per_snapshot:
            return []
        non_null = [p for p in abs_proj_per_snapshot if p is not None]
        if not non_null:
            return [None] * len(abs_proj_per_snapshot)

        device = non_null[0].device
        # GRU state size: the smallest fund count seen — any rows beyond this in a
        # snapshot don't participate in temporal accumulation (zero-filled output).
        min_n = min(p.shape[0] for p in non_null)
        h = torch.zeros(min_n, self.traj_dim, device=device)

        out: List[Optional[torch.Tensor]] = []
        for ap in abs_proj_per_snapshot:
            if ap is None:
                out.append(None)
                continue
            x_t = ap[:min_n]
            h = self.gru_cell(x_t, h)
            n_t = ap.shape[0]
            if n_t > min_n:
                pad = torch.zeros(n_t - min_n, self.traj_dim, device=device)
                out.append(torch.cat([h, pad], dim=0))
            else:
                out.append(h)
        return out
