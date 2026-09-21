"""HET-FiLM: entry point to the model's components.

HET-FiLM is not a separate network class. It is the graph transformer
encoder in ``core/models/HGT.py`` with two add-on modules, switched on by
CLI flags and attached by ``scripts/run/run_model.py``:

1. Prospectus-aware fund encoding (``--use_prospectus``):
   ``ProspectusTextFusion`` gates a 128-d projection of the prospectus
   embedding against the projected numerical fund features and replaces the
   fund node features in every snapshot
   (``MultiTaskEdgePredictor._apply_prospectus_fusion``).
2. Edge-trajectory-conditioned message passing (``--use_edge_trajectory``):
   ``EdgeTrajectoryEncoder`` runs a causal GRU over each persistent edge's
   ``(w_t, dw_t, present_t)`` history, and ``EdgeTrajectoryFiLM`` maps the
   final state to per-edge ``(gamma, beta)``, which ``HGTConv.message``
   applies to the value path. ``MultiTaskEdgePredictor.configure_edge_trajectory``
   builds both modules.

``MultiTaskEdgePredictor`` wraps the backbone for two-stage training: Stage 1
link prediction trains the encoder, and Stage 2 freezes it and fits the weight
head.

The Canadian configuration also passes ``--use_tcetf``, which enables the
text-conditioned branch of ``EdgeTrajectoryFiLM``, and ``--grad_clip 1.0``.
The exact flags for each panel are in ``configs/us/us_hetfilm.pbs`` and
``configs/canada/canada_hetfilm.pbs``.
"""
from .HGT import HGT, HGTConv
from .edge_trajectory import EdgeTrajectoryEncoder, EdgeTrajectoryFiLM
from .prospectus_fusion import ProspectusTextFusion
from .multitask_edge import MultiTaskEdgePredictor

__all__ = [
    "HGT",
    "HGTConv",
    "EdgeTrajectoryEncoder",
    "EdgeTrajectoryFiLM",
    "ProspectusTextFusion",
    "MultiTaskEdgePredictor",
]
