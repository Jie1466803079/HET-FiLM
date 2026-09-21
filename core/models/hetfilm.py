"""HET-FiLM model: entry point to its components.

HET-FiLM is assembled from the modules below; ``scripts/run/run_model.py``
builds it from the flags in ``configs/us/us_hetfilm.pbs`` and
``configs/canada/canada_hetfilm.pbs``.

1. Prospectus-aware fund encoding (``ProspectusTextFusion``): gates a 128-d
   projection of the prospectus embedding against the projected numerical
   fund features and replaces the fund node features in every snapshot
   (``MultiTaskEdgePredictor._apply_prospectus_fusion``).
2. Type-indexed attention encoder (``HGT``, ``HGTConv``): multi-head
   attention whose projections are indexed by meta-relation (the
   formulation of Hu et al., 2020), applied to each quarterly snapshot and
   pooled over the support window by temporal self-attention.
3. Edge-trajectory-conditioned message passing: ``EdgeTrajectoryEncoder``
   runs a causal GRU over each persistent edge's ``(w_t, dw_t, present_t)``
   history, and ``EdgeTrajectoryFiLM`` maps the final state to per-edge
   ``(gamma, beta)``, which ``HGTConv.message`` applies to the value path.
   ``MultiTaskEdgePredictor.configure_edge_trajectory`` builds both modules.

``MultiTaskEdgePredictor`` runs two-stage training: Stage 1 link prediction
trains the encoder, and Stage 2 freezes it and fits the weight head.

In the run scripts, ``--model HGT+`` is the internal key that selects the
attention encoder, and ``--use_prospectus`` and ``--use_edge_trajectory``
switch on modules 1 and 3. The Canadian configuration also passes
``--use_tcetf``, which enables the text-conditioned branch of
``EdgeTrajectoryFiLM``, and ``--grad_clip 1.0``.
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
