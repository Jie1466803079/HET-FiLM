import os

if os.environ.get("SKIP_HEAVY_MODEL_IMPORTS", "0") != "1":
    from .GCN import GCN
    from .GAT import GAT
    from .RGCN import RGCN
    from .HGT import HGT
    from .HAN import HAN
    from .DySAT import DySAT
    from .DyHATR import DyHATR
    from .HTGNN import HTGNN
    from .DHSpace import DHSpace, DHNet
    from .DHSpace_GRU import DHSpace_GRU, DHNet_GRU
    from .DHSpaceGR import DHSpaceGR
    from .DHSpaceEW import DHSpaceEW
    from .DHSpaceMP import DHSpaceMP
    from .DHSpaceMPNAS import DHSpaceMPNAS
    from .DHSpaceMGNAS import DHSpaceMGNAS
    from .DHSpaceNodeGRU import DHSpaceNodeGRU
    from .DHSpaceMeta import DHSpaceMeta
    from .DHSpaceMGCell import DHSpaceMGCell
    from .DHSpaceMGGlobal import DHSpaceMGGlobal
    from .DHSpaceSearch import DHSearcher
    from .ETTE import ETTE
else:
    # Minimal imports to allow edge weight runner
    from .HGT import HGT
try:
    from .load_model import load_model
except Exception:
    load_model = None
from .multitask_edge import MultiTaskEdgePredictor
from .multitask_edge_adaptive import MultiTaskEdgePredictorAdaptive

Sta_MODEL = "GCN GAT RGCN HAN HGT HGT+".split()
Homo_MODEL = "GCN GAT RGCN DySAT".split()
