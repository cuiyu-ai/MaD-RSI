"""Public Model-as-Data algorithm and evaluation primitives."""
from .acquisition import choose_probe
from .adapter import EvidenceConfig, HarnessBackend, prepare_feedback, run_round
from .evidence import build_bundle, validate_analysis_report, validate_digest
from .matrix import response_matrix, response_tensor
from .metrics import compare_harnesses, model_coverage
from .selection import select_queries

__all__ = ["EvidenceConfig", "HarnessBackend", "prepare_feedback", "run_round",
           "response_matrix", "response_tensor", "select_queries", "choose_probe",
           "build_bundle", "validate_digest", "validate_analysis_report",
           "model_coverage", "compare_harnesses"]
