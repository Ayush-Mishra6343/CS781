"""Common reusable modules across all seven phases of In-House Legal Intelligence System."""

from .metrics import compute_macro_f1, compute_pcr_f1_at_k
from .legal_losses import AsymmetricFocalLoss, DistillationLoss
from .codabench_packager import package_submission
from .character_boundary_mapper import align_subwords_to_character_spans
from .peft_modules import LoRALinear, apply_bitfit
from .swiglu_attention import SwiGLU, LocalGlobalDualAttention
from .hybrid_legal_lm import HybridLegalLM
from .transformer_decoder import LegalTransformerDecoder

__all__ = [
    "compute_macro_f1",
    "compute_pcr_f1_at_k",
    "AsymmetricFocalLoss",
    "DistillationLoss",
    "package_submission",
    "align_subwords_to_character_spans",
    "LoRALinear",
    "apply_bitfit",
    "SwiGLU",
    "LocalGlobalDualAttention",
    "HybridLegalLM",
    "LegalTransformerDecoder",
]
