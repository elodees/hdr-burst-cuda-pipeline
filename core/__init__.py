# SCRIPT_NAME: core/__init__.py
"""
Package algorithmique JIT/CUDA pour framework hdr-burst-cuda-pipeline.
Interface d'exposition de l'API publique du dossier core (Nettoyée).
Conforme aux exigences d'intégration modulaire Google (Gcam) & NVIDIA Core Compute.
"""

from .subtract_black_level import subtrack_black_level
from .remove_dead_hot_pixels import remove_dead_hot_pixels
from .cuda_imaging_pipeline import align_burst_frame, robust_temporal_fusion, hdr_local_tone_mapping, Imaging_pipeline_01
from .demosaicing import universal_demosaicing_pipeline

# Exposition propre et complète de l'API (white_balance est définitivement purgé)
__all__ = [
    "subtrack_black_level",
    "remove_dead_hot_pixels",
    "align_burst_frame",
    "robust_temporal_fusion",
    "hdr_local_tone_mapping",
    "Imaging_pipeline_01",
    "universal_demosaicing_pipeline"
]
