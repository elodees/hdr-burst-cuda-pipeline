# SCRIPT_NAME: utils/__init__.py
"""
Package d'outils utilitaires bas niveau pour le framework hdr-burst-cuda-pipeline.
Gère l'ingestion, le parsing des métadonnées EXIF/CFA et l'interfaçage matériel.
Conforme aux exigences d'intégration modulaire Google (Gcam) & NVIDIA Core Compute.
"""

from .raw_extractor import extract_raw_metadata
from .Matlab_reader_01 import matlab_reader_01
from .RAW_2D_or_3D_export_to_MATLAB_file_01 import RAW_2D_or_3D_export_to_matlab_file

# Définition rigoureuse de l'interface publique (API publique du package utils)
__all__ = [
    "extract_raw_metadata",
    "matlab_reader_01",
    "RAW_2D_or_3D_export_to_matlab_file"
]
