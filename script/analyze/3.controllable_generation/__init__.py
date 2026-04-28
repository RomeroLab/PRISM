"""
Controllable Generation Experiment Module

This module implements controllable antibody sequence generation using PRISM v2's
factorized head architecture, enabling region-specific control over mutation patterns.

Components:
- PRISMGenerator: Controllable generation with GL/NGL/Final/Region-specific modes
- BaselineGenerator: Standard generation with ESM2, AbLang2, etc.
- GenerationEvaluator: Comprehensive evaluation metrics
- run_controllable_generation: Main experiment runner
- visualize_generation_results: Result visualization

Key Innovation:
PRISM's factorized heads enable controllable generation:
- GL mode: Conservative, germline-preserving generation
- NGL mode: Diverse, mutation-permissive generation
- Final mode: Balanced, α-weighted generation
- Region-specific: FR→GL (stable), CDR→NGL (explorative)

Usage:
    from script.analyze.controllable_generation import (
        PRISMGenerator,
        BaselineGenerator,
        GenerationEvaluator,
    )
"""

from .prism_generator import PRISMGenerator, load_prism_model, GenerationResult
from .baseline_generator import BaselineGenerator, create_baseline_generator
from .generation_evaluator import GenerationEvaluator, EvaluationMetrics

__all__ = [
    'PRISMGenerator',
    'BaselineGenerator',
    'GenerationEvaluator',
    'load_prism_model',
    'create_baseline_generator',
    'GenerationResult',
    'EvaluationMetrics',
]
