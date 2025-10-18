"""
Top-level package marker for the Aldar Köse ML project.

This file ensures `ml` is recognized as a Python package so imports like
`from ml.src.diffusion.pipeline import StoryboardGenerationPipeline` work in tests and scripts.
"""
__all__ = ["src"]
