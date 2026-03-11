"""
Interactive real-time environment viewer.

Core logic is abstracted in BaseInteractiveViewer; per-organism
implementations (e.g. CElegansInteractiveViewer) handle display
and interaction specifics.
"""

from simulations.interactive.base import BaseInteractiveViewer

__all__ = ["BaseInteractiveViewer"]
