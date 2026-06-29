"""Model backends behind one uniform interface.

Everything in the router/agent depends on ``Model`` (see ``base.py``) and never
on a concrete backend, so models are swappable and trivially mockable in tests.
"""
from .base import Model, Usage, ModelError, count_tokens
from .mock import ScriptedModel
from .fireworks import FireworksModel
from .local import LocalModel

__all__ = [
    "Model",
    "Usage",
    "ModelError",
    "count_tokens",
    "ScriptedModel",
    "FireworksModel",
    "LocalModel",
]
