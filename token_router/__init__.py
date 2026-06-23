"""token-router - a hybrid, token-efficient routing agent.

Each task is routed between a cheap *local* model (Ollama) and a strong
*remote* model (Fireworks AI). The router tries the cheap path first and only
escalates to the remote model when a cheap confidence signal says the local
answer is unreliable - minimizing tokens/cost while holding an accuracy floor.

    from token_router import Agent, RouterPolicy
    from token_router.models import LocalModel, FireworksModel
"""
from .models.base import Model, Usage, ModelError
from .router import RouterPolicy, RouteDecision
from .accounting import Ledger
from .agent import Agent

__version__ = "0.1.0"
__all__ = [
    "Model",
    "Usage",
    "ModelError",
    "RouterPolicy",
    "RouteDecision",
    "Ledger",
    "Agent",
    "__version__",
]
