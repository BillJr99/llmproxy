"""Dialect translation layer.

Importing this package registers every built-in adapter. Use ``get_inbound`` /
``get_outbound`` to look up adapters by client dialect / provider protocol.
"""

from __future__ import annotations

# Importing each module registers its adapters in the registry.
from . import (
    anthropic,  # noqa: F401,E402
    gemini,  # noqa: F401,E402
    openai,  # noqa: F401,E402
)
from .base import (  # noqa: F401
    InboundAdapter,
    OutboundAdapter,
    get_inbound,
    get_outbound,
    register_inbound,
    register_outbound,
)

__all__ = [
    "InboundAdapter",
    "OutboundAdapter",
    "get_inbound",
    "get_outbound",
    "register_inbound",
    "register_outbound",
]
