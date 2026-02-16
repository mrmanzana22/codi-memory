"""Lightweight request tracing via contextvars (stdlib)."""
import uuid
from contextvars import ContextVar

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def new_trace_id() -> str:
    """Generate and set a new trace_id for this call stack."""
    tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    """Get current trace_id (empty string if none set)."""
    return _trace_id.get()


def set_trace_id(tid: str) -> None:
    """Manually set trace_id (for testing)."""
    _trace_id.set(tid)
