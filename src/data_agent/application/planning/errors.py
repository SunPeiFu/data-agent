"""Application-level errors exposed by the planning workflow."""


class ClarificationProtocolError(ValueError):
    """Raised when a clarification response is stale, mismatched, or unauthorized."""
