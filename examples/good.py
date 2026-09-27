"""Demonstrate small, annotated functions with useful and explicit documentation."""


def normalize_status(status: str) -> str:
    """Map a known processing status to its next workflow state.

    Args:
        status: Current status such as new, queued, running, or failed.

    Returns:
        The next status, or unknown when the input is not recognized.
    """
    transitions = {
        "new": "queued",
        "queued": "running",
        "running": "done",
        "failed": "retry",
    }
    return transitions.get(status, "unknown")


def total_positive_values(values: list[int]) -> int:
    """Add positive input values while ignoring zero and negative entries.

    Args:
        values: Integer measurements supplied by the upstream data source.

    Returns:
        The sum of positive measurements, or zero for an empty input.
    """
    return sum(value for value in values if value > 0)
