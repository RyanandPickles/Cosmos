"""Hook called after each time-windowed log file is closed."""

from pathlib import Path


def process_log_file(log_path: Path) -> None:
    """
    Receive a completed .log file.

    Replace this function body with the Python processing function you provide
    later. This placeholder intentionally does nothing.
    """
    del log_path
