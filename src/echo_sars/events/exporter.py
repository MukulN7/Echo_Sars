import json
from pathlib import Path
from typing import List, Union

from echo_sars.events.schema import SAREvent


class EventExporter:
    """Helper to serialize SAREvent instances to JSON or JSON-Lines."""

    @staticmethod
    def to_json(event: SAREvent, indent: int = 2) -> str:
        """Export single event as formatted JSON string."""
        return event.model_dump_json(indent=indent)

    @staticmethod
    def save_event(event: SAREvent, file_path: Union[str, Path]) -> None:
        """Save a single event into a JSON file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(event.model_dump_json(indent=2))

    @staticmethod
    def append_jsonl(event: SAREvent, file_path: Union[str, Path]) -> None:
        """Append event to a JSON-Lines log file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")

