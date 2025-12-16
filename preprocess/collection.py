# -*- coding: utf-8 -*-
"""
collection.py

Paper-style utilities:
- No CLI
- Deterministic JSONL merge order
- Validates each JSONL line is JSON (and re-dumps) to ensure merged JSONL is clean
- Dataset collection abstractions (Collection, NormalizedDataPoint)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

RawDataPoint = Dict[str, Any]

@dataclass
class NormalizedDataPoint:
    PID: str
    Languages: str
    MMSE: Any
    Diagnosis: str
    Participants: Any
    Dataset: str
    Modality: str
    Task: List[str]
    File_ID: str
    Media: str
    Age: Any
    Gender: str
    Education: Any
    Source: str
    Continents: Any
    Countries: Any
    Duration: Any
    Location: str
    Date: Any
    Transcriber: Any
    Moca: Any  # keep only once
    Setting: Any
    Comment: Any
    Text_interviewer_participant: str
    Text_participant: str
    Text_interviewer: str

class Collection(ABC):
    def __init__(self, path: str, language: str) -> None:
        self.path = path
        self.language = language.lower()

    @abstractmethod
    def __iter__(self) -> Iterator[RawDataPoint]:
        raise NotImplementedError

    @abstractmethod
    def normalize_datapoint(self, raw_datapoint: RawDataPoint) -> NormalizedDataPoint:
        raise NotImplementedError

    def get_normalized_data(self) -> Iterator[NormalizedDataPoint]:
        for raw_datapoint in self:
            yield self.normalize_datapoint(raw_datapoint)

class JSONLCombiner:
    """
    Minimal JSONL combiner (paper-style):
    - Deterministic order (sorted paths)
    - Validates each non-empty line is JSON
    - Re-dumps JSON to ensure clean JSONL output
    - Streams line-by-line (does not load everything into memory)

    Backward-compat:
    - Also accepts legacy kwargs: output_directory, output_filename
    """

    def __init__(
        self,
        input_files: List[str],
        output_dir: Optional[Union[str, Path]] = None,
        merged_name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        # legacy arg names support
        if output_dir is None:
            output_dir = kwargs.pop("output_directory", None)
        if merged_name is None:
            merged_name = kwargs.pop("output_filename", None)
        if kwargs:
            raise TypeError(f"Unexpected kwargs: {sorted(kwargs.keys())}")

        if output_dir is None or merged_name is None:
            raise TypeError("JSONLCombiner requires output_dir and merged_name (or legacy output_directory/output_filename).")

        self.input_files = [str(x) for x in (input_files or [])]
        self.output_dir = Path(str(output_dir))
        self.merged_name = str(merged_name)

    def _iter_json_records(self, path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                s = line.strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception as e:
                    raise ValueError(f"Bad JSON at {path}:{ln}: {repr(e)}") from e

                if isinstance(obj, dict):
                    yield obj
                else:
                    # allow non-dict JSON; wrap so each line is an object
                    yield {"_value": obj}

    def combine(self) -> str:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / self.merged_name

        if not self.input_files:
            raise ValueError("JSONLCombiner.input_files is empty.")

        paths: List[Path] = []
        for fp in self.input_files:
            p = Path(fp)
            if not p.exists():
                raise FileNotFoundError(f"Input JSONL not found: {p}")
            paths.append(p)

        # stable deterministic order
        paths = sorted(paths, key=lambda x: str(x))

        with out_path.open("w", encoding="utf-8") as w:
            for p in paths:
                for rec in self._iter_json_records(p):
                    w.write(json.dumps(rec, ensure_ascii=False) + "\n")

        return str(out_path)
