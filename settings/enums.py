# -*- coding: utf-8 -*-
"""
enums.py

Canonical enums used across ASR, text preprocessing, and modeling.

Diagnosis labels (3-way):
    - AD  : Alzheimer (ProbableAD / PossibleAD / AD)
    - HC  : Healthy controls (HC / NC / CTRL / CONTROL)
    - MCI : Mild Cognitive Impairment

Dataset splits:
    - TRAIN / VALID / TEST
"""

from enum import Enum
from typing import Optional
import re

class ADType(str, Enum):
    """Canonical 3-way diagnosis labels: AD / HC / MCI."""
    AD = "AD"
    HC = "HC"
    MCI = "MCI"

    @staticmethod
    def _normalize(label: str) -> str:
        """
        Normalize label strings robustly:
        - upper-case
        - remove whitespace/punctuation (e.g., "Probable AD" -> "PROBABLEAD")
        """
        s = str(label).strip().upper()
        s = re.sub(r"[^A-Z0-9]+", "", s)
        return s

    @classmethod
    def from_any(cls, label: Optional[str]) -> "ADType":
        """Map a raw diagnosis string from any dataset to a canonical ADType."""
        if label is None:
            raise ValueError("Diagnosis label is None")

        s = cls._normalize(label)

        # AD family
        if s in {"AD", "PROBABLEAD", "POSSIBLEAD"}:
            return cls.AD

        # HC family (common variants across datasets)
        if s in {"HC", "NC", "CN", "CTRL", "CONTROL", "CONTROLS", "HEALTHY", "NORMAL"}:
            return cls.HC

        # MCI family
        if s in {"MCI", "MILDCOGNITIVEIMPAIRMENT"}:
            return cls.MCI

        raise ValueError(f"Unknown diagnosis label: {label!r}")

class DatasetSplit(str, Enum):
    """Standard split names for train/valid/test."""
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
