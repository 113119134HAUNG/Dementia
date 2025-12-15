# -*- coding: utf-8 -*-
"""
enums.py (paper-strict, converged)

Canonical enums used across ASR, preprocessing, and modeling.

Diagnosis labels (3-way):
  - AD
  - HC
  - MCI

Policy
------
- ADType.from_any() is the canonical normalization for common label variants.
- YAML text.label_map is allowed ONLY as a dataset-specific override layer
  (e.g., if a dataset has unusual labels not covered by ADType).
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Any
import re

# compile once (paper-strict: deterministic + minimal overhead)
_NON_ALNUM = re.compile(r"[^A-Z0-9]+")

# canonical mapping sets (single point)
_AD_SET = frozenset({"AD", "PROBABLEAD", "POSSIBLEAD"})
_HC_SET = frozenset({"HC", "NC", "CN", "CTRL", "CONTROL", "CONTROLS", "HEALTHY", "NORMAL"})
_MCI_SET = frozenset({"MCI", "MILDCOGNITIVEIMPAIRMENT"})

class ADType(str, Enum):
    """Canonical 3-way diagnosis labels: AD / HC / MCI."""
    AD = "AD"
    HC = "HC"
    MCI = "MCI"

    @staticmethod
    def _normalize(label: Any) -> str:
        """Normalize raw label robustly into compact upper alnum token."""
        s = str(label).strip().upper()
        s = _NON_ALNUM.sub("", s)
        return s

    @classmethod
    def from_any(cls, label: Optional[Any]) -> "ADType":
        """Map a raw diagnosis string from any dataset to canonical ADType."""
        if label is None:
            raise ValueError("Diagnosis label is None")

        s = cls._normalize(label)
        if not s:
            raise ValueError("Diagnosis label is empty")

        if s in _AD_SET:
            return cls.AD
        if s in _HC_SET:
            return cls.HC
        if s in _MCI_SET:
            return cls.MCI

        raise ValueError(f"Unknown diagnosis label: {label!r}")

    @classmethod
    def try_from_any(cls, label: Optional[Any]) -> Optional["ADType"]:
        """Return ADType if known else None (avoid scattered try/except)."""
        try:
            return cls.from_any(label)
        except Exception:
            return None

class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
