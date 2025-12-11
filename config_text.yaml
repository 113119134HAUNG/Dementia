  # -*- coding: utf-8 -*-
"""
enums.py
這些定義在自動語音辨識、文字預處理和建模程式碼中均有使用，以確保標籤的一致性和可重複性。
統一診斷標籤：重度、中度、輕度
- AD  : Alzheimer / ProbableAD / PossibleAD
- HC  : Healthy Control / NC / CTRL / Control
- MCI : Mild Cognitive Impairment
"""

from enum import Enum
from typing import Optional

class ADType(str, Enum):
    AD = "AD"
    HC = "HC"
    MCI = "MCI"

    @classmethod
    def from_any(cls, label: Optional[str]) -> "ADType":
        """Map a raw diagnosis string from any dataset to a canonical ADType.

        Examples:
            ADType.from_any("ProbableAD")  -> ADType.AD
            ADType.from_any("CTRL")       -> ADType.HC
            ADType.from_any("mci")        -> ADType.MCI

        Raises:
            ValueError: if the input label cannot be mapped.
        """
        if label is None:
            raise ValueError("Diagnosis label is None")

        s = str(label).strip().upper()

        # AD family
        if s in {"AD", "PROBABLEAD", "POSSIBLEAD"}:
            return cls.AD

        # HC family
        if s in {"HC", "NC", "CTRL", "CONTROL"}:
            return cls.HC

        # MCI family
        if s in {"MCI"}:
            return cls.MCI

        raise ValueError(f"Unknown diagnosis label: {label!r}")

class DatasetSplit(str, Enum):
    """Standard split names for train/valid/test."""
    TRAIN = "train"
    VALID = "valid"
    TEST  = "test"
