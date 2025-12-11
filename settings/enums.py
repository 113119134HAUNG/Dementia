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

class ADType(str, Enum):
    AD = "AD"
    HC = "HC"
    MCI = "MCI"

    @classmethod
    def from_any(cls, label: Optional[str]) -> "ADType":
        """Map a raw diagnosis string from any dataset to a canonical ADType.

        Examples
        --------
        >>> ADType.from_any("ProbableAD")
        <ADType.AD: 'AD'>
        >>> ADType.from_any("CTRL")
        <ADType.HC: 'HC'>
        >>> ADType.from_any("mci")
        <ADType.MCI: 'MCI'>

        Raises
        ------
        ValueError
            If the input label cannot be mapped.
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
    TEST = "test"
