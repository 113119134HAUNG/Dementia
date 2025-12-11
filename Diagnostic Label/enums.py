# enums.py
from enum import Enum
from typing import Optional

# 統一的診斷標籤
#AD : Alzheimer / ProbableAD / PossibleAD
#HC : Healthy control / NC / CTRL / Control
#MCI: Mild Cognitive Impairment
class ADType(str, Enum):
    AD = "AD"
    HC = "HC"
    MCI = "MCI"

   # 把資料集的診斷字串，轉成標準
    @classmethod
    def from_any(cls, label: Optional[str]) -> "ADType":
      
        if label is None:
            raise ValueError("Diagnosis label is None")

        s = str(label).strip().upper()

        # AD 系列
        if s in {"AD", "PROBABLEAD", "POSSIBLEAD"}:
            return cls.AD

        # HC 系列
        if s in {"HC", "NC", "CTRL", "CONTROL"}:
            return cls.HC

        # MCI 系列
        if s in {"MCI"}:
            return cls.MCI

        raise ValueError(f"Unknown diagnosis label: {label}")

# 資料集切分標記
class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
