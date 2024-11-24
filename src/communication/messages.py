from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.utils.json import from_optional, from_path, from_str


@dataclass
class YoloMessage:
    img_path: Path
    labels: list[str]
    tgt_path: Path | None = None

    @staticmethod
    def from_dict(obj: Any) -> "YoloMessage":
        assert isinstance(obj, dict)

        img_path = from_path(obj.get("img_path"))
        labels = list(map(lambda x: x.strip(), from_str(obj.get("labels")).split(",")))
        tgt_path = from_optional(from_path, obj.get("tgt_path"))

        return YoloMessage(img_path, labels, tgt_path)

    def to_dict(self) -> dict:
        res = dict(img_path=f"{self.img_path}", labels=",".join(self.labels))
        if self.tgt_path is not None:
            res["tgt_path"] = f"{self.tgt_path}"
        return res
