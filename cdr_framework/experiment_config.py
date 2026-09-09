from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AmazonPreprocessingConfig:
    source_domain: str = "Sports_and_Outdoors"
    target_domain: str = "Clothing_Shoes_and_Jewelry"
    raw_dir: Path = Path("data/raw/amazon2014")
    processed_dir: Path = Path("data/processed/sports_to_clothing")
    min_interactions_per_domain: int = 5
    max_text_chars: int = 12000
    seed: int = 42

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_dir", Path(self.raw_dir))
        object.__setattr__(self, "processed_dir", Path(self.processed_dir))

        if self.source_domain != "Sports_and_Outdoors":
            raise ValueError("source_domain must be Sports_and_Outdoors.")
        if self.target_domain != "Clothing_Shoes_and_Jewelry":
            raise ValueError("target_domain must be Clothing_Shoes_and_Jewelry.")
        if self.min_interactions_per_domain < 3:
            raise ValueError("min_interactions_per_domain must be at least 3.")
        if self.max_text_chars <= 0:
            raise ValueError("max_text_chars must be positive.")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AmazonPreprocessingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        dataset = payload.get("dataset", {})
        return cls(**dataset)
