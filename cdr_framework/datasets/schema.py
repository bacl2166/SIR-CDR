from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class InteractionRecord:
    user_id: int | str
    item_id: int | str
    domain: str
    timestamp: int | float
    rating: float | None = None
    review_text: str | None = None
    behavior_type: str = "interaction"


@dataclass(frozen=True)
class ItemMetadataRecord:
    item_id: int | str
    domain: str
    title: str
    category: tuple[str, ...] = ()
    brand_or_creator: str | None = None
    description: str | None = None
    image_ref: str | None = None
    attributes: Mapping[str, str | int | float | bool | None] = field(default_factory=dict)


@dataclass(frozen=True)
class CrossDomainSequence:
    user_id: int | str
    source_domain: str
    target_domain: str
    source_items: tuple[int | str, ...]
    target_items: tuple[int | str, ...]
    positive_target_item: int | str

    def all_history(self) -> tuple[int | str, ...]:
        return self.source_items + self.target_items
