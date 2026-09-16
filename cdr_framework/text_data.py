from dataclasses import dataclass
from pathlib import Path
import json
import torch
from cdr_framework.formal_recommendation import RecommendationRow, collate_recommendation_rows


@dataclass(frozen=True)
class TextRow:
    user_id: int
    source_items: tuple
    target_items: tuple
    positive_target_item: int
    seen_items: frozenset


def load_text_rows(path: Path, max_length: int, num_items: int, target_ids: torch.Tensor):
    targets = set(target_ids.tolist())
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            source, target = tuple(row["source_items"]), tuple(row["target_items"])
            positive = row["positive_target_item"]
            if not source or not target or positive not in targets:
                raise ValueError(f"Invalid histories or label at {path}:{number}")
            if any(not isinstance(i, int) or not 0 < i < num_items or i in targets for i in source):
                raise ValueError(f"Invalid source ID at {path}:{number}")
            if any(i not in targets for i in target):
                raise ValueError(f"Invalid target ID at {path}:{number}")
            rows.append(TextRow(int(row["user_id"]), source[-max_length:], target[-max_length:], positive, frozenset(target)))
    if not rows:
        raise ValueError(f"Empty split: {path}")
    return rows


def collate_text_rows(rows):
    batch = collate_recommendation_rows([
        RecommendationRow(row.source_items, row.target_items, row.positive_target_item) for row in rows
    ])
    return batch, [set(row.seen_items) for row in rows]
