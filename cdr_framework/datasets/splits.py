from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Iterable

from cdr_framework.datasets.schema import InteractionRecord


@dataclass(frozen=True)
class ChronologicalSplit:
    train: tuple[InteractionRecord, ...]
    validation: tuple[InteractionRecord, ...]
    test: tuple[InteractionRecord, ...]


def chronological_leave_one_out(records: Iterable[InteractionRecord]) -> ChronologicalSplit:
    by_user: dict[int | str, list[InteractionRecord]] = defaultdict(list)
    for record in records:
        by_user[record.user_id].append(record)

    train: list[InteractionRecord] = []
    validation: list[InteractionRecord] = []
    test: list[InteractionRecord] = []

    for user_id in sorted(by_user, key=str):
        ordered = sorted(by_user[user_id], key=lambda record: (record.timestamp, str(record.item_id)))
        if len(ordered) < 3:
            raise ValueError(f"user {user_id!r} needs at least 3 interactions for leave-one-out splitting.")
        train.extend(ordered[:-2])
        validation.append(ordered[-2])
        test.append(ordered[-1])

    return ChronologicalSplit(tuple(train), tuple(validation), tuple(test))
