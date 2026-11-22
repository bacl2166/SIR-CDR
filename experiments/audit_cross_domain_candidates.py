"""Download and audit two Amazon2014 pairs under SIR-CDR's existing row rules."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework.datasets.amazon2014 import (
    AMAZON_2014_BASE_URL, _download_file, category_files,
    iter_amazon_records, sha256_file,
)

PAIRS = {
    "food_to_kitchen": ("Grocery_and_Gourmet_Food", "Home_and_Kitchen"),
    "beauty_to_electronics": ("Beauty", "Electronics"),
}


def direction_statistics(source, target):
    """Match build_shared's population, strict source timing and all-prefix rows."""
    users = sorted(u for u in source.keys() & target.keys()
                   if min(len(source[u]), len(target[u])) >= 5)
    source_items = {item for user in users for _, item, _ in source[user]}
    target_items = {item for user in users for _, item, _ in target[user]}
    stats = dict(users=len(users), source_items=len(source_items),
                 target_items=len(target_items),
                 cross_domain_shared_asins=len(source_items & target_items),
                 source_interactions=sum(len(source[u]) for u in users),
                 target_interactions=sum(len(target[u]) for u in users))
    for split in ("train", "validation", "test"):
        stats[f"{split}_rows"] = 0
        stats[f"{split}_positives_in_seen"] = 0
    for user in users:
        first_source = source[user][0][0]
        events = target[user]
        history = set()
        for index, (timestamp, item, _) in enumerate(events):
            if index:
                split = ("test" if index == len(events) - 1 else
                         "validation" if index == len(events) - 2 else "train")
                if first_source < timestamp:
                    stats[f"{split}_rows"] += 1
                    stats[f"{split}_positives_in_seen"] += item in history
            history.add(item)
    return stats


def read_domain(path):
    users = defaultdict(list)
    invalid = 0
    for number, row in enumerate(iter_amazon_records(path), 1):
        user, item, stamp = row.get("reviewerID"), row.get("asin"), row.get("unixReviewTime")
        try:
            if not user or not item or isinstance(stamp, bool):
                raise ValueError("Missing ID or invalid timestamp")
            timestamp = int(stamp)
        except (TypeError, ValueError):
            invalid += 1
            continue
        users[str(user)].append((timestamp, str(item), f"{number:012d}"))
    for events in users.values():
        events.sort()
    return users, invalid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Data/lihb/SIR-CDR"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    raw = args.root.resolve() / "data/raw/amazon2014"
    report = {"schema": "cross-domain-candidate-audit-v1", "files": {}, "directions": {},
              "protocol": "common users >=5/domain; last-two holdouts/domain; all-prefixes; strict source time; repeats retained",
              "selection": "Food-Kitchen and Beauty-Electronics chosen before recommendation metric evaluation",
              "tool_sha256": sha256_file(Path(__file__))}
    electronics = category_files("Electronics", raw).reviews
    # Electronics is shared with the completed Phones experiment. Preserve that input.
    if not electronics.is_file():
        raise FileNotFoundError(f"Expected the existing Phones experiment's raw input: {electronics}")
    for pair, (source, target) in PAIRS.items():
        grouped = {}
        for domain in (source, target):
            path = category_files(domain, raw).reviews
            url = f"{AMAZON_2014_BASE_URL}/{path.name}"
            if not path.exists() and args.download:
                print(f"DOWNLOAD {url}", flush=True)
                _download_file(url, path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; use --download")
            print(f"AUDIT {domain}", flush=True)
            grouped[domain], invalid = read_domain(path)
            report["files"][domain] = dict(path=str(path), url=url, sha256=sha256_file(path),
                                           invalid_rows=invalid)
        for name, left, right in ((pair, source, target),
                                  ("_to_".join(reversed(pair.split("_to_"))), target, source)):
            stats = direction_statistics(grouped[left], grouped[right])
            report["directions"][name] = dict(source=left, target=right, **stats)
            print(json.dumps({"direction": name, **stats}), flush=True)
    output = args.root.resolve() / "artifacts/diagnostics/cross-domain-candidate-audit-v1"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = output / f"audit-{stamp}.json"
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f"REPORT {destination}", flush=True)


if __name__ == "__main__":
    main()
