import argparse
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Summarize completed text-CDR runs without selecting on test")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.root.glob("*/manifest.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        variant = path.parent.name.rsplit("_seed", 1)[0]
        groups.setdefault(variant, []).append(result["best_validation"])
    if not groups:
        raise SystemExit("No completed runs found")
    report = {}
    for variant, runs in groups.items():
        report[variant] = {"seeds": len(runs), "validation": {}}
        for metric in ("R@5", "N@5", "R@10", "N@10"):
            values = [run[metric] for run in runs]
            report[variant]["validation"][metric] = {
                "mean": statistics.mean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else None,
            }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
