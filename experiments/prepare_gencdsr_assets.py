"""Convert the published GenCDSR Book/Movie sequences without inventing timestamps."""
import argparse
import hashlib
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import yaml
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdr_framework.formal_training import _sha256, _write_json

UPSTREAM = "https://github.com/Applied-Machine-Learning-Lab/RecSys2026_GenCDSR.git"
REVISION = "bace0d18f53bfeb56ab9bdd564494055b6c38db7"
EXPECTED_FILES = {
    "Book_Movie.inter.json": "5b0041a60cf6df2428964889ccd6442f5db6431cddfaf126945673e1ae1e4cf3",
    "Book_Movie.domain.json": "30de8715b9c2778f5654b4c7cb36577d5695a7c73e112a2d0d24cdacb1b805c1",
    "Book/Book.inter.json": "2d6630782ad2c933137a9029fe871dcf0912d5aceb2da8ae50e3881f040265ba",
    "Book/Book.item.json": "349c1a0a7897112fd3da96cd7d4a3acaeb87040d2d803ce3f85820a0f2ccec7c",
    "Movie/Movie.inter.json": "66cd65c03ba559c784c03d499a15cf548df5550abae468c889f5f57f0877f625",
    "Movie/Movie.item.json": "95babcea96f76f63918c40198ad118b96afeb3ab74eab8561e3525382794cc98",
}


def extract_reference_archive(archive, destination):
    """Extract just six fixed files; data hashes prove the snapshot without Git."""
    destination = Path(destination)
    prefix = f"RecSys2026_GenCDSR-{REVISION}/data/Book_Movie/"
    verified = {}
    with ZipFile(archive) as source:
        names = source.namelist()
        for name, expected in EXPECTED_FILES.items():
            member = prefix + name
            if names.count(member) != 1:
                raise ValueError(f"Missing/duplicate pinned archive member: {name}")
            if source.getinfo(member).file_size > 32 * 1024**2:
                raise ValueError(f"Unexpected archive size: {name}")
            data = source.read(member)
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Pinned data hash mismatch: {name}")
            path = destination / "data/Book_Movie" / name
            if path.exists() and _sha256(path) != expected:
                raise ValueError(f"Existing reference data changed: {path}")
            verified[path] = data
    # Do not use extractall: unneeded archive paths are never written.
    for path, data in verified.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return destination


def convert_dataset(raw, output, target="Movie", revision=REVISION):
    raw, output = Path(raw), Path(output)
    if target not in {"Book", "Movie"}:
        raise ValueError("target must be Book or Movie")
    source = "Book" if target == "Movie" else "Movie"
    names = ["Book_Movie.inter.json", "Book_Movie.domain.json"] + [
        f"{d}/{d}.{kind}.json" for d in ("Book", "Movie") for kind in ("inter", "item")]
    hashes = {n: _sha256(raw / n) for n in names}
    identity = {"schema": "gencdsr-overlap-prefix-v1", "source": UPSTREAM,
                "revision": revision, "target": target, "inputs": hashes,
                "converter_sha256": _sha256(Path(__file__))}
    manifest = output / "manifest.json"
    if manifest.exists():
        old = json.loads(manifest.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            raise ValueError("Processed identity changed; choose a new output directory")
        if any(_sha256(output / n) != h for n, h in old["outputs"].items()):
            raise ValueError("Processed files changed")
        return old
    if output.exists() and any(output.iterdir()):
        raise ValueError("Refusing to replace an incomplete/unmanaged output directory")
    def read(n):
        return json.loads((raw / n).read_text(encoding="utf-8"))
    mixed, domains = read(names[0]), read(names[1])
    seqs = {d: read(f"{d}/{d}.inter.json") for d in ("Book", "Movie")}
    meta = {d: read(f"{d}/{d}.item.json") for d in ("Book", "Movie")}
    if set(mixed) != set(domains):
        raise ValueError("Mixed user/domain keys differ")
    items, item_to_id, texts = [None], {}, []
    missing_metadata = Counter()
    for d in ("Book", "Movie"):
        for key in sorted(meta[d], key=int):
            iid = len(items); item_to_id[f"{d}:{key}"] = iid
            items.append({"domain": d, "original_item_id": key})
            record = meta[d][key]
            text = "\n".join(str(record.get(k, "")).strip() for k in ("title", "description")).strip()
            if not text:
                # Keep the published catalog. A disclosed constant domain
                # marker contains no invented title, review or attributes.
                text = f"{d}. Metadata unavailable."
                missing_metadata[d] += 1
            texts.append({"item_id": iid, "domain": d, "text": text})
    splits = {s: [] for s in ("train", "validation", "test")}
    excluded, retained = Counter(), []
    for user in sorted(mixed, key=int):
        sequence, labels = mixed[user], domains[user]
        if len(sequence) != len(labels) or any(type(d) is not int or d not in (0,1) for d in labels):
            raise ValueError(f"Malformed domain sequence for user {user}")
        positions = {d: [i for i,z in enumerate(labels) if z == (d == "Movie")] for d in ("Book","Movie")}
        for d in ("Book", "Movie"):
            projection = [sequence[i] for i in positions[d]]
            if projection != seqs[d].get(user, []):
                raise ValueError(f"Domain projection differs for {user}/{d}")
            if any(f"{d}:{i}" not in item_to_id for i in projection):
                raise ValueError(f"Unknown item for user {user}")
        if user not in seqs[source] or user not in seqs[target]:
            excluded["nonoverlap_user"] += 1; continue
        if min(len(positions[source]), len(positions[target])) < 3:
            excluded["short_domain_sequence"] += 1; continue
        # Never import the other domain's validation/test interactions into the
        # training graph. This is an explicit common-subset protocol, not a
        # claim to reproduce the original multi-stage training population.
        source_train = positions[source][:-2]
        target_positions = positions[target]
        def row(j):
            pos = target_positions[j]
            src = [item_to_id[f"{source}:{sequence[i]}"] for i in source_train if i < pos]
            tgt = [item_to_id[f"{target}:{sequence[i]}"] for i in target_positions[:j]]
            return {"user_id": int(user), "source_items": src, "target_items": tgt,
                    "positive_target_item": item_to_id[f"{target}:{sequence[pos]}"],
                    "sequence_position": pos, "timestamp": pos}
        valid, test = row(len(target_positions)-2), row(len(target_positions)-1)
        if not valid["source_items"] or not test["source_items"]:
            excluded["empty_source_before_eval"] += 1; continue
        # Existing SIR ranks unseen items. Keep the published positive positions
        # but explicitly define a novel-item common subset for BOTH systems.
        # Never replace a repeated holdout by an easier earlier/later item.
        if any(r["positive_target_item"] in r["target_items"] for r in (valid,test)):
            excluded["repeated_eval_positive_user"] += 1; continue
        train = []
        for j in range(1, len(target_positions)-2):
            r = row(j)
            if r["positive_target_item"] in r["target_items"]:
                excluded["repeated_training_positive"] += 1
            elif r["source_items"]:
                train.append(r)
            else:
                excluded["training_row_without_source"] += 1
        if not train:
            excluded["user_without_training_rows"] += 1; continue
        retained.append(user)
        splits["train"].extend(train)
        splits["validation"].append(valid); splits["test"].append(test)
    if any(not rows for rows in splits.values()):
        raise ValueError("Empty split after explicit overlap/prefix filtering")
    output.mkdir(parents=True, exist_ok=True)
    mapping = {"source_domain": source, "target_domain": target, "id_to_item": items,
               "item_to_id": item_to_id, "retained_original_users": retained}
    _write_json(output / "mappings.json", mapping)
    for name, rows in {"item_texts": texts, **splits}.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    result = {"identity": identity, "dataset": {"source_domain": source, "target_domain": target},
              "protocol": {"name": "common_overlap_novel_item_subset_target_leave_two",
                           "timestamp_kind": "sequence_position_not_wall_clock",
                           "source_history": "published prefix; other-domain last two withheld",
                           "repeated_positives": "exclude user if either original eval positive was already seen; never re-split",
                           "provenance": "Published Book_Movie; Douban version not stated in README",
                           "direct_paper_comparison": False},
              "users": len(retained), "excluded": dict(excluded),
              "missing_metadata_domain_marker": dict(missing_metadata),
              "rows": {s: len(v) for s,v in splits.items()},
              "positives_in_seen": {s: sum(r["positive_target_item"] in r["target_items"] for r in rows)
                                    for s,rows in splits.items()},
              "outputs": {p.name: _sha256(p) for p in output.iterdir()}}
    _write_json(manifest, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--reference-root", type=Path)
    reference.add_argument("--reference-archive", type=Path, help="Pinned codeload ZIP; all six data hashes verified")
    parser.add_argument("--config", default="configs/gencdsr_book_movie.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--action", choices=("convert", "all"), default="all")
    args = parser.parse_args()
    revision = REVISION
    if args.reference_archive:
        args.reference_root = extract_reference_archive(
            args.reference_archive, ROOT / "data/raw/gencdsr_reference" / REVISION)
    else:
        if not (args.reference_root / "data/Book_Movie").is_dir():
            parser.error("Reference checkout is missing; clone successfully or use --reference-archive")
        revision = subprocess.check_output(["git", "-C", str(args.reference_root), "rev-parse", "HEAD"], text=True).strip()
        if revision != REVISION:
            raise ValueError(f"Reference must be pinned to {REVISION}; found {revision}")
        status = subprocess.check_output(["git", "-C", str(args.reference_root), "status", "--porcelain", "--", "data/Book_Movie"], text=True)
        if status.strip():
            raise ValueError("Reference data have local changes")
    for name, expected in EXPECTED_FILES.items():
        if _sha256(args.reference_root / "data/Book_Movie" / name) != expected:
            raise ValueError(f"Pinned data hash mismatch: {name}")
    cfg = Path(args.config)
    if not cfg.is_absolute(): cfg = ROOT / cfg
    payload = yaml.safe_load(cfg.read_text(encoding="utf-8")); dataset = payload["dataset"]
    output = ROOT / dataset["processed_dir"]
    print(json.dumps(convert_dataset(args.reference_root / "data/Book_Movie", output,
                                    dataset["target_domain"], revision), ensure_ascii=False), flush=True)
    if args.action == "convert": return
    subprocess.run([sys.executable, str(ROOT / "experiments/embed_items.py"),
                    "--config", str(cfg), "--device", args.device], cwd=ROOT, check=True)
    # The legacy CLI parses Amazon-specific domains; call its general run
    # function with explicit Book/Movie labels instead of relabelling the data.
    from cdr_framework.experiment_config import TokenizerTrainingConfig
    from experiments.train_tokenizer import run
    run(TokenizerTrainingConfig.from_yaml(cfg), device=args.device, force=False,
        source_domain=dataset["source_domain"], target_domain=dataset["target_domain"])


if __name__ == "__main__":
    main()
