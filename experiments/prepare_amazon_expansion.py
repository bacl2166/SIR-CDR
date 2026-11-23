"""Prepare four Amazon directions using the verified Phones data protocol."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cdr_framework.datasets import gencdr_inputs, gencdr_protocol
from cdr_framework.datasets.amazon2014 import (
    AMAZON_2014_BASE_URL, _download_file, _validate_gzip, category_files,
)
from cdr_framework.datasets.gencdr_inputs import raw_amazon
from cdr_framework.datasets.gencdr_protocol import SCHEMA, build_shared, sha, verify_shared, write
from experiments.audit_cross_domain_candidates import PAIRS, direction_statistics, read_domain


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def reference_protocol(root):
    reference = root / "data/processed" / SCHEMA / "phones_to_electronics"
    verify_shared(reference)
    rules = read(reference / "protocol.json")
    identity = rules["identity"]
    if (identity["converter_sha256"] != sha(gencdr_protocol.__file__) or
            identity["provenance"]["reader_sha256"] != sha(gencdr_inputs.__file__)):
        raise RuntimeError("Phones reference converter or input reader differs from this checkout")
    if (identity["source_domain"], identity["target_domain"]) != (
            "Cell_Phones_and_Accessories", "Electronics"):
        raise RuntimeError("Unexpected Phones reference domain roles")
    for name, digest in identity["provenance"]["raw_files"].items():
        print(f"VERIFY_REFERENCE_RAW {name}", flush=True)
        if Path(name).name != name or sha(root / "data/raw/amazon2014" / name) != digest:
            raise RuntimeError(f"Phones reference raw input changed: {name}")
    if identity["training_examples"] != "all-prefixes":
        raise RuntimeError("Expected the completed Phones all-prefix training protocol")
    return reference, rules


def ensure_raw(root, audit, download):
    raw = root / "data/raw/amazon2014"
    domains = dict.fromkeys(domain for pair in PAIRS.values() for domain in pair)
    for domain in domains:
        print(f"VERIFY_RAW {domain}", flush=True)
        files = category_files(domain, raw)
        if (not files.reviews.is_file() or
                sha(files.reviews) != audit["files"][domain]["sha256"]):
            raise RuntimeError(f"Review input differs from completed size audit: {domain}")
        for path in (files.reviews, files.metadata):
            if not path.exists() and download:
                print(f"DOWNLOAD {path.name}", flush=True)
                _download_file(f"{AMAZON_2014_BASE_URL}/{path.name}", path)
            if not path.is_file():
                raise FileNotFoundError(f"Missing {path}; use --download-metadata")
            _validate_gzip(path)


def audit_statistics(root, source, target):
    raw = root / "data/raw/amazon2014"
    left, _ = read_domain(category_files(source, raw).reviews)
    right, _ = read_domain(category_files(target, raw).reviews)
    return direction_statistics(left, right), direction_statistics(right, left)


def prepare(root, audit_path, *, download):
    root, audit_path = Path(root).resolve(), Path(audit_path).resolve()
    print("VERIFY_PHONES_REFERENCE", flush=True)
    reference, reference_rules = reference_protocol(root)
    reference_hash = sha(reference / "shared_manifest.json")
    audit_hash = sha(audit_path)
    audit = read(audit_path)
    if audit.get("schema") != "cross-domain-candidate-audit-v1":
        raise ValueError("Expected the completed candidate size audit")
    ensure_raw(root, audit, download)
    report = {"schema": "amazon-expansion-phones-alignment-v1", "complete": True,
              "reference": str(reference), "reference_sha256": reference_hash,
              "audit_sha256": audit_hash, "tool_sha256": sha(__file__), "directions": {}}
    comparable_rules = {k: v for k, v in reference_rules.items() if k != "identity"}
    for forward, (source, target) in PAIRS.items():
        reverse = "_to_".join(reversed(forward.split("_to_")))
        print(f"CHECK_AUDIT {forward} {reverse}", flush=True)
        stats = audit_statistics(root, source, target)
        for name, expected in zip((forward, reverse), stats):
            if any(audit["directions"][name].get(k) != v for k, v in expected.items()):
                raise RuntimeError(f"Input audit counts differ: {name}")
        # Use the exact same static text loader as the completed Phones experiment.
        events, items, provenance = raw_amazon(root, source, target)
        for domain in (source, target):
            filename = category_files(domain, root / "data/raw/amazon2014").reviews.name
            if provenance["raw_files"][filename] != audit["files"][domain]["sha256"]:
                raise RuntimeError(f"Conversion input differs from audited review: {domain}")
        for name, left, right in ((forward, source, target), (reverse, target, source)):
            output = root / "data/processed" / SCHEMA / name
            print(f"CONVERT {name}", flush=True)
            build_shared(events, items, left, right, output, provenance=provenance,
                         training_examples=reference_rules["identity"]["training_examples"])
            actual_rules = {k: v for k, v in read(output / "protocol.json").items() if k != "identity"}
            if actual_rules != comparable_rules:
                raise RuntimeError(f"Processed rules differ from Phones: {name}")
            actual = read(output / "statistics.json")
            expected = audit["directions"][name]
            for key in ("users", "source_items", "target_items", "train_rows", "validation_rows", "test_rows",
                        "train_positives_in_seen", "validation_positives_in_seen", "test_positives_in_seen"):
                if actual.get(key, 0) != expected[key]:
                    raise RuntimeError(f"Processed audit counts differ: {name} {key}")
            report["directions"][name] = {
                "statistics": actual, "protocol_rules_equal": True,
                "shared_manifest_sha256": sha(output / "shared_manifest.json"),
                "text_policy": provenance["text_policy"], "raw_text_audit": provenance["raw_audit"],
            }
        del events, items
    # Recheck all file bindings after conversion, including the original reference.
    verify_shared(reference)
    if sha(reference / "shared_manifest.json") != reference_hash or sha(audit_path) != audit_hash:
        raise RuntimeError("Reference or audit changed while preparing data")
    reference_protocol(root)
    raw_bindings = {}
    for name in report["directions"]:
        output = root / "data/processed" / SCHEMA / name
        verify_shared(output)
        provenance = read(output / "protocol.json")["identity"]["provenance"]
        for filename, digest in provenance["raw_files"].items():
            if filename in raw_bindings and raw_bindings[filename] != digest:
                raise RuntimeError(f"Conflicting raw data identities: {filename}")
            raw_bindings[filename] = digest
    for filename, digest in raw_bindings.items():
        if sha(root / "data/raw/amazon2014" / filename) != digest:
            raise RuntimeError(f"Raw data changed during conversion: {filename}")
    folder = root / "artifacts/diagnostics/amazon-expansion-phones-alignment-v1"
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = folder / f"alignment-{stamp}.json"
    write(destination, report)
    print(f"ALIGNMENT_OK {destination}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/Data/lihb/SIR-CDR"))
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--download-metadata", action="store_true")
    args = parser.parse_args()
    from experiments.recover_llm_evaluation import operation_lock
    lock = args.root.resolve() / "artifacts/diagnostics/amazon-expansion-phones-alignment-v1/controller"
    with operation_lock(lock):
        prepare(args.root, args.audit, download=args.download_metadata)


if __name__ == "__main__":
    main()
