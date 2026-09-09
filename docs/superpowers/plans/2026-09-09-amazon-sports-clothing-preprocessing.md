# Amazon Sports-to-Clothing Preprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a resumable command that downloads or reuses Amazon Reviews 2014 files and produces leakage-free, shared-user Sports-to-Clothing experiment artifacts.

**Architecture:** Keep raw I/O, domain filtering, artifact serialization, and CLI orchestration in separate modules. Stream gzip records instead of loading complete Amazon files into memory, reserve item index `0` for padding, and serialize portable JSON/JSONL outputs with a manifest that fingerprints raw inputs and preprocessing parameters.

**Tech Stack:** Python 3.10, standard library (`ast`, `gzip`, `hashlib`, `json`, `pathlib`), `requests`, `PyYAML`, `unittest`.

**Spec:** `docs/amazon_sports_clothing_preprocessing.md`

## Global Constraints

- Source domain is exactly `Sports_and_Outdoors` and target domain is exactly `Clothing_Shoes_and_Jewelry`.
- Raw data is Amazon Reviews 2014 5-core.
- Keep only original `reviewerID` values present in both domains with at least 5 retained interactions in each domain.
- Item ID is an index and label only; it is never encoded as an input feature.
- Product text uses only `title`, `brand`, `categories`, `feature`, and `description`.
- Validation is the penultimate target event and test is the final target event.
- Every source and target history contains only events strictly earlier than its positive target timestamp.
- Raw and processed data remain under ignored `data/` paths.
- No API calls, Semantic ID training, recommender training, or full-ranking evaluation are part of this plan.

---

### Task 1: Experiment Configuration

**Files:**
- Create: `configs/amazon_sports_clothing.yaml`
- Create: `cdr_framework/experiment_config.py`
- Test: `tests/test_experiment_config.py`

**Interfaces:**
- Consumes: YAML path supplied by the CLI.
- Produces: `AmazonPreprocessingConfig.from_yaml(path: str | Path) -> AmazonPreprocessingConfig`.

- [ ] **Step 1: Write the failing configuration tests**

```python
import tempfile
import unittest
from pathlib import Path

from cdr_framework.experiment_config import AmazonPreprocessingConfig


class AmazonPreprocessingConfigTests(unittest.TestCase):
    def test_loads_expected_domains_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(
                "dataset:\n"
                "  source_domain: Sports_and_Outdoors\n"
                "  target_domain: Clothing_Shoes_and_Jewelry\n"
                "  raw_dir: data/raw/amazon2014\n"
                "  processed_dir: data/processed/sports_to_clothing\n"
                "  min_interactions_per_domain: 5\n",
                encoding="utf-8",
            )
            config = AmazonPreprocessingConfig.from_yaml(path)
            self.assertEqual(config.source_domain, "Sports_and_Outdoors")
            self.assertEqual(config.target_domain, "Clothing_Shoes_and_Jewelry")
            self.assertEqual(config.min_interactions_per_domain, 5)

    def test_rejects_threshold_below_three(self):
        with self.assertRaises(ValueError):
            AmazonPreprocessingConfig(min_interactions_per_domain=2)
```

- [ ] **Step 2: Run the tests and verify the missing module failure**

Run: `python -m unittest tests.test_experiment_config -v`

Expected: FAIL because `cdr_framework.experiment_config` does not exist.

- [ ] **Step 3: Implement the immutable configuration loader**

```python
@dataclass(frozen=True)
class AmazonPreprocessingConfig:
    source_domain: str = "Sports_and_Outdoors"
    target_domain: str = "Clothing_Shoes_and_Jewelry"
    raw_dir: Path = Path("data/raw/amazon2014")
    processed_dir: Path = Path("data/processed/sports_to_clothing")
    min_interactions_per_domain: int = 5
    max_text_chars: int = 12000
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AmazonPreprocessingConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        dataset = payload.get("dataset", {})
        return cls(**dataset)
```

Validate exact domain names, positive text length, and `min_interactions_per_domain >= 3`. Convert path strings to `Path` in `__post_init__` using `object.__setattr__` because the dataclass is frozen.

- [ ] **Step 4: Add the canonical YAML configuration**

```yaml
dataset:
  source_domain: Sports_and_Outdoors
  target_domain: Clothing_Shoes_and_Jewelry
  raw_dir: data/raw/amazon2014
  processed_dir: data/processed/sports_to_clothing
  min_interactions_per_domain: 5
  max_text_chars: 12000
  seed: 42
```

- [ ] **Step 5: Run configuration tests**

Run: `python -m unittest tests.test_experiment_config -v`

Expected: PASS.

- [ ] **Step 6: Commit the configuration unit**

```bash
git add configs/amazon_sports_clothing.yaml cdr_framework/experiment_config.py tests/test_experiment_config.py
git commit -m "Add Amazon preprocessing configuration"
```

### Task 2: Safe Amazon 2014 Raw I/O

**Files:**
- Create: `cdr_framework/datasets/amazon2014.py`
- Modify: `cdr_framework/datasets/__init__.py`
- Test: `tests/test_amazon2014_io.py`

**Interfaces:**
- Consumes: category name, raw directory, and Amazon gzip files.
- Produces: `AmazonCategoryFiles`, `iter_amazon_records(path)`, `download_category_files(category, raw_dir)`, and `sha256_file(path)`.

- [ ] **Step 1: Write failing parser tests with temporary gzip fixtures**

```python
import gzip
import tempfile
import unittest
from pathlib import Path

from cdr_framework.datasets.amazon2014 import iter_amazon_records


class Amazon2014IOTests(unittest.TestCase):
    def test_parses_json_and_python_literal_lines_without_eval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.json.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write('{"asin":"A1","title":"JSON"}\n')
                stream.write("{'asin': 'A2', 'title': 'Literal'}\n")
            records = list(iter_amazon_records(path))
            self.assertEqual([record["asin"] for record in records], ["A1", "A2"])
```

Also test that a malformed non-empty line raises `AmazonRecordParseError` containing the path and one-based line number.

- [ ] **Step 2: Run the parser tests and verify failure**

Run: `python -m unittest tests.test_amazon2014_io -v`

Expected: FAIL because `amazon2014.py` does not exist.

- [ ] **Step 3: Implement safe streaming parsing**

Try `json.loads(line)` first and `ast.literal_eval(line)` second. Require the result to be a dictionary. Never call `eval`. Yield one record at a time from `gzip.open(..., "rt", encoding="utf-8", errors="replace")`.

- [ ] **Step 4: Implement category file paths and resumable downloads**

```python
@dataclass(frozen=True)
class AmazonCategoryFiles:
    reviews: Path
    metadata: Path


def category_files(category: str, raw_dir: str | Path) -> AmazonCategoryFiles:
    root = Path(raw_dir)
    return AmazonCategoryFiles(
        reviews=root / f"reviews_{category}_5.json.gz",
        metadata=root / f"meta_{category}.json.gz",
    )
```

Use the GenCDR-compatible base URL `https://snap.stanford.edu/data/amazon/productGraph/categoryFiles`. Download to a `.part` file, use a `Range` header when a partial file exists, verify gzip integrity, and atomically replace the destination. If a completed valid destination exists, return it without a network request.

- [ ] **Step 5: Test downloader skip, atomic rename, and gzip validation**

Use `unittest.mock.patch("requests.get")`; do not use the network. Assert an existing valid gzip is skipped, a response is written to `.part` then renamed, and invalid gzip content is rejected without replacing an existing valid file.

- [ ] **Step 6: Run raw I/O tests**

Run: `python -m unittest tests.test_amazon2014_io -v`

Expected: PASS.

- [ ] **Step 7: Commit the raw I/O unit**

```bash
git add cdr_framework/datasets/amazon2014.py cdr_framework/datasets/__init__.py tests/test_amazon2014_io.py
git commit -m "Add safe Amazon 2014 raw data IO"
```

### Task 3: Shared-User Filtering and Product Text

**Files:**
- Create: `cdr_framework/datasets/amazon_preprocessing.py`
- Test: `tests/test_amazon_preprocessing.py`

**Interfaces:**
- Consumes: iterables of raw review and metadata dictionaries.
- Produces: `prepare_domain_pair(...) -> PreparedDomainPair` containing mappings, retained events, item texts, and summary statistics.

- [ ] **Step 1: Write failing shared-user and metadata tests**

Create small in-memory fixtures with users who occur in one or both domains, items with and without metadata, and unsorted timestamps. Assert:

```python
prepared = prepare_domain_pair(
    source_reviews=source_reviews,
    target_reviews=target_reviews,
    source_metadata=source_metadata,
    target_metadata=target_metadata,
    min_interactions=5,
    max_text_chars=12000,
)
self.assertEqual(set(prepared.user_to_id), {"shared-user"})
self.assertEqual(prepared.item_to_id["Sports_and_Outdoors:S1"], 1)
self.assertTrue(prepared.item_texts[1].startswith("Title: "))
self.assertNotIn("reviewText", prepared.item_texts[1])
```

Assert item index `0` is reserved, item keys are domain-qualified, users dropping below five interactions after missing-metadata removal are excluded, and events are sorted by `(timestamp, asin)`.

- [ ] **Step 2: Run the tests and verify failure**

Run: `python -m unittest tests.test_amazon_preprocessing -v`

Expected: FAIL because `prepare_domain_pair` is missing.

- [ ] **Step 3: Implement normalized event and prepared-data types**

```python
@dataclass(frozen=True)
class AmazonEvent:
    user_id: str
    asin: str
    domain: str
    timestamp: int


@dataclass(frozen=True)
class PreparedDomainPair:
    user_to_id: dict[str, int]
    item_to_id: dict[str, int]
    id_to_item: list[dict[str, str] | None]
    item_texts: dict[int, str]
    source_events: dict[str, tuple[AmazonEvent, ...]]
    target_events: dict[str, tuple[AmazonEvent, ...]]
    statistics: dict[str, int]
```

- [ ] **Step 4: Implement deterministic product text construction**

Create `build_product_text(record, max_chars)` with fixed field order `Title`, `Brand`, `Categories`, `Features`, `Description`. Decode HTML entities, strip tags, flatten nested lists, collapse whitespace, skip absent fields, and return an empty string if no usable text remains.

- [ ] **Step 5: Implement the filtering fixed point**

Build metadata text only for reviewed ASINs, remove events whose item has no usable metadata, then retain the intersection of users meeting the threshold in both domains. Build mappings after filtering so no unused users or items enter the experiment.

- [ ] **Step 6: Run preprocessing unit tests**

Run: `python -m unittest tests.test_amazon_preprocessing -v`

Expected: PASS.

- [ ] **Step 7: Commit the filtering unit**

```bash
git add cdr_framework/datasets/amazon_preprocessing.py tests/test_amazon_preprocessing.py
git commit -m "Add shared-user Amazon preprocessing"
```

### Task 4: Leakage-Free Splits and Portable Artifacts

**Files:**
- Modify: `cdr_framework/datasets/amazon_preprocessing.py`
- Test: `tests/test_amazon_preprocessing.py`

**Interfaces:**
- Consumes: `PreparedDomainPair`.
- Produces: `build_temporal_splits(prepared) -> TemporalSamples` and `write_preprocessed_artifacts(...) -> Path`.

- [ ] **Step 1: Write failing temporal leakage tests**

Use a target sequence with timestamps `[20, 30, 40, 50, 60]` and source events on both sides of each target. Assert the validation label is timestamp `50`, the test label is timestamp `60`, and each sample contains only source events with timestamps strictly less than its label.

Also assert training examples never use either held-out target item, histories never include their own positive item, and samples with empty source or target history are skipped and counted.

- [ ] **Step 2: Run the split test and verify failure**

Run: `python -m unittest tests.test_amazon_preprocessing.AmazonPreprocessingTests.test_temporal_splits_exclude_future_events -v`

Expected: FAIL because `build_temporal_splits` is missing.

- [ ] **Step 3: Implement sample generation**

```python
@dataclass(frozen=True)
class RecommendationSample:
    user_id: int
    source_items: tuple[int, ...]
    target_items: tuple[int, ...]
    positive_target_item: int
    timestamp: int


@dataclass(frozen=True)
class TemporalSamples:
    train: tuple[RecommendationSample, ...]
    validation: tuple[RecommendationSample, ...]
    test: tuple[RecommendationSample, ...]
```

Create one training sample for each eligible target event before the two holdouts. Validation uses `target[-2]`; test uses `target[-1]`. For every sample, select source events using `source.timestamp < positive.timestamp`.

- [ ] **Step 4: Implement deterministic artifact writing**

Write atomically through a temporary directory, then replace the processed directory. Produce:

```text
mappings.json
item_texts.jsonl
train.jsonl
validation.jsonl
test.jsonl
statistics.json
manifest.json
```

JSON uses UTF-8, sorted keys where applicable, and no Python pickle. `manifest.json` records schema version `1`, source/target domains, threshold, seed, raw filenames, file sizes, SHA-256 values, output row counts, and UTC creation time.

- [ ] **Step 5: Test a write/read round trip**

Write fixtures to a temporary directory, parse every JSON/JSONL output, assert stable mappings and counts, and assert a second run replaces outputs without appending duplicate records.

- [ ] **Step 6: Run preprocessing tests**

Run: `python -m unittest tests.test_amazon_preprocessing -v`

Expected: PASS.

- [ ] **Step 7: Commit split and artifact generation**

```bash
git add cdr_framework/datasets/amazon_preprocessing.py tests/test_amazon_preprocessing.py
git commit -m "Add leakage-free Amazon preprocessing artifacts"
```

### Task 5: Preprocessing CLI and Documentation

**Files:**
- Create: `experiments/prepare_amazon.py`
- Modify: `README.md`
- Modify: `docs/amazon_sports_clothing_preprocessing.md`
- Test: `tests/test_prepare_amazon_cli.py`

**Interfaces:**
- Consumes: `--config`, optional `--skip-download`, and optional `--force`.
- Produces: processed artifacts and a concise terminal summary; exit code `0` on success and nonzero on invalid/missing data.

- [ ] **Step 1: Write a failing CLI integration test**

Invoke `experiments/prepare_amazon.py` through `subprocess.run` with a temporary config and tiny gzip fixtures. Assert exit code `0`, presence of `manifest.json`, and summary lines for retained users, source items, target items, train rows, validation rows, and test rows.

Add a missing-file case using `--skip-download` and assert a nonzero exit code with all missing expected paths listed.

- [ ] **Step 2: Run the CLI tests and verify failure**

Run: `python -m unittest tests.test_prepare_amazon_cli -v`

Expected: FAIL because `experiments/prepare_amazon.py` does not exist.

- [ ] **Step 3: Implement CLI orchestration**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser
```

Resolve relative data paths against the repository root rather than the caller's current directory. Refuse to overwrite an existing manifest unless `--force` is set. Never print environment variables or credentials.

- [ ] **Step 4: Document only commands that now exist**

Move `prepare_amazon.py` from the planned-only block into the executable section. Keep embedding, tokenizer, training, and evaluation commands explicitly marked as unavailable.

- [ ] **Step 5: Run focused and complete test suites**

Run:

```bash
python -m unittest tests.test_experiment_config tests.test_amazon2014_io tests.test_amazon_preprocessing tests.test_prepare_amazon_cli -v
python -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run an offline CLI smoke test**

Run the CLI against test fixtures or already-downloaded local raw files with `--skip-download`; do not invoke external APIs. Verify `manifest.json` and every JSONL file can be parsed.

- [ ] **Step 7: Commit the executable preprocessing stage**

```bash
git add experiments/prepare_amazon.py README.md docs/amazon_sports_clothing_preprocessing.md tests/test_prepare_amazon_cli.py
git commit -m "Add Amazon preprocessing command"
```

### Task 6: Final Review and Server Handoff

**Files:**
- Review only: all files changed in Tasks 1-5.

**Interfaces:**
- Consumes: completed preprocessing implementation.
- Produces: reviewed Git commits ready to push and exact server commands.

- [ ] **Step 1: Inspect the complete diff and repository status**

Run:

```bash
git status --short
git diff origin/main...HEAD --check
git diff origin/main...HEAD --stat
```

- [ ] **Step 2: Verify security and artifact exclusions**

Search for API key literals and confirm `data/`, `.part`, processed artifacts, and local checksums are ignored. Confirm no test fixture contains a real credential.

- [ ] **Step 3: Run the complete verification suite again**

Run: `python -m unittest discover -s tests -v`

Expected: all tests PASS with zero failures and zero errors.

- [ ] **Step 4: Push and verify the remote branch**

```bash
git push origin main
git status --short --branch
```

- [ ] **Step 5: Provide the server commands**

```bash
cd /root/autodl-tmp/SIR-CDR
git pull origin main
conda activate sir-cdr
python experiments/prepare_amazon.py \
  --config configs/amazon_sports_clothing.yaml \
  --skip-download
cat data/processed/sports_to_clothing/statistics.json
```
