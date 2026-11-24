"""Opt-in dual-NDCG policy; preserves the existing model and raw metric definitions."""
from contextlib import contextmanager
from dataclasses import dataclass
import math
from pathlib import Path

from cdr_framework import llm_training
from cdr_framework.datasets.gencdr_protocol import sha
from cdr_framework.llm_config import LLMCDRConfig
from experiments import recover_llm_evaluation as recovery

MEAN_METRIC = "NDCG@5_10_mean"
SELECTION_POLICY = "mean(NDCG@5,NDCG@10), then mean(HR@5,HR@10), NDCG@10; smaller pool/weight, normalization"


@dataclass(frozen=True)
class AmazonConfig(LLMCDRConfig):
    def __post_init__(self):
        metric = self.selection_metric
        if metric == MEAN_METRIC:
            object.__setattr__(self, "selection_metric", "NDCG@10")
        try:
            super().__post_init__()
            if not {5, 10}.issubset(self.top_ks):
                raise ValueError("Dual NDCG requires cutoffs 5 and 10")
        finally:
            object.__setattr__(self, "selection_metric", metric)


def metric_key(metrics):
    def value(short, full):
        result = float(metrics[short] if short in metrics else metrics[full])
        if not math.isfinite(result) or not 0 <= result <= 1:
            raise ValueError(f"Invalid recommendation metric: {full}")
        return result
    n5, n10 = value("N@5", "NDCG@5"), value("N@10", "NDCG@10")
    h5, h10 = value("R@5", "HR@5"), value("R@10", "HR@10")
    return ((n5 + n10) / 2, (h5 + h10) / 2, n10)


def select_fusion(results):
    return max(results, key=lambda r: (*metric_key(r["metrics"]),
        -r["setting"]["pool"], -r["setting"]["weight"], r["setting"]["normalization"]))


@contextmanager
def installed():
    """Scoped per worker process; the adapter's source hash enters every checkpoint."""
    evaluate, signature = llm_training.evaluate, llm_training.signature
    select, writer, execution_policy = recovery.select_protected, recovery._write_json, recovery.POLICY

    def adapted_evaluate(*args, **kwargs):
        result = dict(evaluate(*args, **kwargs))
        result[MEAN_METRIC] = metric_key(result)[0]
        return result

    def adapted_signature(config):
        result = signature(config)
        result["version"] += "+amazon-dual-ndcg-v1"
        result["code"][Path(__file__).name] = sha(__file__)
        return result

    def write_report(path, value):
        if Path(path).name == "search.json" and "results" in value:
            value["selection_policy"] = SELECTION_POLICY
        writer(path, value)

    try:
        llm_training.evaluate, llm_training.signature = adapted_evaluate, adapted_signature
        recovery.select_protected, recovery._write_json = select_fusion, write_report
        recovery.POLICY = {**execution_policy, "schema": "amazon-dual-ndcg-snapshot-v1",
                           "selection_policy": SELECTION_POLICY, "adapter_sha256": sha(__file__)}
        yield
    finally:
        llm_training.evaluate, llm_training.signature = evaluate, signature
        recovery.select_protected, recovery._write_json, recovery.POLICY = select, writer, execution_policy
