# -*- coding: utf-8 -*-
"""Unit tests for the layered checkpoint identity check (inference-only code drift opt-in)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdr_framework.text_training import evaluation_identity, verify_checkpoint_identity  # noqa: E402


def _sig(**overrides):
    sig = {
        "version": "text-cdr-v4",
        "config": {"hidden_dim": 128},
        "artifacts": {"train": "abc"},
        "code": {"text_model.py": "hash1"},
    }
    sig.update(overrides)
    return sig


class CheckpointIdentityTests(unittest.TestCase):
    def test_text_signature_hashes_shared_tensor_operations(self):
        from cdr_framework.text_config import TextCDRConfig
        from cdr_framework import text_training

        with mock.patch.object(text_training, "_artifact_fingerprints", return_value={}):
            identity = text_training.signature(TextCDRConfig())
        self.assertIn("ops.py", identity["code"])

    def test_checkpoint_training_payload_excludes_runtime_scoring_but_keeps_loss_weights(self):
        from dataclasses import replace
        from cdr_framework.text_config import TextCDRConfig
        from cdr_framework.text_training import _training_config_payload

        base = TextCDRConfig()
        runtime = replace(
            base,
            generation_score_weight=2.0,
            fusion_normalization="zscore",
            rerank_candidates=1000,
            evaluation_batch_size=3,
        )
        self.assertEqual(_training_config_payload(base), _training_config_payload(runtime))
        changed_loss = replace(base, generation_loss_weight=2.0)
        self.assertNotEqual(_training_config_payload(base), _training_config_payload(changed_loss))

    def test_evaluation_identity_tracks_every_runtime_ranking_setting(self):
        from dataclasses import replace
        from cdr_framework.text_config import TextCDRConfig

        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best_model.pt"
            checkpoint.write_bytes(b"checkpoint")
            config = TextCDRConfig(
                rerank_candidates=321,
                retrieval_temperature=0.25,
                retrieval_score_weight=2.0,
                generation_score_weight=0.5,
                fusion_normalization="log_softmax",
                evaluation_batch_size=7,
            )
            identity = evaluation_identity(config, checkpoint, "validation", "hybrid")
            for key in (
                "checkpoint_sha256", "split", "mode", "rerank_candidates",
                "retrieval_temperature", "retrieval_score_weight",
                "generation_score_weight", "fusion_normalization", "evaluation_batch_size",
            ):
                self.assertIn(key, identity)
            changed = evaluation_identity(
                replace(config, generation_score_weight=1.5), checkpoint, "validation", "hybrid"
            )
            self.assertNotEqual(identity, changed)
            self.assertEqual(checkpoint.read_bytes(), b"checkpoint")

    def test_text_config_validates_explicit_loss_score_and_calibration_fields(self):
        from cdr_framework.text_config import TextCDRConfig

        config = TextCDRConfig(
            retrieval_loss_weight=0.5,
            generation_loss_weight=2.0,
            retrieval_score_weight=1.5,
            generation_score_weight=0.25,
            fusion_normalization="zscore",
        )
        self.assertEqual(config.retrieval_loss_weight, 0.5)
        self.assertEqual(config.generation_score_weight, 0.25)
        for kwargs in (
            {"retrieval_loss_weight": 0.0, "generation_loss_weight": 0.0},
            {"retrieval_score_weight": 0.0, "generation_score_weight": 0.0},
            {"retrieval_loss_weight": float("nan")},
            {"fusion_normalization": "softmax"},
        ):
            with self.assertRaises(ValueError):
                TextCDRConfig(**kwargs)

    def test_legacy_yaml_weights_migrate_to_score_weights_and_conflicts_fail(self):
        from cdr_framework.text_config import TextCDRConfig

        with tempfile.TemporaryDirectory() as directory:
            legacy = Path(directory) / "legacy.yaml"
            legacy.write_text(
                "text_recommendation:\n  retrieval_weight: 2.0\n  generation_weight: 0.5\n",
                encoding="utf-8",
            )
            config = TextCDRConfig.from_yaml(legacy)
            self.assertEqual(config.retrieval_score_weight, 2.0)
            self.assertEqual(config.generation_score_weight, 0.5)
            self.assertEqual(config.retrieval_loss_weight, 1.0)
            conflict = Path(directory) / "conflict.yaml"
            conflict.write_text(
                "text_recommendation:\n  retrieval_weight: 2.0\n  retrieval_score_weight: 1.0\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                TextCDRConfig.from_yaml(conflict)

    def test_accepts_matching_identity(self):
        with mock.patch("cdr_framework.text_training.signature", return_value=_sig()):
            ok, why = verify_checkpoint_identity({"identity": _sig()}, object())
        self.assertTrue(ok)

    def test_rejects_config_drift_even_when_code_allowed(self):
        cp = {"identity": _sig()}
        with mock.patch("cdr_framework.text_training.signature",
                        return_value=_sig(config={"hidden_dim": 256})):
            ok, why = verify_checkpoint_identity(cp, object(), require_code=False)
        self.assertFalse(ok)
        self.assertIn("config", why)

    def test_rejects_artifact_drift_even_when_code_allowed(self):
        cp = {"identity": _sig()}
        with mock.patch("cdr_framework.text_training.signature",
                        return_value=_sig(artifacts={"train": "xyz"})):
            ok, why = verify_checkpoint_identity(cp, object(), require_code=False)
        self.assertFalse(ok)
        self.assertIn("artifact", why)

    def test_code_drift_rejected_by_default(self):
        cp = {"identity": _sig(code={"text_model.py": "deadbeef"})}
        with mock.patch("cdr_framework.text_training.signature", return_value=_sig()):
            ok, why = verify_checkpoint_identity(cp, object())
        self.assertFalse(ok)
        self.assertIn("code", why)

    def test_code_drift_allowed_explicitly(self):
        cp = {"identity": _sig(code={"text_model.py": "deadbeef"})}
        with mock.patch("cdr_framework.text_training.signature", return_value=_sig()):
            ok, why = verify_checkpoint_identity(cp, object(), require_code=False)
        self.assertTrue(ok)
        self.assertIn("allowed", why)

    def test_missing_identity_rejected(self):
        with mock.patch("cdr_framework.text_training.signature", return_value=_sig()):
            ok, why = verify_checkpoint_identity({"model": 1}, object())
        self.assertFalse(ok)
        self.assertIn("missing", why)


    def test_fusion_normalization_is_an_explicit_config_field(self):
        from dataclasses import asdict, replace

        from cdr_framework.text_config import TextCDRConfig

        base = TextCDRConfig.from_yaml(str(ROOT / "configs/text_sports_clothing.yaml"))
        runtime = replace(base, evaluation_batch_size=16, fusion_normalization="log_softmax")
        self.assertEqual(runtime.fusion_normalization, "log_softmax")
        self.assertIn("fusion_normalization", asdict(runtime))


    def test_fusion_normalization_survives_replace_loop(self):
        from dataclasses import replace

        from cdr_framework.text_config import TextCDRConfig

        base = TextCDRConfig.from_yaml(str(ROOT / "configs/text_sports_clothing.yaml"))
        runtime_base = replace(base, evaluation_batch_size=16, fusion_normalization="zscore")
        for weight in (0.5, 1.0):
            runtime = replace(runtime_base, generation_score_weight=weight)
            self.assertEqual(runtime.fusion_normalization, "zscore")


    def test_label_smoothing_parses_and_stays_out_of_signature_payload(self):
        from cdr_framework.formal_training import _config_payload
        from cdr_framework.text_config import TextCDRConfig

        base = TextCDRConfig()
        self.assertAlmostEqual(base.label_smoothing, 0.0)
        self.assertNotIn("label_smoothing", _config_payload(base))
        tuned = TextCDRConfig.from_yaml(str(ROOT / "configs/text_sports_clothing_l3a.yaml"))
        self.assertAlmostEqual(tuned.label_smoothing, 0.1)
        self.assertNotIn("label_smoothing", _config_payload(tuned))


if __name__ == "__main__":
    unittest.main()
