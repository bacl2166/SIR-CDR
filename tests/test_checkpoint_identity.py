# -*- coding: utf-8 -*-
"""Unit tests for the layered checkpoint identity check (inference-only code drift opt-in)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cdr_framework.text_training import verify_checkpoint_identity  # noqa: E402


def _sig(**overrides):
    sig = {
        "version": "text-cdr-v3",
        "config": {"hidden_dim": 128},
        "artifacts": {"train": "abc"},
        "code": {"text_model.py": "hash1"},
    }
    sig.update(overrides)
    return sig


class CheckpointIdentityTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
