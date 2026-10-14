import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch

from experiments.tune_llm_parameters import parameter_settings,rank_parameters,select_protected,run


class ParameterTests(unittest.TestCase):
    def test_ranking_preserves_existing_calibration_masks_and_labels(self):
        from test_text_pipeline import fixture
        from cdr_framework.text_training import build_model,load_rows
        from cdr_framework.text_data import collate_text_rows
        from dataclasses import replace
        from experiments.tune_llm_fusion import collect_scores
        with tempfile.TemporaryDirectory() as d:
            config=fixture(Path(d));model,cat=build_model(config,torch.device('cpu'))
            rows=load_rows(config,cat,'validation')
            cache=collect_scores(model,rows,config,torch.device('cpu'),8)
            batch,seen=collate_text_rows(rows)
            for norm in ('none','zscore','log_softmax'):
                model.config=replace(config,rerank_candidates=5,generation_score_weight=.625,fusion_normalization=norm)
                expected=model.rank(batch,seen,3,'hybrid')
                setting={'pool':5,'weight':.625,'normalization':norm}
                actual=rank_parameters(cache,setting,3)
                self.assertTrue(torch.equal(actual,expected))
                cache['positive']=torch.tensor([999,999])
                self.assertTrue(torch.equal(actual,rank_parameters(cache,setting,3)))
            self.assertEqual(len(parameter_settings(config)),17)

    def test_selection_protects_all_four_metrics(self):
        base={k:.1 for k in ('R@5','R@10','N@5','N@10')}
        anchor={'setting':{'pool':500,'weight':.5,'normalization':'zscore'},'metrics':base}
        worse={'setting':{'pool':1000,'weight':1.,'normalization':'zscore'},'metrics':dict(base,**{'N@10':.2,'R@5':.09})}
        better={'setting':{'pool':500,'weight':.625,'normalization':'zscore'},'metrics':dict(base,**{'N@10':.11})}
        self.assertEqual(select_protected([anchor,worse,better])['setting'],better['setting'])
        self.assertEqual(select_protected([anchor,worse]),anchor)

    def test_compact_checkpoint_search_verify_and_locked_test(self):
        from test_llm_training import config_at,offline_backbone
        from cdr_framework import llm_training
        from experiments.tune_llm_fusion import run as first_phase
        with tempfile.TemporaryDirectory() as d,patch('cdr_framework.llm_reasoner.QwenLatentReasoner.from_pretrained',offline_backbone):
            root=Path(d);config=config_at(root)
            llm_training.train(config,torch.device('cpu'))
            cache_root=root/'phase1';out=root/'phase2'
            first_phase(config,torch.device('cpu'),cache_root,'all')
            run(config,cache_root,out,'sweep',torch.device('cpu'))
            report=json.loads((out/'search.json').read_text())
            self.assertEqual(len(report['results']),17)
            run(config,cache_root,out,'verify',torch.device('cpu'))
            original=(out/'search.json').read_bytes()
            report['best']['setting']['weight']+=.125
            (out/'search.json').write_text(json.dumps(report))
            with self.assertRaisesRegex(RuntimeError,'verified|report'):
                run(config,cache_root,out,'test',torch.device('cpu'))
            (out/'search.json').write_bytes(original)
            run(config,cache_root,out,'test',torch.device('cpu'))
            self.assertTrue((config.output_dir/'test.lock').exists())
            saved=(config.output_dir/'test.json').read_bytes()
            run(config,cache_root,out,'test',torch.device('cpu'))
            self.assertEqual((config.output_dir/'test.json').read_bytes(),saved)
            with self.assertRaisesRegex(RuntimeError,'test'):
                run(config,cache_root,out,'sweep',torch.device('cpu'))
