"""Behavioral checks for the bounded validation-only study policy."""
import copy
from dataclasses import replace
import importlib.util
import math
from pathlib import Path
import unittest


SPEC = importlib.util.spec_from_file_location(
    "ndcg_policy", Path(__file__).resolve().parents[1] / "cdr_framework" / "ndcg_study.py")
policy = importlib.util.module_from_spec(SPEC)
if SPEC.loader and SPEC.origin and Path(SPEC.origin).exists():
    SPEC.loader.exec_module(policy)


def metrics(value=0.2, examples=100):
    return {"R@5": value, "N@5": value, "R@10": value,
            "N@10": value, "R@20": value, "N@20": value,
            "examples": examples}


def screenings():
    return [dict(job, metrics=metrics()) for job in policy.screen_jobs()]


def finals():
    rows = []
    for pair in ("sports_to_clothing", "clothing_to_sports"):
        for cid in ("c00", "c01"):
            for seed in (42, 43, 44):
                rows.append({"pair": pair, "candidate": cid, "seed": seed,
                             "results": [{"setting": setting, "metrics": metrics()}
                                         for setting in policy.fusion_settings()]})
    return rows


class StudyPolicyTests(unittest.TestCase):
    def test_job_admission_alternates_directions(self):
        promotions = {pair: ['c00', 'c01'] for pair in policy.PAIRS}
        for jobs in (policy.screen_jobs(), policy.confirmation_jobs(promotions),
                     policy.validation_jobs(promotions)):
            with self.subTest(first_jobs=jobs[:2]):
                self.assertEqual([job['pair'] for job in jobs[:2]], list(policy.PAIRS))
                for index in range(0, len(jobs), 2):
                    self.assertEqual([job['pair'] for job in jobs[index:index + 2]],
                                     list(policy.PAIRS))

    def test_candidate_overrides_construct_the_original_llm_config(self):
        from cdr_framework.llm_config import LLMCDRConfig
        baseline = LLMCDRConfig(llm_revision='a' * 40)
        for spec in policy.candidate_specs():
            with self.subTest(candidate=spec['id']):
                configured = replace(baseline, **spec['overrides'])
                self.assertEqual(configured.selection_metric, 'NDCG@10')

    def test_fixed_staging_has_24_unique_training_jobs_and_12_validations(self):
        screen = policy.screen_jobs()
        promotions = policy.promote(screenings())
        self.assertEqual(promotions, {"sports_to_clothing": ["c00", "c01"],
                                      "clothing_to_sports": ["c00", "c01"]})
        confirmation = policy.confirmation_jobs(promotions)
        all_jobs = screen + confirmation
        self.assertEqual(len(screen), 16)
        self.assertEqual(len(confirmation), 8)
        self.assertEqual(len({(j['pair'], j['candidate'], j['seed']) for j in all_jobs}), 24)
        self.assertEqual(len(policy.validation_jobs(promotions)), 12)
        self.assertTrue(all(set(j) == {"pair", "candidate", "seed"} for j in all_jobs))
        specs = policy.candidate_specs()
        self.assertEqual(specs[0], {"id": "c00", "overrides": {"selection_metric": "NDCG@10"}})
        self.assertEqual(specs[7]["overrides"], {"selection_metric": "NDCG@10",
                         "learning_rate": 3e-5, "retrieval_temperature": .05,
                         "generation_loss_weight": .5, "cpf_weight": .03})
        self.assertEqual(specs[2]["overrides"]["learning_rate"], 3e-4)

    def test_screen_priority_is_ndcg_then_recall_and_lower_candidate(self):
        rows = screenings()
        for row in rows:
            if row['pair'] == 'sports_to_clothing':
                if row['candidate'] == 'c07':
                    row['metrics']['N@10'] = .3
                elif row['candidate'] == 'c05':
                    row['metrics']['R@10'] = .4
        self.assertEqual(policy.promote(reversed(rows))["sports_to_clothing"], ["c07", "c05"])

    def test_screen_requires_all_16_distinct_seed42_jobs(self):
        for mutation in (lambda r: r[:-1], lambda r: r + [copy.deepcopy(r[0])],
                         lambda r: [dict(r[0], seed=43)] + r[1:],
                         lambda r: [dict(r[0], candidate="c08")] + r[1:]):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                policy.promote(mutation(screenings()))

    def test_promotions_require_two_distinct_known_candidates_each_pair(self):
        for bad in ({}, {p: ["c00"] for p in policy.PAIRS},
                    {p: ["c00", "c00"] for p in policy.PAIRS},
                    {p: ["c00", "c08"] for p in policy.PAIRS}):
            for builder in (policy.confirmation_jobs, policy.validation_jobs):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    builder(bad)

    def test_fixed_48_settings_are_unique_and_fresh(self):
        settings = policy.fusion_settings()
        self.assertEqual(len(settings), 48)
        self.assertEqual({s['pool'] for s in settings}, {500, 1000})
        self.assertEqual({s['normalization'] for s in settings}, {"none", "zscore"})
        self.assertEqual({s['weight'] for s in settings},
                         {0, .125, .25, .375, .5, .625, .75, .875, 1, 1.25, 1.5, 2})
        self.assertEqual(len({tuple(sorted(s.items())) for s in settings}), 48)
        settings[0]['pool'] = 9
        self.assertEqual(policy.fusion_settings()[0]['pool'], 500)

    def test_common_setting_and_three_seed_mean_prevent_best_seed_selection(self):
        rows = finals()
        for row in rows:
            if row['candidate'] == 'c00':
                for result in row['results']:
                    result['metrics'] = metrics(.9 if row['seed'] == 42 else .1)
            else:
                for result in row['results']:
                    setting = result['setting']
                    good = setting == {"pool": 1000, "normalization": "zscore",
                                       "weight": .5}
                    result['metrics'] = metrics({42: .4, 43: .5, 44: .6}[row['seed']] if good else .2)
        result = policy.select_final(reversed(rows))
        self.assertEqual(len(result['all_summaries']), 192)
        for direction in result['directions']:
            self.assertEqual(direction['candidate'], 'c01')
            self.assertEqual(direction['setting']['weight'], .5)
            self.assertEqual(direction['seeds'], [42, 43, 44])
            self.assertAlmostEqual(direction['metrics']['N@10']['mean'], .5)
            self.assertAlmostEqual(direction['metrics']['N@10']['std'], .1)
            self.assertEqual([r['metrics']['N@10'] for r in direction['per_seed']], [.4, .5, .6])

    def test_final_ties_prefer_lower_candidate_pool_weight_and_none(self):
        for direction in policy.select_final(finals())['directions']:
            self.assertEqual(direction['candidate'], 'c00')
            self.assertEqual(direction['setting'], {"pool": 500,
                            "normalization": "none", "weight": 0})

    def test_final_cannot_mix_each_seeds_individually_best_setting(self):
        rows = finals()
        for row in rows:
            for index, result in enumerate(row['results']):
                if row['candidate'] == 'c00':
                    result['metrics'] = metrics(.9 if index == row['seed'] - 42 else .1)
                else:
                    result['metrics'] = metrics(.4)
        for direction in policy.select_final(rows)['directions']:
            self.assertEqual(direction['candidate'], 'c01')
            self.assertAlmostEqual(direction['metrics']['N@10']['mean'], .4)

    def test_final_lexicographic_priority_uses_all_four_metrics(self):
        for preferred in ('R@10', 'N@5', 'R@5'):
            rows = finals()
            for row in rows:
                if row['candidate'] == 'c01':
                    for result in row['results']:
                        result['metrics'][preferred] = .3
            with self.subTest(preferred=preferred):
                self.assertTrue(all(d['candidate'] == 'c01'
                                    for d in policy.select_final(rows)['directions']))

    def test_final_requires_12_jobs_two_candidates_and_three_seeds(self):
        invalid = [finals()[:-1], finals() + [copy.deepcopy(finals()[0])]]
        rows = finals()
        rows[0]['seed'] = 41
        invalid.append(rows)
        rows = finals()
        rows[0]['candidate'] = 'c02'
        invalid.append(rows)
        for rows in invalid:
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                policy.select_final(rows)

    def test_final_requires_all_48_unique_known_settings_for_every_model(self):
        for kind in ('missing', 'duplicate', 'unknown', 'extra_key'):
            rows = finals()
            results = rows[0]['results']
            if kind == 'missing':
                results.pop()
            elif kind == 'duplicate':
                results[-1] = copy.deepcopy(results[0])
            elif kind == 'unknown':
                results[0]['setting']['weight'] = .123
            else:
                results[0]['setting']['split'] = 'test'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                policy.select_final(rows)

    def test_bad_metrics_rejected_for_screen_and_final(self):
        for value in (math.nan, math.inf, -.1, 1.1, True, '0.2', None, 10**400):
            screens = screenings()
            screens[0]['metrics']['N@10'] = value
            rows = finals()
            rows[0]['results'][0]['metrics']['N@10'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.promote(screens)
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.select_final(rows)

    def test_example_counts_are_positive_integers_and_consistent_per_pair(self):
        for value in (0, -1, 1.5, True, 99):
            screens = screenings()
            screens[0]['metrics']['examples'] = value
            rows = finals()
            rows[0]['results'][0]['metrics']['examples'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.promote(screens)
            with self.subTest(value=value), self.assertRaises(ValueError):
                policy.select_final(rows)

    def test_metric_aliases_are_canonical_and_conflicts_rejected(self):
        screens = screenings()
        for row in screens:
            row['metrics'] = {k.replace('R@', 'HR@').replace('N@', 'NDCG@'): v
                              for k, v in row['metrics'].items()}
        self.assertEqual(policy.promote(screens)['sports_to_clothing'], ['c00', 'c01'])
        screens[0]['metrics']['N@10'] = .4
        with self.assertRaises(ValueError):
            policy.promote(screens)
        rows = finals()
        del rows[0]['results'][0]['metrics']['N@20']
        with self.assertRaises(ValueError):
            policy.select_final(rows)

    def test_examples_are_optional_and_directions_can_have_different_counts(self):
        rows = finals()
        for row in rows:
            for result in row['results']:
                if row['pair'] == 'sports_to_clothing':
                    del result['metrics']['examples']
                else:
                    result['metrics']['examples'] = 200
        self.assertEqual(len(policy.select_final(rows)['directions']), 2)


if __name__ == '__main__':
    unittest.main()
