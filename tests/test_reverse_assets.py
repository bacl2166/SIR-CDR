import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import gzip

from test_douban_pipeline import raw_fixture
from cdr_framework.datasets.reverse_assets import prepare_douban_reverse, prepare_amazon_reverse, verify_reverse
from experiments.prepare_gencdr_douban import sha256


class ReverseDatasetTests(unittest.TestCase):
    def test_amazon_raw_data_gets_new_target_holdouts(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); raw=root/'data/raw/amazon2014'; raw.mkdir(parents=True)
            for domain,prefix,start in (('Sports_and_Outdoors','S',1),('Clothing_Shoes_and_Jewelry','C',2)):
                for kind in ('reviews','meta'):
                    name=f'reviews_{domain}_5.json.gz' if kind=='reviews' else f'meta_{domain}.json.gz'
                    with gzip.open(raw/name,'wt',encoding='utf-8') as stream:
                        for i in range(5):
                            row={'asin':f'{prefix}{i}'}
                            if kind=='reviews': row.update(reviewerID='u',unixReviewTime=start+2*i)
                            else: row['title']=f'Static {prefix}{i}'
                            stream.write(json.dumps(row)+'\n')
            out=prepare_amazon_reverse('clothing_to_sports',root)
            mapping=json.loads((out/'mappings.json').read_text())
            validation=json.loads((out/'validation.jsonl').read_text())
            test=json.loads((out/'test.jsonl').read_text())
            self.assertEqual(mapping['source_domain'],'Clothing_Shoes_and_Jewelry')
            self.assertEqual(validation['positive_target_item'],mapping['item_to_id']['Sports_and_Outdoors:S3'])
            self.assertEqual(test['positive_target_item'],mapping['item_to_id']['Sports_and_Outdoors:S4'])
            self.assertTrue(verify_reverse(out))

    def test_douban_holdouts_and_tags_follow_reverse_training(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); original_raw = raw_fixture(root)
            raw = root / 'reverse.zip'
            with zipfile.ZipFile(original_raw) as src, zipfile.ZipFile(raw, 'w') as dst:
                for name in src.namelist():
                    content = src.read(name).decode('utf-8')
                    if name.endswith('moviereviews_cleaned.txt'):
                        content = content.replace('2020-01-05', '2020-01-01')
                    if name.endswith('bookreviews_cleaned.txt'):
                        content = content.replace('2020-01-07\tTRAIN_TAG', '2020-01-07\tVALIDATION_ONLY_LABEL')
                        content = content.replace('2020-01-04\tTRAIN_TAG', '2020-01-04\tLAST_TRAIN_POSITIVE_TAG')
                    dst.writestr(name, content)
            out = prepare_douban_reverse(raw, root / 'reverse', expected_sha256=sha256(raw))
            mapping = json.loads((out / 'mappings.json').read_text())
            self.assertEqual((mapping['source_domain'], mapping['target_domain']), ('Movie', 'Book'))
            # The original forward split cannot be obtained by swapping columns:
            # reverse validation/test positives must now be Book items.
            rows = {s: [json.loads(v) for v in (out / f'{s}.jsonl').read_text().splitlines()]
                    for s in ('train', 'validation', 'test')}
            self.assertEqual(rows['validation'][0]['positive_target_item'], mapping['item_to_id']['Book:45'])
            self.assertEqual(rows['test'][0]['positive_target_item'], mapping['item_to_id']['Book:46'])
            texts = (out / 'item_texts.jsonl').read_text()
            self.assertNotIn('FUTURE_LABEL', texts)
            self.assertNotIn('VALIDATION_ONLY_LABEL', texts)
            self.assertIn('LAST_TRAIN_POSITIVE_TAG', texts)
            self.assertNotIn('TARGET_LABEL_MUST_NOT_ENTER_TEXT', texts)
            self.assertNotIn('FORBIDDEN_TAG', texts)
            self.assertTrue(verify_reverse(out))
            original = (out / 'reverse_manifest.json').read_bytes()
            prepare_douban_reverse(raw, out, expected_sha256=sha256(raw))
            self.assertEqual(original, (out / 'reverse_manifest.json').read_bytes())


if __name__ == '__main__':
    unittest.main()
