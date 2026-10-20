import json
from pathlib import Path
import tempfile
import unittest

import torch

from cdr_framework.datasets.reverse_assets import write_json
from cdr_framework.embeddings.reverse_assets import build_plan, materialize
from experiments.prepare_gencdr_douban import sha256


def embedding_fixture(root, changed=False, revision=None):
    forward = root / 'data/processed/sports_to_clothing'
    reverse = root / 'data/processed/clothing_to_sports'
    output = root / 'artifacts/embeddings/qwen3_8b/sports_to_clothing'
    for path in (forward, reverse, output): path.mkdir(parents=True)
    entries = [None, {'domain':'Sports_and_Outdoors','asin':'same'},
               {'domain':'Clothing_Shoes_and_Jewelry','asin':'same'}]
    for path, rows in ((forward, entries), (reverse, [None, entries[2], entries[1]])):
        write_json(path / 'mappings.json', {'id_to_item': rows})
        with (path / 'item_texts.jsonl').open('w') as stream:
            for i, row in enumerate(rows[1:], 1):
                text = row['domain'] + (' changed' if changed and path == reverse and i == 2 else '')
                stream.write(json.dumps({'item_id':i,'domain':row['domain'],'text':text})+'\n')
    write_json(reverse / 'reverse_manifest.json', {'complete': True, 'identity':{}, 'files':{}})
    matrix = torch.zeros(3,768); matrix[1,0]=1; matrix[2,1]=1
    torch.save(matrix, output / 'text_embeddings.pt')
    meta = {'model':'Qwen/Qwen3-Embedding-8B', 'dimension':768, 'provider':'qwen3_local',
            'normalized':True,'max_sequence_length':8192,'batch_size':1,
            'input_sha256':sha256(forward / 'item_texts.jsonl'), 'item_count':2,
            'matrix_shape':[3,768], 'padding_row_is_zero':True, 'completed_batches':2}
    if revision: meta['model_revision']=revision
    write_json(output / 'manifest.json', meta)
    return reverse, matrix


class ReverseEmbeddingTests(unittest.TestCase):
    def test_incompatible_source_model_is_not_a_pin_for_new_encoding(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); embedding_fixture(root,revision='a'*40)
            path=root/'artifacts/embeddings/qwen3_8b/sports_to_clothing/manifest.json'
            meta=json.loads(path.read_text()); meta['model']='different/model'
            write_json(path,meta)
            plan=build_plan('clothing_to_sports',root)
            self.assertEqual(plan['reuse_ids'],[0,0])
            self.assertIsNone(plan['model_revision'])
            self.assertTrue(plan['new_encoding_runtime']['provider_sha256'])

    def test_full_legacy_reuse_permuted_by_domain_and_original_id(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); _, old=embedding_fixture(root)
            plan=build_plan('clothing_to_sports',root)
            self.assertEqual(plan['reuse_ids'],[2,1])
            self.assertEqual(plan['revision_provenance'],'legacy-frozen-matrix-unrecorded-model-revision')
            path=materialize('clothing_to_sports',root,allow_encode=False)
            actual=torch.load(path,map_location='cpu',weights_only=True)
            self.assertTrue(torch.equal(actual,old[[0,2,1]]))
            before=path.read_bytes()
            materialize('clothing_to_sports',root,allow_encode=False)
            self.assertEqual(before,path.read_bytes())

    def test_unknown_revision_cannot_mix_copied_and_new_vectors(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); embedding_fixture(root,changed=True)
            plan=build_plan('clothing_to_sports',root)
            self.assertEqual(plan['reuse_ids'],[0,0])
            self.assertIsNone(materialize('clothing_to_sports',root,allow_encode=False))

    def test_pinned_partial_reuse_encodes_only_changed_text(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); _,old=embedding_fixture(root,changed=True,revision='a'*40)
            build_plan('clothing_to_sports',root)
            class Provider:
                def encode_text(self,texts):
                    self.texts=texts
                    result=torch.zeros(len(texts),768); result[:,2]=1
                    return result
            provider=Provider()
            path=materialize('clothing_to_sports',root,allow_encode=True,provider=provider,revision='a'*40)
            matrix=torch.load(path,map_location='cpu',weights_only=True)
            self.assertTrue(torch.equal(matrix[1],old[2]))
            self.assertEqual(provider.texts,['Sports_and_Outdoors changed'])
            self.assertEqual(matrix[2,2].item(),1.)

    def test_interrupted_encoding_resumes_and_finished_matrix_is_hash_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); embedding_fixture(root,changed=True,revision='a'*40)
            build_plan('clothing_to_sports',root)
            class Provider:
                def encode_text(self,texts): raise RuntimeError('simulated interruption')
            with self.assertRaisesRegex(RuntimeError,'interruption'):
                materialize('clothing_to_sports',root,allow_encode=True,provider=Provider(),revision='a'*40)
            chunk=root/'artifacts/embeddings/qwen3_8b/clothing_to_sports/chunks/batch-000000.pt'
            before=chunk.read_bytes()
            class Resumed:
                def encode_text(self,texts):
                    values=torch.zeros(len(texts),768); values[:,3]=1; return values
            path=materialize('clothing_to_sports',root,allow_encode=True,provider=Resumed(),revision='a'*40)
            self.assertEqual(before,chunk.read_bytes())
            path.write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'):
                materialize('clothing_to_sports',root,allow_encode=True,provider=Resumed(),revision='a'*40)


if __name__ == '__main__': unittest.main()
