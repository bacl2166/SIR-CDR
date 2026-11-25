"""Read-only boundary: no download, converter, sampling or split construction."""
import json
from pathlib import Path

import torch

from cdr_framework.datasets.gencdr_protocol import verify_shared, sha, SCHEMA


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def load_rows(path, sources, targets):
    rows = []
    with Path(path).open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            s, t, y = row['source_items'], row['target_items'], row['positive_target_item']
            if (not s or not t or type(y) is not int or y not in targets
                    or any(type(i) is not int or i not in sources for i in s)
                    or any(type(i) is not int or i not in targets for i in t)):
                raise ValueError(f'Invalid existing sample: {path}:{line_no}')
            # Existing protocol retains repeat events. A seen positive remains
            # a training label and is masked/counts as a miss during evaluation.
            rows.append(row)
    if not rows:
        raise ValueError(f'Empty existing split: {path}')
    return rows


class ExistingAssets:
    def __init__(self, root, pair, *, verify=True):
        self.root, self.pair = Path(root), pair
        self.processed = self.root / 'data/processed' / SCHEMA / pair
        self.embedding = self.root / 'artifacts/embeddings' / SCHEMA / pair
        if verify:
            verify_shared(self.processed)
        self.mapping = read(self.processed / 'mappings.json')
        self.stats = read(self.processed / 'statistics.json')
        self.names = (self.mapping['source_domain'], self.mapping['target_domain'])
        self.ids = self.mapping['id_to_item']
        self.sources = {i for i, item in enumerate(self.ids) if item and item['domain'] == self.names[0]}
        self.targets = {i for i, item in enumerate(self.ids) if item and item['domain'] == self.names[1]}
        if len(self.sources | self.targets) != len(self.ids) - 1 or self.sources & self.targets:
            raise RuntimeError('Catalog/domain mapping inconsistent')
        self.vectors = torch.load(self.embedding / 'text_embeddings.pt', map_location='cpu', weights_only=True).float()
        if (self.vectors.ndim != 2 or len(self.vectors) != len(self.ids)
                or not torch.isfinite(self.vectors).all() or self.vectors[0].count_nonzero()):
            raise RuntimeError('Existing embeddings/mapping inconsistent')
        self.domains = torch.zeros(len(self.ids), dtype=torch.long)
        self.domains[list(self.targets)] = 1
        manifest = read(self.embedding / 'manifest.json')
        if (manifest.get('matrix_shape') != list(self.vectors.shape)
                or manifest.get('item_count') != len(self.ids)-1
                or manifest.get('input_sha256') != sha(self.processed / 'item_texts.jsonl')
                or manifest.get('completed_batches') != (len(self.ids)-2+manifest['batch_size'])//manifest['batch_size']):
            raise RuntimeError('Existing embedding provenance incomplete')

    def rows(self, split):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('Unknown frozen split')
        result = load_rows(self.processed / f'{split}.jsonl', self.sources, self.targets)
        if len(result) != self.stats[f'{split}_rows']:
            raise RuntimeError('Existing sample count changed')
        return result

    def seal(self):
        paths = [self.processed / 'shared_manifest.json', self.embedding / 'manifest.json',
                 self.embedding / 'text_embeddings.pt']
        paths += [self.processed / name for name in read(self.processed / 'shared_manifest.json')['files']]
        return {str(p.relative_to(self.root)): sha(p) for p in paths}

    def reverse_training(self, reverse):
        """Read the other existing direction, translate IDs in RAM, never create rows."""
        if self.names != reverse.names[::-1]:
            raise ValueError('Reverse direction domains differ')
        def identity(item):
            return item['domain'], item.get('original_id', item.get('asin'))
        canonical = {identity(item): i for i, item in enumerate(self.ids) if item}
        translate = {i: canonical[identity(item)] for i, item in enumerate(reverse.ids) if item}
        if len(translate) != len(canonical):
            raise RuntimeError('Reverse catalogs differ')
        result = []
        for row in reverse.rows('train'):
            r = dict(row)
            r.update(source_items=[translate[i] for i in row['source_items']],
                     target_items=[translate[i] for i in row['target_items']],
                     positive_target_item=translate[row['positive_target_item']])
            result.append(r)
        return result
