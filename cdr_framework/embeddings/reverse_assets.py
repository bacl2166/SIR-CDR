"""ID-safe frozen-vector reuse and resumable encoding for independent reverse assets."""
import json
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

import torch

from cdr_framework.datasets.reverse_assets import PAIRS, write_json
from cdr_framework.embeddings.qwen import load_item_texts
from experiments.prepare_gencdr_douban import sha256


def locations(pair, root):
    root=Path(root); forward=PAIRS[pair][2]
    return (root/'data/processed'/pair, root/'data/processed'/forward,
            root/'artifacts/embeddings/qwen3_8b'/pair, root/'artifacts/embeddings/qwen3_8b'/forward)


def _original(row):
    return row['domain'], row.get('original_id',row.get('asin'))


def build_plan(pair, root):
    processed, forward, output, source=locations(pair,root)
    mappings=json.loads((processed/'mappings.json').read_text(encoding='utf-8'))['id_to_item']
    items=load_item_texts(processed/'item_texts.jsonl')
    if [i.item_id for i in items] != list(range(1,len(mappings))):
        raise RuntimeError('Reverse text IDs and mapping rows differ')
    binding={'processed_manifest_sha256':sha256(processed/'reverse_manifest.json'),
             'mappings_sha256':sha256(processed/'mappings.json'),
             'input_sha256':sha256(processed/'item_texts.jsonl'), 'tool_sha256':sha256(__file__)}
    reuse=[0]*len(items); meta={}; source_hashes={}; compatible=False
    needed=[forward/'mappings.json',forward/'item_texts.jsonl',source/'manifest.json',source/'text_embeddings.pt']
    if all(p.is_file() for p in needed):
        source_hashes={str(p.relative_to(Path(root))):sha256(p) for p in needed}
        meta=json.loads((source/'manifest.json').read_text(encoding='utf-8'))
        if meta.get('input_sha256') != sha256(forward/'item_texts.jsonl'):
            raise RuntimeError('Forward vectors are not bound to current forward item texts')
        old_mapping=json.loads((forward/'mappings.json').read_text(encoding='utf-8'))['id_to_item']
        old_items=load_item_texts(forward/'item_texts.jsonl')
        old_by_key={_original(old_mapping[i.item_id]):i for i in old_items}
        if len(old_by_key)!=len(old_items): raise RuntimeError('Ambiguous forward domain/item identities')
        compatible=(meta.get('model')=='Qwen/Qwen3-Embedding-8B' and meta.get('dimension')==768
                    and meta.get('provider')=='qwen3_local' and meta.get('normalized') is True
                    and meta.get('max_sequence_length')==(512 if pair=='douban_movie_book' else 8192))
        complete=(meta.get('item_count')==len(old_items) and meta.get('matrix_shape')==[len(old_mapping),768]
                  and meta.get('padding_row_is_zero') is True
                  and meta.get('completed_batches')==(len(old_items)+meta.get('batch_size',1)-1)//meta.get('batch_size',1))
        if not complete: raise RuntimeError('Forward embedding manifest is incomplete')
        if compatible:
            for n,item in enumerate(items):
                previous=old_by_key.get(_original(mappings[item.item_id]))
                if previous is not None and previous.text==item.text:
                    reuse[n]=previous.item_id
    revision=meta.get('model_revision') if compatible else None
    if revision and (len(revision)!=40 or any(c not in '0123456789abcdef' for c in revision)):
        raise RuntimeError('Invalid source embedding model revision')
    if not revision and not all(reuse):
        # Missing model provenance forbids mixing copied legacy vectors with
        # newly computed vectors. Recompute the entire coherent matrix instead.
        reuse=[0]*len(items)
    runtime={}
    if not all(reuse):
        for name in ('torch','transformers','sentence-transformers'):
            try: runtime[name]=version(name)
            except PackageNotFoundError: runtime[name]='unavailable'
        runtime['provider_sha256']=sha256(Path(__file__).with_name('qwen3_local.py'))
    plan={'schema':'reverse-vector-plan-v1','pair':pair,'binding':binding,'source_hashes':source_hashes,
          'model':'Qwen/Qwen3-Embedding-8B','dimension':768,'batch_size':2 if pair=='douban_movie_book' else 1,
          'max_sequence_length':512 if pair=='douban_movie_book' else 8192,
          'model_revision':revision, 'source_model_config_sha256':meta.get('model_config_sha256') if any(reuse) else None,
          'new_encoding_runtime':runtime, 'reuse_ids':reuse, 'reused_items':sum(v>0 for v in reuse),
          'new_items':sum(v==0 for v in reuse),
          'revision_provenance':('pinned-source-model-revision' if revision else
              'legacy-frozen-matrix-unrecorded-model-revision' if all(reuse) else 'fresh-pinned-encoding-required')}
    output.mkdir(parents=True,exist_ok=True)
    path=output/'reuse_plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:
            raise RuntimeError('Reverse embedding plan changed; refusing to reuse incompatible outputs')
    else: write_json(path,plan)
    print(json.dumps({k:plan[k] for k in ('pair','reused_items','new_items','revision_provenance')}),flush=True)
    return plan


def _tensor_atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.part'); torch.save(value,temporary); temporary.replace(path)


def materialize(pair, root, *, allow_encode, provider=None, revision=None):
    processed, _, output, source=locations(pair,root)
    plan_path=output/'reuse_plan.json'; plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['binding']['tool_sha256']!=sha256(__file__): raise RuntimeError('Vector materializer code changed')
    for key,name in (('processed_manifest_sha256','reverse_manifest.json'),('mappings_sha256','mappings.json'),('input_sha256','item_texts.jsonl')):
        if sha256(processed/name)!=plan['binding'][key]: raise RuntimeError('Reverse vector inputs changed')
    for name,digest in plan['source_hashes'].items():
        if sha256(Path(root)/name)!=digest: raise RuntimeError('Frozen forward vector source changed')
    if plan['new_items'] and not allow_encode:
        print(f'{pair}: {plan["new_items"]} vectors await the GPU stage',flush=True); return None
    if plan['new_items'] and (provider is None or not revision or len(revision)!=40):
        raise RuntimeError('New encoding requires a provider and immutable model revision')
    if plan['model_revision'] and revision and revision!=plan['model_revision']:
        raise RuntimeError('Mixed encoding model revision differs from copied vectors')
    effective_revision=revision or plan['model_revision']
    identity={'schema':'reverse-embedding-v1','plan_sha256':sha256(plan_path),
              'model_revision':effective_revision,'execution':'mixed-or-new' if plan['new_items'] else 'frozen-copy',
              'tool_sha256':sha256(__file__)}
    manifest_path=output/'manifest.json'; matrix_path=output/'text_embeddings.pt'
    if manifest_path.exists():
        existing=json.loads(manifest_path.read_text(encoding='utf-8'))
        if existing.get('identity')!=identity or existing.get('matrix_sha256')!=sha256(matrix_path):
            raise RuntimeError('Reverse embedding output identity/hash mismatch')
        print(f'Reusing complete reverse vectors: {pair}',flush=True); return matrix_path
    items=load_item_texts(processed/'item_texts.jsonl')
    batch=plan['batch_size']; total=(len(items)+batch-1)//batch
    def finish(matrix):
        if not torch.isfinite(matrix).all() or (matrix[1:].norm(dim=1)==0).any():
            raise RuntimeError('Reverse matrix is incomplete or nonfinite')
        _tensor_atomic(matrix_path,matrix)
        write_json(output/'item_index.json',{'padding_item_id':0,'row_is_item_id':True,'item_ids':[r.item_id for r in items]})
        write_json(output/'progress.json',{'identity':identity,'completed_batches':total,'total_batches':total,
                   'item_count':len(items),'reused_items':plan['reused_items'],'new_items':plan['new_items']})
        write_json(manifest_path,{'complete':True,'identity':identity,'matrix_sha256':sha256(matrix_path),
                   'matrix_shape':list(matrix.shape),'input_sha256':plan['binding']['input_sha256'],
                   'model':plan['model'],'model_revision':effective_revision,'dimension':768,
                   'reused_items':plan['reused_items'],'new_items':plan['new_items'],
                   'revision_provenance':plan['revision_provenance']})
        return matrix_path
    old=None
    if plan['reused_items']:
        old=torch.load(source/'text_embeddings.pt',map_location='cpu',weights_only=True)
        if old.ndim!=2 or old.shape[1]!=768 or not torch.isfinite(old).all() or old[0].abs().sum()!=0:
            raise RuntimeError('Frozen source matrix shape, padding or values invalid')
    if not plan['new_items']:
        # One dense permutation, avoiding tens of thousands of tiny chunk files
        # for an entirely reusable legacy Amazon matrix.
        return finish(old[torch.tensor([0]+plan['reuse_ids'])].clone())
    chunks=output/'chunks'; chunks.mkdir(exist_ok=True)
    progress=output/'progress.json'
    if progress.exists() and json.loads(progress.read_text(encoding='utf-8')).get('identity')!=identity:
        raise RuntimeError('Reverse embedding resume identity mismatch')
    for number,start in enumerate(range(0,len(items),batch)):
        rows=items[start:start+batch]; ids=[r.item_id for r in rows]
        path=chunks/f'batch-{number:06d}.pt'
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if (saved.get('identity')!=identity or saved['item_ids'].tolist()!=ids
                    or saved['embeddings'].shape!=(len(rows),768) or not torch.isfinite(saved['embeddings']).all()):
                raise RuntimeError('Reverse cached chunk identity/shape mismatch')
        else:
            vectors=torch.empty(len(rows),768,dtype=torch.float32); missing=[]
            for n,row in enumerate(rows):
                previous=plan['reuse_ids'][start+n]
                if previous: vectors[n]=old[previous]
                else: missing.append(n)
            if missing:
                encoded=provider.encode_text([rows[n].text for n in missing]).detach().cpu().float()
                if encoded.shape!=(len(missing),768) or not torch.isfinite(encoded).all():
                    raise RuntimeError('New reverse embedding values invalid')
                vectors[missing]=encoded
            _tensor_atomic(path,{'identity':identity,'item_ids':torch.tensor(ids),'embeddings':vectors})
        if (number+1)%100==0 or number+1==total:
            write_json(progress,{'identity':identity,'completed_batches':number+1,'total_batches':total,
                       'item_count':len(items),'reused_items':plan['reused_items'],'new_items':plan['new_items']})
            print(f'Reverse vectors {pair}: {number+1}/{total}',flush=True)
    matrix=torch.zeros(len(items)+1,768)
    for path in sorted(chunks.glob('batch-*.pt')):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['identity']!=identity: raise RuntimeError('Unexpected chunk identity')
        matrix[saved['item_ids']]=saved['embeddings']
    return finish(matrix)
