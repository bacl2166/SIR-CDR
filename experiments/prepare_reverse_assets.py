"""Parallel CPU preparation and separate serial GPU assets for three reverse tasks.

Never launches recommender training or writes any forward data/model outputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

from cdr_framework.datasets.reverse_assets import (
    PAIRS, prepare_amazon_reverse, prepare_douban_reverse, verify_reverse, write_json)
from cdr_framework.embeddings.reverse_assets import build_plan, materialize, locations
from experiments.prepare_gencdr_douban import sha256


def tokenizer_config(pair,root):
    from cdr_framework.experiment_config import TokenizerTrainingConfig
    base=ROOT/('configs/douban_book_movie.yaml' if pair=='douban_movie_book' else
               f'configs/amazon_{"sports_clothing" if pair=="clothing_to_sports" else "phones_electronics"}.yaml')
    cfg=TokenizerTrainingConfig.from_yaml(base); root=Path(root)
    return replace(cfg,embeddings_path=root/'artifacts/embeddings/qwen3_8b'/pair/'text_embeddings.pt',
        item_texts_path=root/'data/processed'/pair/'item_texts.jsonl',
        output_dir=root/'artifacts/tokenizer/qwen3_8b'/pair)


def tokenize(pair,root,device):
    from experiments.train_tokenizer import run
    cfg=tokenizer_config(pair,root)
    receipt=Path(root)/'artifacts/reverse_assets'/pair/'tokenizer_execution.json'
    identity={'config':{k:str(v) if isinstance(v,Path) else v for k,v in asdict(cfg).items()},
              'input_sha256':sha256(cfg.embeddings_path),'texts_sha256':sha256(cfg.item_texts_path),
              'domains':list(PAIRS[pair][:2]), 'driver_sha256':sha256(__file__)}
    quality=cfg.output_dir/'quality_report.json'
    complete=quality.exists() and json.loads(quality.read_text()).get('passed') is True
    if receipt.exists():
        previous=json.loads(receipt.read_text(encoding='utf-8'))
        if previous['identity']!=identity or (not complete and previous['device']!=device):
            raise RuntimeError('Reverse tokenizer identity/device changed during an incomplete run')
    else:
        receipt.parent.mkdir(parents=True,exist_ok=True)
        write_json(receipt,{'identity':identity,'device':device})
    run(cfg,device=device,force=False,source_domain=PAIRS[pair][0],target_domain=PAIRS[pair][1])
    if not json.loads(quality.read_text()).get('passed'):
        raise RuntimeError('Reverse tokenizer quality gate did not pass')


def _forward_workers():
    if not Path('/proc').is_dir(): return []
    active=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid(): continue
        try: arguments=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        if any(name in arguments for name in ('experiments/run_douban.py','experiments/run_llm_cdr.py',
                                               'experiments/run_llm_fast.py','experiments/recover_llm_evaluation.py')):
            active.append(process.name)
    return active


def ensure_gpu_idle():
    workers=_forward_workers()
    if workers: raise RuntimeError(f'Forward recommendation workflow is active (PIDs {workers}); run CPU stage only')
    try:
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'],text=True)
    except (OSError,subprocess.CalledProcessError) as error:
        raise RuntimeError('Cannot confirm GPU idle state; GPU preparation refused') from error
    lines=[s.strip() for s in output.splitlines() if s.strip()]
    if any(not s.isdigit() for s in lines):
        raise RuntimeError('GPU process inventory is ambiguous; GPU preparation refused')
    active=[s for s in lines if int(s)!=os.getpid()]
    if active: raise RuntimeError(f'GPU has active compute processes {active}; no new 8B/tokenizer process started')


def _provider(pair,root,plan,device):
    from huggingface_hub import snapshot_download
    from cdr_framework.embeddings import Qwen3EmbeddingProvider
    receipt=Path(root)/'artifacts/reverse_assets'/pair/'embedding_model.json'
    revision=plan['model_revision']
    if receipt.exists():
        value=json.loads(receipt.read_text(encoding='utf-8'))
        if value['model']!=plan['model'] or (revision and value['revision']!=revision):
            raise RuntimeError('Reverse model receipt differs')
        revision=value['revision']
    snapshot=Path(snapshot_download(plan['model'],revision=revision or 'main'))
    if len(snapshot.name)!=40: raise RuntimeError('Embedding model snapshot is not pinned')
    if revision and snapshot.name!=revision: raise RuntimeError('Embedding model revision differs')
    if plan.get('source_model_config_sha256') and sha256(snapshot/'config.json')!=plan['source_model_config_sha256']:
        raise RuntimeError('Pinned model config differs from the reused source encoder')
    if not receipt.exists():
        receipt.parent.mkdir(parents=True,exist_ok=True)
        write_json(receipt,{'model':plan['model'],'revision':snapshot.name})
    os.environ['QWEN3_REVERSE_EMBEDDING_PATH']=str(snapshot)
    provider=Qwen3EmbeddingProvider(model=plan['model'],dim=768,batch_size=plan['batch_size'],device=device,
        max_sequence_length=plan['max_sequence_length'],model_path_env='QWEN3_REVERSE_EMBEDDING_PATH')
    return provider,snapshot.name


def worker(pair,action,root,device,cpu_tokenizer):
    import torch
    from experiments.recover_llm_evaluation import operation_lock
    root=Path(root); torch.set_num_threads(1)
    with operation_lock(root/'artifacts/reverse_assets'/pair/'worker_lock'):
        if action=='cpu':
            print(f'=== {pair}: REBUILD REVERSE SPLITS AND TEXT ===',flush=True)
            if pair=='douban_movie_book':
                prepare_douban_reverse(root/'data/raw/douban_fengzhu_v1.zip',root/'data/processed'/pair)
            else: prepare_amazon_reverse(pair,root)
        verify_reverse(root/'data/processed'/pair)
        print(f'=== {pair}: VERIFY VECTOR REUSE ===',flush=True)
        plan=build_plan(pair,root)
        if action=='gpu' and plan['new_items']:
            ensure_gpu_idle()
            provider,revision=_provider(pair,root,plan,device)
            materialize(pair,root,allow_encode=True,provider=provider,revision=revision)
            del provider
            import gc
            gc.collect(); torch.cuda.empty_cache()
        else:
            materialize(pair,root,allow_encode=False)
        if action=='gpu':
            ensure_gpu_idle()
            tokenize(pair,root,device)
        elif cpu_tokenizer and plan['new_items']==0:
            tokenize(pair,root,'cpu')
    print(f'=== REVERSE {pair} {action.upper()} COMPLETED ===',flush=True)


def launch(pair,action,root,*,device='cuda',cpu_tokenizer=False):
    root=Path(root); log=root/'logs/reverse_assets'/f'{pair}_{action}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).resolve()),'--action',action,'--worker',pair,
             '--root',str(root),'--device',device]
    if cpu_tokenizer: command.append('--cpu-tokenizer')
    environment=os.environ.copy()
    environment.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                       TOKENIZERS_PARALLELISM='false',PYTHONIOENCODING='utf-8')
    with log.open('a',encoding='utf-8') as stream:
        stream.write(f'\n=== START {pair} {action} ===\n'); stream.flush()
        process=subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=environment)
        print(f'{pair}: PID={process.pid} LOG={log}',flush=True)
        return process.wait()


def parallel_cpu(pairs,workers,root,*,cpu_tokenizer=False):
    def prepare(pair): return pair,launch(pair,'cpu',root,cpu_tokenizer=cpu_tokenizer)
    with ThreadPoolExecutor(max_workers=workers) as pool: results=dict(pool.map(prepare,pairs))
    destination=Path(root)/'artifacts/reverse_assets'; destination.mkdir(parents=True,exist_ok=True)
    write_json(destination/'cpu_summary.json',{'exit_codes':results,'recommendation_training_started':False})
    if any(results.values()): raise RuntimeError(f'Reverse preparation failed: {results}; inspect per-task logs')


def status(pairs,root):
    for pair in pairs:
        processed,_,embeddings,_=locations(pair,root)
        report={'pair':pair,'processed':(processed/'reverse_manifest.json').exists()}
        for name,path,keys in (
            ('reuse_plan',embeddings/'reuse_plan.json',('reused_items','new_items','revision_provenance')),
            ('embedding',embeddings/'manifest.json',('complete','matrix_shape','reused_items','new_items')),
            ('tokenizer',Path(root)/'artifacts/tokenizer/qwen3_8b'/pair/'quality_report.json',('passed','collision_rate'))):
            data=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
            report[name]={k:data[k] for k in keys if k in data} or 'pending'
        print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action',choices=('cpu','gpu','all','status'),default='cpu')
    parser.add_argument('--pairs',nargs='+',choices=tuple(PAIRS),default=list(PAIRS))
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--cpu-tokenizer',action='store_true',help='CPU-train tokenizer when all vectors were copied')
    parser.add_argument('--worker',choices=tuple(PAIRS),help=argparse.SUPPRESS)
    parser.add_argument('--root',type=Path,default=ROOT,help=argparse.SUPPRESS)
    args=parser.parse_args()
    if not 1<=args.workers<=3: parser.error('--workers must be 1..3')
    if len(set(args.pairs))!=len(args.pairs): parser.error('Duplicate reverse tasks')
    if args.worker:
        if args.action not in ('cpu','gpu'): parser.error('Worker requires cpu or gpu')
        worker(args.worker,args.action,args.root,args.device,args.cpu_tokenizer); return
    if args.action=='status': status(args.pairs,args.root); return
    from experiments.recover_llm_evaluation import operation_lock
    with operation_lock(args.root/'artifacts/reverse_assets/controller'):
        if args.action in ('cpu','all'): parallel_cpu(args.pairs,args.workers,args.root,cpu_tokenizer=args.cpu_tokenizer)
        if args.action in ('gpu','all'):
            ensure_gpu_idle()
            for pair in args.pairs:
                if launch(pair,'gpu',args.root,device=args.device):
                    raise RuntimeError(f'Reverse GPU assets failed: {pair}; inspect per-task log')
    print('=== REVERSE ASSET PREPARATION COMPLETED ===',flush=True)


if __name__=='__main__': main()
