import argparse
import json
import sys
from pathlib import Path
from dataclasses import replace
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cdr_framework.text_config import TextCDRConfig
from cdr_framework.text_training import VARIANTS, variant_config, train, evaluate_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Text-only SIR-CDR v4 training, ablation and full-target evaluation")
    parser.add_argument("--config", default=str(ROOT / "configs/text_sports_clothing.yaml"))
    parser.add_argument("--action", choices=("train", "evaluate", "suite"), default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=VARIANTS, default="full")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--mode", choices=("retrieval", "hybrid", "exhaustive", "generate"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--allow-code-mismatch", action="store_true",
                        help="Inference-only: accept newer code while config/artifacts are still verified.")
    args = parser.parse_args()
    config = TextCDRConfig.from_yaml(args.config)
    config = replace(config, **{name: value if value.is_absolute() else ROOT / value
        for name in ("processed_dir", "tokenizer_dir", "output_dir") for value in [getattr(config, name)]})
    if args.output_root:
        config = replace(config, output_dir=args.output_root.resolve())
    results = []
    for variant in args.variants if args.action == "suite" else [args.variant]:
        for seed in args.seeds:
            current = variant_config(config, variant, seed)
            print(f"Variant={variant} seed={seed} output={current.output_dir}", flush=True)
            if args.action == "evaluate":
                result = evaluate_checkpoint(current, torch.device(args.device), args.split, args.mode,
                                             code_check=not args.allow_code_mismatch)
            else:
                result = train(current, torch.device(args.device))
            results.append({"variant": variant, "seed": seed, "result": result})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
