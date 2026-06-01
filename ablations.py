"""
Runs LANTERN ablations via subprocess.

Ablations:
- A0_full:             Time2Vec + Attr-Attn (default)
- A1_no_attr_attention: removes attribute attention (mean pool attrs)
- A2_no_time2vec:       removes Time2Vec (t_self zeros)
- A3_no_both:           removes both
"""

import argparse
import json
import shlex
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Ablation:
    tag: str
    args: Dict[str, Any]


def make_ablations() -> List[Ablation]:
    return [
        # Ablation(tag="A0_full", args={"_raw_flags": []}),
        Ablation(tag="A1_no_attr_attn_attrs_as_covariates", args={"_raw_flags": ["--ablate-attr-attention"]}),
        Ablation(tag="A2_no_time_irregularity", args={"_raw_flags": ["--ablate-time2vec"]}),
        Ablation(tag="A3_no_both", args={"_raw_flags": ["--ablate-attr-attention", "--ablate-time2vec"]}),
    ]


def _to_flag_name(k: str) -> str:
    return "--" + k.replace("_", "-")


def build_cmd(main_script: str, base_args: Dict[str, Any], ablation_args: Dict[str, Any], run_output_dir: Path) -> List[str]:
    cmd = ["python", main_script, "--train", "--eval"]

    merged = dict(base_args)
    merged.update(ablation_args)

    merged["output_dir"] = str(run_output_dir)

    raw_flags = merged.pop("_raw_flags", []) or []

    for k, v in merged.items():
        if v is None:
            continue
        if isinstance(v, bool):
            raise ValueError(f"Boolean arg '{k}' provided ({v}). Use _raw_flags for explicit flags.")
        if isinstance(v, (list, tuple)):
            cmd.append(_to_flag_name(k))
            cmd.extend([str(x) for x in v])
        else:
            cmd.extend([_to_flag_name(k), str(v)])

    cmd.extend(list(raw_flags))
    return cmd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--main-script", type=str, required=True, help="Path to your LANTERN main script.")
    ap.add_argument("--csv-path", type=str, default="Final_Preprocessed_RAND_LTCI_LONG.csv")
    ap.add_argument("--output-root", type=str, default="ABLATION_RUNS_LANTERN")
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--split-path", type=str, default="", help="Path to .npz split file to reuse across runs.")
    return ap.parse_args()


def main():
    args = parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    base_args: Dict[str, Any] = {"csv_path": args.csv_path}
    if args.split_path:
        base_args["split_path"] = args.split_path

    base_raw = []
    if args.skip_baselines:
        base_raw.append("--skip-baselines")

    ablations = make_ablations()

    manifest = {
        "created_at": datetime.now().isoformat(),
        "main_script": args.main_script,
        "csv_path": args.csv_path,
        "output_root": str(output_root),
        "runs": [],
    }

    failures = []

    for abl in ablations:
        run_id = abl.tag
        run_dir = output_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        merged_args = dict(abl.args)
        merged_args["_raw_flags"] = base_raw + list(merged_args.get("_raw_flags", []))

        cmd = build_cmd(
            main_script=args.main_script,
            base_args={k: v for k, v in base_args.items()},
            ablation_args=merged_args,
            run_output_dir=run_dir,
        )

        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        print("\n" + "=" * 90)
        print(f"RUN: {run_id}")
        print(cmd_str)

        manifest["runs"].append(
            {"run_id": run_id, "tag": abl.tag, "output_dir": str(run_dir), "cmd": cmd}
        )

        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Run failed: {run_id} (exit={e.returncode})")
            failures.append({"run_id": run_id, "exit_code": e.returncode, "cmd": cmd})
            if args.fail_fast:
                break

    manifest_path = output_root / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "-" * 90)
    print(f"[Saved] {manifest_path}")

    if failures:
        print("\n[Some runs failed]")
        for f in failures:
            print(f" - {f['run_id']}: exit {f['exit_code']}")
        raise SystemExit(2)

    print("\n[All runs completed successfully]")


if __name__ == "__main__":
    main()