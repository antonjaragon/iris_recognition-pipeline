#!/usr/bin/env python3
import argparse
import itertools
import shutil
import subprocess
import sys
from pathlib import Path

def infer_mask_path(code_path: Path, codes_dir: Path, masks_dir: Path) -> Path:
    """
    Map data/irisCode/.../<name>_irisCode.bmp
    ->  data/irisCode_masks/.../<name>_irisCode_mask.bmp
    while preserving the relative subfolders.
    """
    rel = code_path.relative_to(codes_dir)
    mask_name = rel.name.replace("_irisCode.bmp", "_irisCode_mask.bmp")
    return masks_dir / rel.parent / mask_name

def main():
    ap = argparse.ArgumentParser(
        description="Compare every pair of iris codes using `hd` and aggregate results."
    )
    ap.add_argument("--codes-dir", type=Path, default=Path("data/irisCode"),
                    help="Directory containing iris code images (default: data/irisCode)")
    ap.add_argument("--masks-dir", type=Path, default=Path("data/irisCode_masks"),
                    help="Directory containing iris code mask images (default: data/irisCode_masks)")
    ap.add_argument("--pattern", type=str, default="*.bmp",
                    help="Glob pattern for code images (default: *.bmp)")
    ap.add_argument("--output", type=Path, default=Path("data/distances/hamming_distance.txt"),
                    help="Aggregated output file (default: data/distances/hamming_distance.txt)")
    args = ap.parse_args()

    codes_dir = args.codes_dir
    masks_dir = args.masks_dir
    pattern = args.pattern
    out_file = args.output

    if not codes_dir.exists():
        print(f"[ERROR] Codes directory not found: {codes_dir}", file=sys.stderr)
        sys.exit(1)
    if not masks_dir.exists():
        print(f"[ERROR] Masks directory not found: {masks_dir}", file=sys.stderr)
        sys.exit(1)

    code_paths = sorted(codes_dir.rglob(pattern))
    if len(code_paths) < 2:
        print(f"[ERROR] Need at least two code images in {codes_dir} matching '{pattern}'", file=sys.stderr)
        sys.exit(1)

    # Fresh output
    out_file.parent.mkdir(parents=True, exist_ok=True)
    if out_file.exists():
        out_file.unlink()

    print(f"Found {len(code_paths)} code images under {codes_dir}.")
    total_pairs = (len(code_paths) * (len(code_paths) - 1)) // 2
    print(f"Total pairs to compare: {total_pairs}")

    tmp_dir = Path(".hd_tmp")
    tmp_dir.mkdir(exist_ok=True)

    pairs_done = 0
    try:
        for p1, p2 in itertools.combinations(code_paths, 2):
            m1 = infer_mask_path(p1, codes_dir, masks_dir)
            m2 = infer_mask_path(p2, codes_dir, masks_dir)

            if not m1.exists() or not m2.exists():
                missing = []
                if not m1.exists(): missing.append(str(m1))
                if not m2.exists(): missing.append(str(m2))
                print(f"[WARN] Missing mask(s); skipping pair:\n  {p1}\n  {p2}\n  Missing: {', '.join(missing)}")
                continue

            tmp_out = tmp_dir / "pair.txt"
            if tmp_out.exists():
                tmp_out.unlink()

            cmd = [
                "hd",
                "-i", str(p1), str(p2),
                "-m", str(m1), str(m2),
                "-o", str(tmp_out)
            ]

            pairs_done += 1
            print(f"[{pairs_done}/{total_pairs}] hd -i {p1.name} {p2.name} -m {m1.name} {m2.name}", flush=True)

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[ERROR] hd failed for:\n  {p1}\n  {p2}\n  -> {e}", file=sys.stderr)
                continue

            if tmp_out.exists():
                with tmp_out.open("r", encoding="utf-8", errors="ignore") as fin, \
                     out_file.open("a", encoding="utf-8") as fout:
                    shutil.copyfileobj(fin, fout)
                tmp_out.unlink(missing_ok=True)

        print(f"\n✅ Done. Aggregated results saved to: {out_file}")

    finally:
        try:
            tmp_dir.rmdir()
        except OSError:
            pass  # directory not empty (harmless)

if __name__ == "__main__":
    main()
