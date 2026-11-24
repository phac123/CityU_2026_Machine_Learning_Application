"""Render the ten-molecule show case of the report (Section 5.5).

    python tools/showcase.py --all            # all three branches, newest checkpoints
    python tools/showcase.py --method edm     # one branch

For every branch this writes

    <out>/showcase_<method>.png   a 5 x 2 grid of RDKit depictions, one per
                                              sample, with a legend under each drawing
    <out>/showcase_table.tex      the LaTeX body of the summary table
    <out>/showcase_<method>.json  everything the figure shows, for auditing

The samples are the *same ten draws* as Table 20/21 of the report: by default the script
reads the indices from ``utils/report_numbers.json`` (seed 0, indices 4, 10, 19, ...) and
reuses the ``test_eval/samples.{smi,xyz}`` files the evaluation already produced, so no
GPU is needed when those files exist. With ``--resample`` it calls ``evaluate.py`` first.

For the two geometric branches the molecule is rebuilt the way the metrics rebuild it:
bonds are inferred from the distances with ``molgen.<method>.metrics.infer_bonds`` and the
molecule is drawn from that graph, so *what is drawn is what was scored*. A sample that
RDKit cannot sanitise is still drawn (the bonds are visible) and marked ``unsanitised``;
the legend also reports the minimum interatomic distance, the number of clashing pairs
(< 0.9 Å), the radius of gyration and the formula.

For the language model the string is drawn directly, and each sample is marked
``novel`` or ``seen in train`` by canonical-SMILES comparison with the training split
(cached in ``figures/results/.train_canon.json``).

Needs: torch, RDKit, Pillow. Runs on CPU.
"""

from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

METHODS = ("llm", "edm", "geobfn")
LABEL = {"llm": "LLM", "edm": "EDM", "geobfn": "GeoBFN"}
COLOUR = {"llm": (0xB0, 0x60, 0x40), "edm": (0x38, 0x54, 0x94), "geobfn": (0x2F, 0x6F, 0x4F)}
INK = (30, 30, 30)
MUTED = (110, 110, 110)
DEFAULT_INDICES = [4, 10, 19, 44, 67, 77, 127, 157, 208, 210]
CELL_W, CELL_H = 470, 470
HEADER, PAD = 96, 18


# --------------------------------------------------------------------------- #
# data access
# --------------------------------------------------------------------------- #
def run_dir(log_root: str, method: str) -> Optional[str]:
    found = sorted(glob.glob(os.path.join(log_root, method, "*", "config.json")),
                   key=os.path.getmtime)
    return os.path.dirname(found[-1]) if found else None


def checkpoint(run: str) -> str:
    for name in ("last.pt", "best.pt"):
        path = os.path.join(run, name)
        if os.path.exists(path):
            return path
    raise SystemExit(f"no checkpoint in {run}")


def indices_from_report() -> List[int]:
    path = os.path.join(ROOT, "utils", "report_numbers.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("llm", {}).get("samples") or []
        if len(rows) >= 10:
            return [int(r["index"]) for r in rows[:10]]
    return list(DEFAULT_INDICES)


def sample_files(run: str, method: str, resample: bool, num: int) -> Tuple[str, str]:
    """Return (payload_path, metrics_json_path), sampling first if needed."""
    ext = "smi" if method == "llm" else "xyz"
    payload = os.path.join(run, "test_eval", f"samples.{ext}")
    metrics = os.path.join(run, "test_eval", "metrics.json")
    if resample or not os.path.exists(payload):
        print(f"[showcase] sampling {method}: this runs evaluate.py once")
        cmd = [sys.executable, os.path.join(ROOT, "evaluate.py"),
               f"method={method}", f"eval.ckpt={checkpoint(run)}", f"eval.num={num}",
               "eval.use_ema=false", "eval.seed=0", f"eval.out={payload}",
               f"eval.json_out={metrics}"]
        subprocess.run(cmd, check=True, cwd=ROOT)
    return payload, metrics


def read_smiles(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def read_xyz(path: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    frames = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        while True:
            header = f.readline()
            if not header:
                break
            if not header.strip():
                continue
            n = int(header.strip())
            f.readline()                                    # comment line
            z, pos = [], []
            for _ in range(n):
                parts = f.readline().split()
                z.append(int(parts[0]))
                pos.append([float(v) for v in parts[1:4]])
            frames.append((np.asarray(z), np.asarray(pos)))
    return frames


def train_canonical_set(root: str, cache_dir: str) -> set:
    """Canonical SMILES of the training split, cached in the output directory."""
    cache = os.path.join(cache_dir, ".train_canon.json")
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return set(json.load(f))
    from rdkit import Chem

    path = os.path.join(root, "raw", "train.smi")
    out = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                smi = line.split()[0] if line.strip() else ""
                if not smi:
                    continue
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    out.add(Chem.MolToSmiles(mol))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        json.dump(sorted(out), f)
    print(f"[showcase] cached {len(out)} canonical training SMILES")
    return out


# --------------------------------------------------------------------------- #
# per-sample description
# --------------------------------------------------------------------------- #
def formula(z: Sequence[int]) -> str:
    symbols = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
    counts: Dict[int, int] = {}
    for zi in z:
        counts[int(zi)] = counts.get(int(zi), 0) + 1
    order = [6, 1] + sorted(k for k in counts if k not in (6, 1))
    return "".join(f"{symbols.get(k, k)}{counts[k] if counts[k] > 1 else ''}"
                   for k in order if k in counts)


def geometry(pos: np.ndarray) -> Dict[str, float]:
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nearest = d.min(axis=1)
    return {"min_d": float(nearest.min()),
            "clashes": int((nearest < 0.9).sum()),
            "rg": float(np.sqrt(((pos - pos.mean(axis=0)) ** 2).sum(axis=1).mean()))}


def draw_geometric(method: str, frames: List[Tuple[np.ndarray, np.ndarray]],
                   picks: Sequence[int]) -> Tuple[List[dict], list]:
    """Rebuild the molecule of every picked frame and draw it."""
    import torch
    from rdkit import Chem
    from rdkit.Chem import Draw

    metrics = importlib.import_module(f"molgen.{method}.metrics")
    rows, images = [], []
    for idx in picks:
        z, pos = frames[idx]
        geo = geometry(pos)
        mol, bonds, remaining = metrics.build_mol(torch.from_numpy(pos).float(),
                                                  torch.from_numpy(z).long())
        raw = mol.GetMol()
        try:
            Chem.SanitizeMol(raw)
            status = "sanitised"
        except Exception:                                  # noqa: BLE001
            status = "unsanitised"
            raw.UpdatePropertyCache(strict=False)
            Chem.FastFindRings(raw)
        saturated = bool(all(r == 0 for r in remaining))
        try:
            smiles = Chem.MolToSmiles(raw) if status == "sanitised" else ""
        except Exception:                                  # noqa: BLE001
            smiles = ""
        try:
            image = Draw.MolToImage(raw, size=(420, 330))
        except Exception:                                  # noqa: BLE001
            image = Image.new("RGB", (420, 330), "white")
        images.append(image.convert("RGB"))
        rows.append({"index": int(idx), "formula": formula(z), "atoms": int(len(z)),
                     "smiles": smiles, "status": status, "saturated": saturated,
                     **{k: round(v, 3) for k, v in geo.items()}})
    return rows, images


def draw_strings(smiles: List[str], picks: Sequence[int], train: set) -> Tuple[List[dict], list]:
    from rdkit import Chem
    from rdkit.Chem import Draw

    rows, images = [], []
    for idx in picks:
        smi = smiles[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            images.append(Image.new("RGB", (420, 330), "white"))
            rows.append({"index": int(idx), "smiles": smi, "formula": "", "atoms": 0,
                         "status": "unparsable", "novel": None})
            continue
        canon = Chem.MolToSmiles(mol)
        novel = canon not in train
        images.append(Draw.MolToImage(mol, size=(420, 330)).convert("RGB"))
        rows.append({"index": int(idx), "smiles": smi, "canonical": canon,
                     "formula": formula([a.GetAtomicNum() for a in mol.GetAtoms()]),
                     "atoms": int(mol.GetNumAtoms()), "status": "sanitised",
                     "novel": bool(novel)})
    return rows, images


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def font(size: int):
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:                                  # noqa: BLE001
            continue
    return ImageFont.load_default()


def render(method: str, rows: List[dict], images: list, run: str, out_png: str) -> None:
    cols, n = 5, len(rows)
    rows_n = max(1, (n + cols - 1) // cols)
    width = cols * CELL_W + PAD * 2
    height = HEADER + rows_n * CELL_H + PAD
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.rectangle([0, 0, width, HEADER - 14], fill=(246, 244, 240))
    draw.rectangle([0, HEADER - 14, width, HEADER - 10], fill=COLOUR[method])
    draw.text((PAD + 6, 18), f"{LABEL[method]} \u00b7 ten samples",
              fill=COLOUR[method], font=font(30))
    draw.text((PAD + 6, 56),
              f"run {os.path.basename(run)} \u00b7 seed 0 \u00b7 the same draw indices as "
              f"Table 20/21 \u00b7 molecules rebuilt from the scored graph",
              fill=MUTED, font=font(17))

    for i, (row, image) in enumerate(zip(rows, images)):
        cx = PAD + (i % cols) * CELL_W
        cy = HEADER + (i // cols) * CELL_H
        canvas.paste(image, (cx + 20, cy + 6))
        draw.rectangle([cx + 20, cy + 6, cx + 440, cy + 336], outline=(225, 222, 218))
        draw.text((cx + 22, cy + 348), f"#{row['index']}  {row.get('formula', '')}",
                  fill=INK, font=font(19))
        if method == "llm":
            line = ("novel" if row.get("novel") else "seen in train")
            detail = f"{line} \u00b7 {row['atoms']} atoms"
        else:
            detail = (f"{row['status']} \u00b7 d_min {row['min_d']:.3f} \u00c5 \u00b7 "
                      f"{row['clashes']} clashes \u00b7 Rg {row['rg']:.2f} \u00c5")
        draw.text((cx + 22, cy + 374), detail, fill=MUTED, font=font(15))
        if method != "llm" and not row["saturated"]:
            draw.text((cx + 22, cy + 396), "unsatisfied valence", fill=(168, 75, 28),
                      font=font(15))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    canvas.save(out_png)
    print(f"wrote {out_png}  ({canvas.size[0]}x{canvas.size[1]})")


def summary_rows(summary: Dict[str, List[dict]]) -> str:
    """LaTeX body of the summary table (rows only, header included)."""
    head = ("Method & drawn & sanitised & saturated & mean $d_{\\min}$ (\\AA) & "
            "clashes & mean $R_g$ (\\AA) \\\\\n\\midrule\n")
    body = []
    for method in METHODS:
        rows = summary.get(method)
        if not rows:
            continue
        ok = sum(1 for r in rows if r["status"] == "sanitised")
        sat = sum(1 for r in rows if r.get("saturated"))
        sat_text = f"{sat}/{len(rows)}" if method != "llm" else "---"
        dmin = np.mean([r["min_d"] for r in rows]) if method != "llm" else float("nan")
        rg = np.mean([r["rg"] for r in rows]) if method != "llm" else float("nan")
        ncl = sum(r.get("clashes", 0) for r in rows) if method != "llm" else None
        dmin_text = f"{dmin:.3f}" if dmin == dmin else "---"
        rg_text = f"{rg:.2f}" if rg == rg else "---"
        clash_text = f"{ncl}" if ncl is not None else "---"
        body.append(f"{LABEL[method]} & {len(rows)} & {ok} & {sat_text} & {dmin_text} & "
                    f"{clash_text} & {rg_text} \\\\\n")
    return head + "".join(body) + "\\bottomrule\n"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=METHODS, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--log-root", default="log")
    p.add_argument("--data-root", default=os.environ.get("DATA_DIR", "data/qm9"))
    p.add_argument("--out", default="showcase",
                   help="output directory (the report reads figures/results)")
    p.add_argument("--num", type=int, default=1024, help="samples drawn if re-sampling")
    p.add_argument("--resample", action="store_true")
    args = p.parse_args()

    methods = METHODS if args.all or args.method is None else (args.method,)
    picks = indices_from_report()
    print(f"[showcase] indices {picks}")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    summary, payload = {}, {}
    for method in methods:
        run = run_dir(args.log_root, method)
        if not run:
            print(f"[showcase] no run for {method}: skipped")
            continue
        path, metrics_path = sample_files(run, method, args.resample, args.num)
        if method == "llm":
            smiles = read_smiles(path)
            train = train_canonical_set(args.data_root, out_dir)
            rows, images = draw_strings(smiles, [i for i in picks if i < len(smiles)], train)
        else:
            frames = read_xyz(path)
            rows, images = draw_geometric(method, frames, [i for i in picks if i < len(frames)])
        render(method, rows, images, run, os.path.join(out_dir, f"showcase_{method}.png"))
        with open(os.path.join(out_dir, f"showcase_{method}.json"), "w", encoding="utf-8") as f:
            json.dump({"method": method, "run": os.path.basename(run),
                       "metrics": json.load(open(metrics_path, encoding="utf-8"))
                       if os.path.exists(metrics_path) else {},
                       "samples": rows}, f, indent=2)
        summary[method] = rows
        payload[method] = rows

    if summary:
        path = os.path.join(out_dir, "showcase_table.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(summary_rows(summary))
        print(f"wrote {path}")
        with open(os.path.join(out_dir, "showcase_all.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()

