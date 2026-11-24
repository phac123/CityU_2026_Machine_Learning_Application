"""Download QM9 and convert it into the npz/smi artefacts the three methods read.

    python tools/fetch_qm9.py --out /workspace/data/qm9
    python tools/fetch_qm9.py --out /workspace/data/qm9 --split scaffold
    python tools/fetch_qm9.py --out /workspace/data/qm9 \
        --from-local /workspace/data/qm9/dsgdb9nsd.xyz.tar.bz2   # no internet needed
    python tools/fetch_qm9.py --out /tmp/qm9 --limit 2000        # quick subset

Why this exists.  ``molgen/preprocess/build_dataset.py`` reads ``gdb9.sdf``; that file
is a 130 MB re-packaging hosted by DeepChem.  The *canonical* QM9 release is the
``dsgdb9nsd.xyz.tar.bz2`` archive (~80 MB on figshare) whose 133 885 files already
carry the atomic numbers, the B3LYP coordinates **and** the SMILES on their last line,
so the whole conversion is possible without RDKit.  This tool downloads that archive,
streams it once, and writes exactly the layout ``build_dataset.py`` writes---same
``raw/{train,valid,test}.{npz,smi}`` and the same ``dataset_stats.json``---using the
same helper functions, so the two routes cannot drift apart.

RDKit is used only to canonicalise SMILES and to compute Bemis-Murcko scaffolds (the
``scaffold`` split).  Without it the script still converts, writing the raw SMILES from
the archive and falling back to the random split (with a warning).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import time
import urllib.request
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from molgen.preprocess.build_dataset import (  # noqa: E402
    random_split,
    scaffold_split,
    write_split,
)

# The canonical release (figshare item 3195389) and DeepChem's mirror of it.
MIRRORS = (
    ("https://figshare.com/ndownloader/files/3195389", "dsgdb9nsd.xyz.tar.bz2"),
    ("https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz",
     "gdb9.tar.gz"),
)
EXPECTED_MOLECULES = 133_885
MIN_BYTES = 40 * 1024 * 1024           # anything smaller is an error page, not QM9

SYMBOL_TO_Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9}
SMILES_TOKEN = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$/\\%.:]+$")


# --------------------------------------------------------------------------- #
# download
# --------------------------------------------------------------------------- #
def download(dest_dir: str, url: Optional[str] = None, filename: Optional[str] = None,
             timeout: int = 60, retries: int = 3) -> str:
    """Download the archive to ``dest_dir`` (skipped when it is already there)."""
    os.makedirs(dest_dir, exist_ok=True)
    candidates = [(url, filename)] if url else list(MIRRORS)
    last_error: Optional[Exception] = None

    for candidate_url, candidate_name in candidates:
        name = candidate_name or os.path.basename(candidate_url)
        path = os.path.join(dest_dir, name)
        if os.path.exists(path) and os.path.getsize(path) >= MIN_BYTES:
            print(f"[fetch] reusing {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
            return path
        for attempt in range(1, retries + 1):
            try:
                print(f"[fetch] {candidate_url} (attempt {attempt}/{retries})")
                request = urllib.request.Request(
                    candidate_url, headers={"User-Agent": "molgen-fetch/1.0"})
                with urllib.request.urlopen(request, timeout=timeout) as response, \
                        open(path + ".part", "wb") as out:
                    total = int(response.headers.get("Content-Length") or 0)
                    done = 0
                    while True:
                        chunk = response.read(1 << 20)
                        if not chunk:
                            break
                        out.write(chunk)
                        done += len(chunk)
                        if total:
                            pct = 100.0 * done / total
                            print(f"\r[fetch]   {done / 1e6:6.1f} / {total / 1e6:.1f} MB "
                                  f"({pct:4.1f}%)", end="", flush=True)
                    print()
                if os.path.getsize(path + ".part") < MIN_BYTES:
                    raise IOError(f"only {os.path.getsize(path + '.part')} bytes: "
                                  "not a QM9 archive")
                os.replace(path + ".part", path)
                print(f"[fetch] saved {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
                return path
            except Exception as exc:  # noqa: BLE001 - try the next mirror / attempt
                last_error = exc
                print(f"[fetch] failed: {type(exc).__name__}: {exc}")
                if os.path.exists(path + ".part"):
                    os.remove(path + ".part")
                time.sleep(2 * attempt)

    raise SystemExit(
        "could not download QM9:\n"
        f"  {last_error}\n"
        "Download the archive by hand on a machine with internet access and pass it:\n"
        "  python tools/fetch_qm9.py --out <dir> --from-local <dsgdb9nsd.xyz.tar.bz2>"
    )


# --------------------------------------------------------------------------- #
# parsing the canonical .xyz format
# --------------------------------------------------------------------------- #
def to_float(text: str) -> float:
    """Parse one QM9 number.

    A handful of entries in the published archive use Mathematica's exponent
    notation (``2.1997*^-6``) instead of ``2.1997e-06``; Python's float() rejects it,
    so the ``*^`` is rewritten first.
    """
    return float(text.replace("*^", "e").replace("* ^", "e"))


def canonical_smiles(smiles: str) -> str:
    """Canonical form when RDKit is available, the raw string otherwise."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol) if mol is not None else smiles
    except Exception:
        return smiles


def parse_qm9_xyz(text: str) -> Optional[dict]:
    """Parse one ``dsgdb9nsd_XXXXXX.xyz`` file into a record for ``write_split``.

    Layout: ``n`` / comment line ``gdb <index> <17 properties> <symbols>`` /
    ``n`` atom lines ``symbol x y z <mulliken>`` / one line of 50+ harmonic
    frequencies / one line with the two SMILES (GDB-9 and relaxed) / a line of
    three integers.  Coordinates are already in Angstrom.
    """
    lines = text.splitlines()
    if len(lines) < 3:
        return None
    try:
        n_atoms = int(lines[0].strip())
    except ValueError:
        return None
    if n_atoms <= 0 or len(lines) < 2 + n_atoms:
        return None

    comment = lines[1].split()
    index = None
    if len(comment) >= 2 and comment[0] == "gdb":
        try:
            index = int(comment[1])
        except ValueError:
            index = None

    z: List[int] = []
    pos: List[List[float]] = []
    for line in lines[2:2 + n_atoms]:
        parts = line.split()
        if len(parts) < 4:
            return None
        element = parts[0]
        if element not in SYMBOL_TO_Z:
            return None                      # outside the H/C/N/O/F subset: skip
        z.append(SYMBOL_TO_Z[element])
        pos.append([to_float(parts[1]), to_float(parts[2]), to_float(parts[3])])

    # the SMILES pair sits after the frequency line: two SMILES-like tokens
    smiles = ""
    for line in lines[2 + n_atoms:]:
        parts = line.split()
        if len(parts) == 2 and all(SMILES_TOKEN.match(p) for p in parts):
            smiles = parts[1]                # the relaxed-geometry SMILES
            break
        if len(parts) == 1 and SMILES_TOKEN.match(parts[0]) and not parts[0].isdigit():
            smiles = parts[0]
            break

    return {
        "index": index,
        "z": np.asarray(z, dtype=np.int64),
        "pos": np.asarray(pos, dtype=np.float64),
        "smiles": canonical_smiles(smiles) if smiles else "",
    }


def iter_records(archive: str, limit: Optional[int] = None,
                 verbose_every: int = 20000) -> List[dict]:
    """Stream every molecule out of the archive (or a directory of .xyz files)."""
    records: List[dict] = []
    skipped = 0
    started = time.time()

    if os.path.isdir(archive):
        names = sorted(f for f in os.listdir(archive) if f.endswith(".xyz"))
        print(f"[read] {len(names)} .xyz files in {archive}")
        for name in names:
            with open(os.path.join(archive, name), encoding="utf-8",
                      errors="ignore") as f:
                rec = parse_qm9_xyz(f.read())
            if rec is None:
                skipped += 1
            else:
                records.append(rec)
            if limit and len(records) >= limit:
                break
    else:
        mode = "r:bz2" if archive.endswith(".bz2") else "r:gz"
        with tarfile.open(archive, mode) as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".xyz")]
            print(f"[read] {len(members)} members in {os.path.basename(archive)}")
            for i, member in enumerate(members, 1):
                handle = tar.extractfile(member)
                if handle is None:
                    skipped += 1
                    continue
                rec = parse_qm9_xyz(handle.read().decode("utf-8", errors="ignore"))
                if rec is None:
                    skipped += 1
                else:
                    records.append(rec)
                if i % verbose_every == 0:
                    print(f"[read]   {i}/{len(members)} ({time.time() - started:.0f}s)")
                if limit and len(records) >= limit:
                    break

    print(f"[read] kept {len(records)} molecules, skipped {skipped} "
          f"({time.time() - started:.0f}s)")
    return records


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #
def convert(records: Sequence[dict], out_root: str, split: str,
            ratios: Sequence[float], seed: int, source: str) -> Dict:
    """Split the records and write the artefacts with build_dataset's helpers."""
    if split == "scaffold":
        try:
            idx = scaffold_split(records, ratios)
        except Exception as exc:            # RDKit missing / no scaffold
            print(f"[split] scaffold split unavailable ({type(exc).__name__}: {exc}); "
                  "falling back to a random split")
            idx = random_split(len(records), ratios, seed=seed)
            split = "random"
    else:
        idx = random_split(len(records), ratios, seed=seed)

    stats = []
    for name in ("train", "valid", "test"):
        subset = [records[i] for i in idx[name]]
        stats.append(write_split(out_root, name, subset))
        print(f"[write] {name}: {len(subset)} molecules -> {out_root}/raw/{name}.npz")

    summary = {
        "source": source,
        "format": "qm9-xyz",
        "split": split,
        "ratios": list(ratios),
        "seed": seed,
        "n_molecules": len(records),
        "smiles_without_rdkit_canonicalisation": sum(1 for r in records if not r["smiles"]),
        "splits": stats,
    }
    with open(os.path.join(out_root, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] stats -> {out_root}/dataset_stats.json")
    return summary


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True, help="output root, e.g. /workspace/data/qm9")
    p.add_argument("--from-local", default=None,
                   help="existing .tar.bz2/.tar.gz or a directory of .xyz files "
                        "(skips the download)")
    p.add_argument("--url", default=None, help="override the download URL")
    p.add_argument("--split", default="random", choices=["random", "scaffold"])
    p.add_argument("--ratios", default="0.8,0.1,0.1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None,
                   help="keep only the first N molecules (for a quick dry run)")
    p.add_argument("--keep-archive", action="store_true",
                   help="keep the downloaded archive next to the artefacts")
    args = p.parse_args()

    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or abs(sum(ratios) - 1.0) > 1e-6:
        raise SystemExit("--ratios must be three fractions summing to 1")

    # the files may already be there: convert them instead of downloading again
    if args.from_local is None:
        present = [f for f in os.listdir(args.out) if f.endswith(".xyz")] \
            if os.path.isdir(args.out) else []
        if len(present) > 1000:
            print(f"[plan] found {len(present)} .xyz files in {args.out}; "
                  "converting them instead of downloading")
            args.from_local = args.out

    if args.from_local:
        archive = args.from_local
        source = f"local:{os.path.basename(archive)}"
        if not os.path.exists(archive):
            raise SystemExit(f"--from-local {archive} does not exist")
    else:
        archive = download(args.out, url=args.url)
        source = args.url or "https://figshare.com/ndownloader/files/3195389"

    records = iter_records(archive, limit=args.limit)
    if not records:
        raise SystemExit("no molecule could be parsed: is this the QM9 archive?")
    if args.limit is None and len(records) < 0.9 * EXPECTED_MOLECULES:
        print(f"[warn] only {len(records)} of {EXPECTED_MOLECULES} molecules were read; "
              "the archive may be truncated")

    convert(records, args.out, args.split, ratios, args.seed, source)

    if not args.keep_archive and archive.startswith(args.out) and not args.from_local:
        os.remove(archive)
        print(f"[clean] removed {archive} (pass --keep-archive to keep it)")

    print("\nNext:\n"
          f"  python -m molgen.preprocess.explore --data {args.out}/raw/train.smi "
          f"--out logs/analysis/explore --tsne\n"
          f"  bash run_experiments.sh")


if __name__ == "__main__":
    main()

