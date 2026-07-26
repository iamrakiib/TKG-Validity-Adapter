from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser(description="Merge TeMP score chunks into one HVA score dump")
    p.add_argument("--chunk-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pattern", default="temp_scores_*.npz")
    args = p.parse_args()

    chunk_dir = Path(args.chunk_dir)
    files = sorted(chunk_dir.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"No score chunk files found in {chunk_dir} with pattern {args.pattern}")

    scores, triples = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        scores.append(z["scores"].astype(np.float32))
        triples.append(z["triples"].astype(np.int64))
    scores = np.concatenate(scores, axis=0)
    triples = np.concatenate(triples, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, scores=scores, triples=triples, entity=np.asarray(["object"]))
    print({"chunks": len(files), "scores_shape": scores.shape, "triples_shape": triples.shape, "out": str(out)})


if __name__ == "__main__":
    main()
