from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import numpy as np

Quad = Tuple[int, int, int, int]

@dataclass
class DatasetSplits:
    dataset: str
    num_entities: int
    num_relations: int
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def _read_stat(dataset_dir: Path) -> Tuple[int, int]:
    stat_path = dataset_dir / "stat.txt"
    if not stat_path.exists():
        raise FileNotFoundError(f"Missing stat.txt in {dataset_dir}")
    parts = stat_path.read_text().strip().replace("\t", " ").split()
    if len(parts) < 2:
        raise ValueError(f"stat.txt must contain at least num_entities and num_relations: {stat_path}")
    return int(parts[0]), int(parts[1])


def read_quad_file(path: Path) -> np.ndarray:
    """Read a TKG split file and return columns [s, r, o, t].

    The uploaded ICEWS files contain an additional final column in some folders;
    the first four columns are the temporal fact used by this paper.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    rows: List[List[int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.replace("\t", " ").split()
            if len(parts) < 4:
                raise ValueError(f"Bad line {line_no} in {path}: {line!r}")
            rows.append([int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])])
    if not rows:
        return np.zeros((0, 4), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def load_dataset(dataset: str, data_root: str | Path = "data") -> DatasetSplits:
    dataset = dataset.upper()
    dataset_dir = Path(data_root) / dataset
    n_ent, n_rel = _read_stat(dataset_dir)
    train = read_quad_file(dataset_dir / "train.txt")
    valid = read_quad_file(dataset_dir / "valid.txt")
    test = read_quad_file(dataset_dir / "test.txt")
    return DatasetSplits(dataset=dataset, num_entities=n_ent, num_relations=n_rel, train=train, valid=valid, test=test)


def group_by_time(quads: np.ndarray) -> List[Tuple[int, np.ndarray]]:
    """Return [(timestamp, facts_at_timestamp)] sorted by timestamp."""
    if len(quads) == 0:
        return []
    order = np.argsort(quads[:, 3], kind="stable")
    quads = quads[order]
    out: List[Tuple[int, np.ndarray]] = []
    start = 0
    while start < len(quads):
        t = int(quads[start, 3])
        end = start + 1
        while end < len(quads) and int(quads[end, 3]) == t:
            end += 1
        out.append((t, quads[start:end]))
        start = end
    return out


def build_time_filter_map(all_quads: np.ndarray) -> Dict[Tuple[int, int, int], set]:
    """Map (s, r, t) -> all true objects at the same timestamp.

    During filtered evaluation, other correct objects for the same query and time
    are removed from the candidate list, except the target object itself.
    """
    filt: Dict[Tuple[int, int, int], set] = {}
    for s, r, o, t in all_quads.tolist():
        filt.setdefault((int(s), int(r), int(t)), set()).add(int(o))
    return filt
