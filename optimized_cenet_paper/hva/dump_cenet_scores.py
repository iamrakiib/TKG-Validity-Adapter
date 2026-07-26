from __future__ import annotations

"""Export object-candidate score dumps from a trained CENET checkpoint.

This script is the bridge between the CENET backbone and the paper variants:
Exact Recency, RHVC, HVA dual-branch, and HVA exact-only.

Reviewer-safety rules:
  * The checkpoint produces scores first.
  * HVA/RHVC later select top-K from these scores only.
  * The gold answer is never inserted into the candidate set.
  * This script only exports backbone scores and query triples.
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Allow importing CENET modules from repository root when executed as python -m hva.dump_cenet_scores
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402


def _load_pickle(path: Path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def _load_frequency(path: Path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    # CENET history preprocessing stores scipy sparse matrices.
    return obj.toarray() if hasattr(obj, 'toarray') else np.asarray(obj)


def _split_files(split: str):
    if split == 'valid':
        return 'valid.txt', 'dev'
    if split == 'test':
        return 'test.txt', 'test'
    raise ValueError(split)


def _device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _to_device(x, device):
    return x.to(device)


def _compute_object_scores(model, batch_data, score_kind: str):
    """Return CENET object scores for queries (s,r,?,t).

    CENET internally computes s_preds from (s,r) to candidate object entities.
    When score_kind='oracle', the predicted oracle history mask is applied, matching
    the original CENET evaluation branch used as the main reported output.
    When score_kind='no_oracle', raw CENET object scores are exported.
    """
    quadruples, s_history_event_o, _o_history_event_s, \
        s_history_label_true, _o_history_label_true, s_frequency, _o_frequency = batch_data

    if utils.isListEmpty(s_history_event_o):
        return None

    s = quadruples[:, 0]
    r = quadruples[:, 1]
    o = quadruples[:, 2]

    # History/non-history tags follow the original CENET forward pass.
    s_history_tag = s_frequency.clone()
    s_non_history_tag = s_frequency.clone()
    s_history_tag[s_history_tag != 0] = model.args.lambdax
    s_non_history_tag[s_history_tag == 1] = -model.args.lambdax
    s_non_history_tag[s_history_tag == 0] = model.args.lambdax
    s_history_tag[s_history_tag == 0] = -model.args.lambdax

    s_frequency_sm = F.softmax(s_frequency, dim=1)
    s_frequency_hidden = model.tanh(model.linear_frequency(s_frequency_sm))

    _s_nce_loss, s_preds = model.calculate_nce_loss(
        s, o, r, model.rel_embeds[:model.num_rel],
        model.linear_pred_layer_s1, model.linear_pred_layer_s2,
        s_history_tag, s_non_history_tag,
    )

    if score_kind == 'no_oracle':
        return s_preds

    # Predicted-oracle branch, matching CENET's original Oracle evaluation.
    _s_ce_loss, s_pred_history_label, _acc = model.oracle_loss(
        s, r, model.rel_embeds[:model.num_rel], s_history_label_true, s_frequency_hidden
    )
    tmp_label = torch.squeeze(s_pred_history_label).clone().detach()
    tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
    tmp_label[torch.where(tmp_label < 0.5)[0]] = 0

    s_history_oid = []
    for i in range(quadruples.shape[0]):
        s_history_oid.append([])
        for con_events in s_history_event_o[i]:
            # con_events columns are [relation, object].
            if len(con_events) > 0:
                s_history_oid[-1] += con_events[:, 1].tolist()

    s_mask = torch.zeros(quadruples.shape[0], model.num_e, device=s_preds.device)
    for i in range(quadruples.shape[0]):
        if tmp_label[i] > 0:
            if len(s_history_oid[i]) > 0:
                s_mask[i, s_history_oid[i]] = 1
        else:
            s_mask[i, :] = 1
            if len(s_history_oid[i]) > 0:
                s_mask[i, s_history_oid[i]] = 0
    return s_preds * s_mask


def main():
    p = argparse.ArgumentParser(description='Dump CENET backbone object score matrices for HVA/RHVC variants.')
    p.add_argument('--dataset', required=True, choices=['ICEWS14','ICEWS18'])
    p.add_argument('--split', required=True, choices=['valid','test'])
    p.add_argument('--data-root', default='data')
    p.add_argument('--model-dir', default=None, help='CENET experiment folder containing models/<dataset>_best.pth')
    p.add_argument('--checkpoint', default=None, help='Direct path to <dataset>_best.pth')
    p.add_argument('--out', required=True)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--score-kind', choices=['oracle','no_oracle'], default='oracle')
    p.add_argument('--allow-missing-valid', action='store_true', help='For repository smoke checks only. Do not use for final paper runs.')
    args = p.parse_args()

    device = _device()
    data_dir = Path(args.data_root) / args.dataset
    split_file, prefix = _split_files(args.split)
    if not (data_dir / split_file).exists():
        raise FileNotFoundError(f'{data_dir/split_file} not found. For paper runs, provide the official {args.dataset} {args.split} split.')

    data, _ = utils.load_quadruples(str(data_dir), split_file)
    required = [
        data_dir / f'{prefix}_history_sub.txt',
        data_dir / f'{prefix}_history_ob.txt',
        data_dir / f'{prefix}_s_label.txt',
        data_dir / f'{prefix}_o_label.txt',
        data_dir / f'{prefix}_s_frequency.txt',
        data_dir / f'{prefix}_o_frequency.txt',
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise FileNotFoundError('Missing CENET history files. Run scripts/00_prepare_cenet_history_<dataset>.sh first. Missing: ' + ', '.join(missing))

    s_history_data = _load_pickle(data_dir / f'{prefix}_history_sub.txt')
    o_history_data = _load_pickle(data_dir / f'{prefix}_history_ob.txt')
    s_history = s_history_data[0]
    o_history = o_history_data[0]
    s_label = _load_pickle(data_dir / f'{prefix}_s_label.txt')
    o_label = _load_pickle(data_dir / f'{prefix}_o_label.txt')
    s_frequency = _load_frequency(data_dir / f'{prefix}_s_frequency.txt')
    o_frequency = _load_frequency(data_dir / f'{prefix}_o_frequency.txt')

    if args.checkpoint is not None:
        ckpt = Path(args.checkpoint)
    elif args.model_dir is not None:
        ckpt = Path(args.model_dir) / 'models' / f'{args.dataset}_best.pth'
    else:
        raise ValueError('Provide either --checkpoint or --model-dir')
    if not ckpt.exists():
        raise FileNotFoundError(str(ckpt))

    try:
        model = torch.load(str(ckpt), map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(str(ckpt), map_location=device)
    model = model.to(device)
    model.eval()

    # Keep batch size synchronized with this dump run.
    model.args.batch_size = args.batch_size

    all_scores, all_triples = [], []
    with torch.no_grad():
        for batch_data in utils.make_batch(
            data, s_history, o_history, s_label, o_label, s_frequency, o_frequency, args.batch_size
        ):
            triples_np = batch_data[0].astype(np.int64)
            batch_data[0] = _to_device(torch.from_numpy(batch_data[0]), device)
            batch_data[3] = _to_device(torch.from_numpy(batch_data[3]).float(), device)
            batch_data[4] = _to_device(torch.from_numpy(batch_data[4]).float(), device)
            batch_data[5] = _to_device(torch.from_numpy(batch_data[5]).float(), device)
            batch_data[6] = _to_device(torch.from_numpy(batch_data[6]).float(), device)
            scores = _compute_object_scores(model, batch_data, args.score_kind)
            if scores is None:
                continue
            all_scores.append(scores.detach().cpu().numpy().astype(np.float32))
            all_triples.append(triples_np)

    if not all_scores:
        raise RuntimeError('No CENET scores were produced. Check history preprocessing and checkpoint compatibility.')
    scores = np.concatenate(all_scores, axis=0)
    triples = np.concatenate(all_triples, axis=0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, scores=scores, triples=triples, score_kind=np.asarray([args.score_kind]))
    print({'dataset': args.dataset, 'split': args.split, 'score_kind': args.score_kind, 'scores_shape': scores.shape, 'triples_shape': triples.shape, 'out': str(out)})


if __name__ == '__main__':
    main()
