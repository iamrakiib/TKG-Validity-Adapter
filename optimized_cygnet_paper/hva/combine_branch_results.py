from __future__ import annotations
import argparse, json, os
from typing import Dict


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def avg_metric(a: Dict, b: Dict, key: str):
    return 0.5 * (float(a.get(key, 0.0)) + float(b.get(key, 0.0)))


def combine(obj: Dict, sub: Dict):
    out = {
        'dataset': obj.get('dataset', sub.get('dataset', '')),
        'method': obj.get('method', obj.get('mode', '')),
        'branch': 'combined_object_subject_average',
        'object_result': obj,
        'subject_result': sub,
    }
    # Support both HVA/RHVC scripts and exact-recency scripts.
    for split_key in ['test_overall', 'valid_overall']:
        if split_key in obj and split_key in sub:
            out[split_key] = {
                'count': int(obj[split_key].get('count', 0)) + int(sub[split_key].get('count', 0)),
                'MRR': avg_metric(obj[split_key], sub[split_key], 'MRR'),
                'Hits@1': avg_metric(obj[split_key], sub[split_key], 'Hits@1'),
                'Hits@3': avg_metric(obj[split_key], sub[split_key], 'Hits@3'),
                'Hits@10': avg_metric(obj[split_key], sub[split_key], 'Hits@10'),
            }
    for diag_key in ['test_diagnostics', 'valid_diagnostics']:
        if diag_key in obj and diag_key in sub:
            out[diag_key] = {}
            keys = set(obj[diag_key].keys()) | set(sub[diag_key].keys())
            for k in keys:
                ov, sv = obj[diag_key].get(k), sub[diag_key].get(k)
                if isinstance(ov, dict) and isinstance(sv, dict):
                    out[diag_key][k] = {
                        kk: (avg_metric(ov, sv, kk) if isinstance(ov.get(kk, None), (int, float)) and isinstance(sv.get(kk, None), (int, float)) else ov.get(kk, sv.get(kk)))
                        for kk in set(ov.keys()) | set(sv.keys())
                    }
    return out


def main():
    p = argparse.ArgumentParser(description='Average CyGNet object and subject branch results for paper tables')
    p.add_argument('--object-result', required=True)
    p.add_argument('--subject-result', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    result = combine(load_json(args.object_result), load_json(args.subject_result))
    save_json(result, args.out)
    print(json.dumps({k: result[k] for k in result.keys() if k in {'test_overall','valid_overall'}}, indent=2))
    print('Saved combined result to:', args.out)


if __name__ == '__main__':
    main()
