import os
import torch
from utils.CorrptTriples import CorruptTriples
from utils.utils import cuda
import numpy as np


class EvaluationFilter:
    def __init__(self, args, calc_score, graph_dict_train, graph_dict_val, graph_dict_test):
        self.args = args
        self.calc_score = calc_score
        self.graph_dict_train = graph_dict_train
        self.graph_dict_val = graph_dict_val
        self.graph_dict_test = graph_dict_test
        self.graph_dict_total = {**self.graph_dict_train, **self.graph_dict_val, **self.graph_dict_test}
        self._score_dump_counter = 0
        self._dump_score_dir = getattr(args, "dump_score_dir", None)
        if self._dump_score_dir:
            os.makedirs(self._dump_score_dir, exist_ok=True)
        self.get_true_head_and_tail_all()

    def get_true_head_and_tail_all(self):
        self.true_heads = dict()
        self.true_tails = dict()
        times = list(self.graph_dict_total.keys())
        for t in times:
            if self.args.dataset_dir == 'extrapolation':
                g = self.graph_dict_total[t]
                triples = torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1)
            else:
                triples = []
                for g in self.graph_dict_train[t], self.graph_dict_val[t], self.graph_dict_test[t]:
                    triples.append(torch.stack([g.edges()[0], g.edata['type_s'], g.edges()[1]]).transpose(0, 1))
                triples = torch.cat(triples, dim=0)
            true_head, true_tail = CorruptTriples.get_true_head_and_tail_per_graph(triples)
            self.true_heads[t] = true_head
            self.true_tails[t] = true_tail

    def calc_metrics_single_graph(self, ent_mean, rel_enc_means, all_ent_embeds, samples, graph, time, eval_bz=100):
        with torch.no_grad():
            s = samples[:, 0]
            r = samples[:, 1]
            o = samples[:, 2]
            test_size = samples.shape[0]
            num_ent = all_ent_embeds.shape[0]
            o_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="tail")
            s_mask = self.mask_eval_set(samples, test_size, num_ent, time, graph, mode="head")
            # Object prediction, matching the HVA paper setting.
            ranks_o = self.perturb_and_get_rank(ent_mean, rel_enc_means, all_ent_embeds, s, r, o,
                                                 test_size, o_mask, graph, eval_bz, mode='tail', time=time)
            # Subject prediction remains available for original TeMP reporting.
            ranks_s = self.perturb_and_get_rank(ent_mean, rel_enc_means, all_ent_embeds, s, r, o,
                                                 test_size, s_mask, graph, eval_bz, mode='head', time=time)
            ranks = torch.cat([ranks_s, ranks_o])
            ranks += 1  # change to 1-indexed
        return ranks

    def perturb_and_get_rank(self, ent_mean, rel_enc_means, all_ent_embeds, s, r, o, test_size,
                             mask, graph, batch_size=100, mode='tail', time=None):
        """Perturb one element in the triplets.

        HVA alignment: when --dump-score-dir is provided, this function also
        exports the unnormalized, filtered candidate score matrix for object
        prediction. The dump never inserts the gold candidate into top-k;
        HVA later selects top-k only from these TeMP scores.
        """
        n_batch = (test_size + batch_size - 1) // batch_size
        ranks = []
        for idx in range(n_batch):
            batch_start = idx * batch_size
            batch_end = min(test_size, (idx + 1) * batch_size)
            batch_r = rel_enc_means[r[batch_start: batch_end]]

            if mode == 'tail':
                batch_s = ent_mean[s[batch_start: batch_end]]
                batch_o = all_ent_embeds
                target = o[batch_start: batch_end]
            else:
                batch_s = all_ent_embeds
                batch_o = ent_mean[o[batch_start: batch_end]]
                target = s[batch_start: batch_end]
            target = torch.tensor([graph.ids[i.item()] for i in target])

            if self.args.use_cuda:
                target = cuda(target)

            unmasked_score = self.calc_score(batch_s, batch_r, batch_o, mode=mode)
            mask_bool = mask[batch_start:batch_end].bool()
            masked_score = torch.where(mask_bool, -10e6 * unmasked_score.new_ones(unmasked_score.shape), unmasked_score)

            self._maybe_dump_scores(masked_score, s[batch_start:batch_end], r[batch_start:batch_end],
                                    o[batch_start:batch_end], graph, time, mode)

            score = torch.sigmoid(masked_score)  # bsz, n_ent
            ranks.append(self.sort_and_rank(score, target))
        return torch.cat(ranks)

    def _maybe_dump_scores(self, score_tensor, s_local, r_tensor, o_local, graph, time, mode):
        if not self._dump_score_dir:
            return
        object_only = getattr(self.args, "dump_score_object_only", True)
        if object_only and mode != "tail":
            return
        if time is None:
            return
        t_val = int(time.item()) if torch.is_tensor(time) else int(time)
        score_np = score_tensor.detach().cpu().numpy().astype(np.float32)
        queries = []
        num_rels_for_hva = int(getattr(self.args, "num_rels_for_hva", 0) or 0)
        s_cpu = s_local.detach().cpu().tolist() if torch.is_tensor(s_local) else list(s_local)
        o_cpu = o_local.detach().cpu().tolist() if torch.is_tensor(o_local) else list(o_local)
        r_cpu = r_tensor.detach().cpu().tolist() if torch.is_tensor(r_tensor) else list(r_tensor)
        for s_i, r_i, o_i in zip(s_cpu, r_cpu, o_cpu):
            gs = int(graph.ids[int(s_i)])
            go = int(graph.ids[int(o_i)])
            rr = int(r_i)
            if mode == "tail":
                queries.append([gs, rr, go, t_val])
            else:
                # Subject prediction can be canonicalized as object prediction
                # with an inverse relation when --dump-both-directions is used.
                inv_r = rr + num_rels_for_hva if num_rels_for_hva > 0 else rr
                queries.append([go, inv_r, gs, t_val])
        out_path = os.path.join(self._dump_score_dir, f"temp_scores_{mode}_{t_val}_{self._score_dump_counter:08d}.npz")
        self._score_dump_counter += 1
        np.savez_compressed(out_path, scores=score_np, triples=np.asarray(queries, dtype=np.int64), entity=np.asarray(["object"]))

    def mask_eval_set(self, test_triplets, test_size, num_ent, time, graph, mode='tail'):
        time = int(time.item())
        mask = test_triplets.new_zeros(test_size, num_ent)
        for i in range(test_size):
            h, r, t = test_triplets[i]
            h, r, t = h.item(), r.item(), t.item()
            if mode == 'tail':
                tails = self.true_tails[time][(h, r)]
                tail_idx = np.array(list(map(lambda x: graph.ids[x], tails)))
                mask[i][tail_idx] = 1
                mask[i][graph.ids[t]] = 0
            elif mode == 'head':
                heads = self.true_heads[time][(r, t)]
                head_idx = np.array(list(map(lambda x: graph.ids[x], heads)))
                mask[i][head_idx] = 1
                mask[i][graph.ids[h]] = 0
        return mask.byte()

    def sort_and_rank(self, score, target):
        _, indices = torch.sort(score, dim=1, descending=True)
        indices = torch.nonzero(indices == target.view(-1, 1))
        indices = indices[:, 1].view(-1)
        return indices
