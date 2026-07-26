import torch
from torch import nn
import torch.nn.functional as F

from history_validity_gate import (
    HistoryValidityAdapter,
    build_topk_candidate_ids,
    build_topk_history_features_dual,
    canonicalize_cygnet_queries,
    scatter_topk_back,
)


class link_prediction(nn.Module):
    def __init__(
        self,
        i_dim,
        h_dim,
        num_rels,
        num_times,
        time_stamp=1,
        alpha=0.5,
        use_cuda=False,
        use_history_gate=False,
        hva_topk=256,
        hva_mode="off",
        hva_gamma_exact=0.005,
        hva_gamma_near=0.08,
        hva_stale_init=0.2,
    ):
        super(link_prediction, self).__init__()

        self.i_dim = int(i_dim)
        self.h_dim = int(h_dim)
        self.num_rels = int(num_rels)
        self.num_times = int(num_times)
        self.time_stamp = int(time_stamp)
        self.alpha = float(alpha)
        self.use_cuda = bool(use_cuda)

        self.use_history_gate = bool(use_history_gate) and hva_mode in {"exact_only", "dual_branch"}
        self.hva_topk = int(hva_topk)
        self.hva_mode = str(hva_mode)

        self.ent_init_embeds = nn.Parameter(torch.Tensor(self.i_dim, self.h_dim))
        self.w_relation = nn.Parameter(torch.Tensor(self.num_rels, self.h_dim))
        self.tim_init_embeds = nn.Parameter(torch.Tensor(1, self.h_dim))

        self.generate_mode = Generate_mode(self.h_dim, self.h_dim, self.i_dim)
        self.copy_mode = Copy_mode(self.h_dim, self.i_dim, use_cuda=self.use_cuda)

        if self.use_history_gate:
            self.history_validity_adapter = HistoryValidityAdapter(
                num_relations=2 * self.num_rels,
                mode=self.hva_mode,
                gamma_exact=hva_gamma_exact,
                gamma_near=hva_gamma_near,
                stale_init=hva_stale_init,
            )
        else:
            self.history_validity_adapter = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.ent_init_embeds, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain("relu"))
        nn.init.xavier_uniform_(self.tim_init_embeds, gain=nn.init.calculate_gain("relu"))

    def _ensure_quadruple_tensor(self, quadruple):
        if torch.is_tensor(quadruple):
            return quadruple.long().to(self.ent_init_embeds.device)
        return torch.tensor(quadruple, dtype=torch.long, device=self.ent_init_embeds.device)

    def _ensure_copy_tensor(self, copy_vocabulary):
        if torch.is_tensor(copy_vocabulary):
            return copy_vocabulary.to(device=self.ent_init_embeds.device, dtype=torch.float32)
        return torch.tensor(copy_vocabulary, dtype=torch.float32, device=self.ent_init_embeds.device)

    def get_init_time(self, quadruple_tensor):
        time_idx = torch.div(
            quadruple_tensor[:, 3],
            self.time_stamp,
            rounding_mode="floor",
        ).long()
        time_idx = torch.clamp(time_idx, min=0, max=self.num_times - 1)

        device = self.ent_init_embeds.device
        steps = torch.arange(1, self.num_times + 1, device=device, dtype=self.tim_init_embeds.dtype).unsqueeze(1)
        init_tim = self.tim_init_embeds * steps
        return init_tim[time_idx]

    def get_raw_m_t(self, quadruple_tensor):
        h_idx = quadruple_tensor[:, 0]
        r_idx = quadruple_tensor[:, 1]
        h = self.ent_init_embeds[h_idx]
        r = self.w_relation[r_idx]
        return h, r

    def get_raw_m_t_sub(self, quadruple_tensor):
        t_idx = quadruple_tensor[:, 2]
        r_idx = quadruple_tensor[:, 1]
        t = self.ent_init_embeds[t_idx]
        r = self.w_relation[r_idx]
        return t, r

    def _apply_history_validity_adapter(self, base_log_scores, quadruple_tensor, entity, history_context):
        if (
            not self.use_history_gate
            or self.history_validity_adapter is None
            or history_context is None
        ):
            return base_log_scores

        canonical_queries_np = canonicalize_cygnet_queries(
            quadruple_tensor.detach().cpu().numpy(),
            entity=entity,
            num_rels=self.num_rels,
        )
        canonical_queries = torch.from_numpy(canonical_queries_np).to(
            device=base_log_scores.device,
            dtype=torch.long,
        )

        rel_ids = canonical_queries[:, 1]
        gold_ids = canonical_queries[:, 2]

        candidate_ids = build_topk_candidate_ids(
            base_scores=base_log_scores,
            gold_ids=gold_ids,
            topk_cands=self.hva_topk,
        )
        base_scores_topk = torch.gather(base_log_scores, 1, candidate_ids)

        with torch.no_grad():
            (
                seen_sr,
                dt_sr,
                freq_sr,
                seen_so,
                dt_so,
                freq_so,
                seen_ro,
                dt_ro,
                freq_ro,
            ) = build_topk_history_features_dual(
                query_triples=canonical_queries_np,
                candidate_ids=candidate_ids,
                sr_hist=history_context["sr"],
                so_hist=history_context["so"],
                ro_hist=history_context["ro"],
                device=base_log_scores.device,
                mode=self.hva_mode,
            )

        adjusted_topk_scores, _ = self.history_validity_adapter(
            base_scores_topk,
            rel_ids,
            seen_sr, dt_sr, freq_sr,
            seen_so, dt_so, freq_so,
            seen_ro, dt_ro, freq_ro,
        )
        return scatter_topk_back(base_log_scores, candidate_ids, adjusted_topk_scores)

    def forward(self, quadruple, copy_vocabulary, entity, history_context=None):
        quadruple_tensor = self._ensure_quadruple_tensor(quadruple)
        copy_tensor = self._ensure_copy_tensor(copy_vocabulary)

        if entity == "object":
            ent_embed, rel_embed = self.get_raw_m_t(quadruple_tensor)
        elif entity == "subject":
            ent_embed, rel_embed = self.get_raw_m_t_sub(quadruple_tensor)
        else:
            raise ValueError(f"Unsupported entity branch: {entity}")

        tim_embed = self.get_init_time(quadruple_tensor)

        score_g = self.generate_mode(ent_embed, rel_embed, tim_embed, entity)
        score_c = self.copy_mode(ent_embed, rel_embed, tim_embed, copy_tensor, entity)

        base_probs = score_c * self.alpha + score_g * (1.0 - self.alpha)
        base_probs = torch.clamp(base_probs, min=1e-12)
        base_log_scores = torch.log(base_probs)

        return self._apply_history_validity_adapter(
            base_log_scores=base_log_scores,
            quadruple_tensor=quadruple_tensor,
            entity=entity,
            history_context=history_context,
        )

    def regularization_loss(self, reg_param):
        regularization_loss = (
            torch.mean(self.w_relation.pow(2))
            + torch.mean(self.ent_init_embeds.pow(2))
            + torch.mean(self.tim_init_embeds.pow(2))
        )
        return regularization_loss * reg_param


class Copy_mode(nn.Module):
    def __init__(self, hidden_dim, output_dim, use_cuda):
        super(Copy_mode, self).__init__()
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.use_cuda = bool(use_cuda)
        self.tanh = nn.Tanh()
        self.W_s = nn.Linear(hidden_dim * 3, output_dim)

    def forward(self, ent_embed, rel_embed, time_embed, copy_vocabulary, entity):
        if entity == "object":
            m_t = torch.cat((ent_embed, rel_embed, time_embed), dim=1)
        elif entity == "subject":
            m_t = torch.cat((rel_embed, ent_embed, time_embed), dim=1)
        else:
            raise ValueError(f"Unsupported entity branch: {entity}")

        q_s = self.tanh(self.W_s(m_t))
        encoded_mask = (copy_vocabulary <= 0).float() * (-100.0)
        score_c = q_s + encoded_mask
        return F.softmax(score_c, dim=1)


class Generate_mode(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(Generate_mode, self).__init__()
        self.W_mlp = nn.Linear(hidden_size * 3, output_dim)

    def forward(self, ent_embed, rel_embed, tim_embed, entity):
        if entity == "object":
            m_t = torch.cat((ent_embed, rel_embed, tim_embed), dim=1)
        elif entity == "subject":
            m_t = torch.cat((rel_embed, ent_embed, tim_embed), dim=1)
        else:
            raise ValueError(f"Unsupported entity branch: {entity}")

        score_g = self.W_mlp(m_t)
        return F.softmax(score_g, dim=1)
