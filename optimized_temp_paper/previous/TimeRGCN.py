from baselines.StaticRGCN import StaticRGCN
from models.RGCN import RGCN
import dgl
import numpy as np
from utils.utils import comp_deg_norm, move_dgl_to_cuda
import torch.nn as nn
import torch
from utils.utils import node_norm_to_edge_norm
import math


class TimeRGCN(StaticRGCN):
    def __init__(self, args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test):
        super(TimeRGCN, self).__init__(args, num_ents, num_rels, graph_dict_train, graph_dict_val, graph_dict_test)

    def build_model(self):
        super().build_model()
        self.static_embed_size = math.floor(0.9 * self.embed_size)
        self.temporal_embed_size = self.embed_size - self.static_embed_size

        self.w_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))
        self.b_temp_ent_embeds = nn.Parameter(torch.Tensor(self.num_ents, self.temporal_embed_size))

        nn.init.xavier_uniform_(self.w_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.b_temp_ent_embeds, gain=nn.init.calculate_gain('relu'))


    def get_all_embeds_Gt(self, t, g, convoluted_embeds):
        all_embeds_g = self.ent_embeds.new_zeros(self.ent_embeds.shape)

        static_ent_embeds = self.ent_embeds
        ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)

        temp_ent_embeds = torch.sin(t * self.w_temp_ent_embeds.view(-1, self.temporal_embed_size) +
                                    self.b_temp_ent_embeds.view(-1, self.temporal_embed_size))
        input_embeddings = static_ent_embeds * torch.cat((ones, temp_ent_embeds), dim=-1)
        if self.args.use_embed_for_non_active:
            all_embeds_g[:] = input_embeddings[:]
        else:
            all_embeds_g[:] = self.ent_encoder.forward_isolated(input_embeddings)[:]

        for k, v in g.ids.items():
            all_embeds_g[v] = convoluted_embeds[k]
        return all_embeds_g


    def get_per_graph_ent_embeds(self, t_list, graph_train_list, val=False):
        if val:
            sampled_graph_list = graph_train_list
        else:
            sampled_graph_list = []
            for g in graph_train_list:
                src, rel, dst = g.edges()[0], g.edata['type_s'], g.edges()[1]
                total_idx = np.random.choice(np.arange(src.shape[0]), size=int(0.5 * src.shape[0]), replace=False)
                sg = g.edge_subgraph(total_idx, preserve_nodes=True)
                node_norm = comp_deg_norm(sg)
                sg.ndata.update({'id': g.ndata['id'], 'norm': torch.from_numpy(node_norm).view(-1, 1)})
                sg.edata['norm'] = node_norm_to_edge_norm(sg, torch.from_numpy(node_norm).view(-1, 1))
                sg.edata['type_s'] = rel[total_idx]
                sg.ids = g.ids
                sampled_graph_list.append(sg)

        # time_embeds = []
        # for t, g in zip(t_list, graph_train_list):
        #     temp_ent_embeds = torch.sin(t * self.w_ent_embeds[g.ndata['id']].view(-1, self.embed_size) +
        #                   self.b_ent_embeds[g.ndata['id']].view(-1, self.embed_size))
        #     time_embeds.append(temp_ent_embeds)

        ent_embeds = []
        for t, g in zip(t_list, graph_train_list):
            static_ent_embeds = self.ent_embeds[g.ndata['id']].view(-1, self.embed_size)
            ones = static_ent_embeds.new_ones(static_ent_embeds.shape[0], self.static_embed_size)
            temp_ent_embeds = torch.sin(t * self.w_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size) +
                                        self.b_temp_ent_embeds[g.ndata['id']].view(-1, self.temporal_embed_size))

            ent_embeds.append(static_ent_embeds * torch.cat((ones, temp_ent_embeds), dim=-1))


        batched_graph = dgl.batch(sampled_graph_list)
        batched_graph.ndata['h'] = torch.cat(ent_embeds, dim=0)
        if self.use_cuda:
            move_dgl_to_cuda(batched_graph)
        node_sizes = [len(g.nodes()) for g in graph_train_list]
        enc_ent_mean_graph = self.ent_encoder(batched_graph)
        ent_enc_embeds = enc_ent_mean_graph.ndata['h']
        per_graph_ent_embeds = ent_enc_embeds.split(node_sizes)

        return per_graph_ent_embeds
