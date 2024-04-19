import torch
import torch.nn as nn
from torch_scatter import scatter
from einops.layers.torch import Rearrange
import core.model_utils.gMHA_wrapper as gMHA_wrapper

from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN


class GraphMLPMixer(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0,
                 graph_level=False):

        super().__init__()
        self.dropout = dropout
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.graph_level = graph_level

        self.pooling = pooling
        self.res = res
        self.patch_rw_dim = patch_rw_dim

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(self.patch_rw_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=n_patches)

        self.transformer_encoder = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        self.output_decoder = MLP(nhid if self.graph_level else nhid*2, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())

        # Node PE
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        # Patch Encoder
        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)

        # Patch PE
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(data.patch_pe)
        mixer_x = self.reshape(subgraph_x)

        # MLPMixer
        mixer_x = self.transformer_encoder(mixer_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~data.mask)

        if self.graph_level:    
            # Global Average Pooling
            x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / \
                data.mask.sum(1, keepdim=True)
        else:
            mixer_x = Rearrange('B p d -> (B p) d')(mixer_x)
            x = torch.cat([x, mixer_x[batch_x]], dim=-1)
            x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')

        # Readout
        x = self.output_decoder(x)
        return x


class MPGNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean'):

        super().__init__()
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.pooling = pooling

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = GNN(nin=nhid, nout=nhid, nlayer_gnn=nlayer_gnn,
                        gnn_type=gnn_type, bn=bn, dropout=dropout, res=res)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = self.gnns(x, data.edge_index, edge_attr)

        # graph leval task
        x = scatter(x, data.batch, dim=0, reduce=self.pooling)
        x = self.output_decoder(x)

        return x


class GraphMLPMixer4TreeNeighbour(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 gMHA_type='MLPMixer',
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 patch_rw_dim=0):

        super().__init__()
        self.dropout = dropout
        self.n_patches = n_patches
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.nlayer_gnn = nlayer_gnn
        self.nlayer_mlpmixer = nlayer_mlpmixer
        self.pooling = pooling

        self.patch_rw_dim = patch_rw_dim

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)
        if self.patch_rw_dim > 0:
            self.patch_rw_encoder = MLP(patch_rw_dim, nhid, 1)

        self.layer0_keys = FeatureEncoder(node_type, nfeat_node, nhid)
        self.layer0_values = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=self.n_patches)
        self.mlp_mixer = getattr(gMHA_wrapper, gMHA_type)(
            nhid=nhid, dropout=mlpmixer_dropout, nlayer=nlayer_mlpmixer, n_patches=n_patches)

        self.output_decoder = MLP(
            nhid*2, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        # Patch Encoder
        x = data.x
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        # MLPMixer
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        if self.patch_rw_dim > 0:
            subgraph_x += self.patch_rw_encoder(data.patch_pe)
        mixer_x = self.reshape(subgraph_x)
        mixer_x = self.mlp_mixer(mixer_x, data.coarsen_adj if hasattr(
            data, 'coarsen_adj') else None, ~data.mask)

        # Node level task
        mixer_x = Rearrange('B p d -> (B p) d')(mixer_x)
        x = torch.cat([x, mixer_x[batch_x]], dim=-1)
        x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce='mean')

        # Readout
        root_nodes = x[data.root_mask]
        x = self.output_decoder(root_nodes)
        return x


class MPGNN4TreeNeighbour(nn.Module):
    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean'):

        super().__init__()
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.pooling = pooling

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.layer0_keys = FeatureEncoder(node_type, nfeat_node, nhid)
        self.layer0_values = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = GNN(nin=nhid, nout=nhid, nlayer_gnn=nlayer_gnn,
                        gnn_type=gnn_type, bn=bn, dropout=dropout, res=res)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        x = data.x
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed

        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = self.gnns(x, data.edge_index, edge_attr)

        # node leval task
        root_nodes = x[data.root_mask]
        x = self.output_decoder(root_nodes)

        return x
