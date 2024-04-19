from core.model import *


def create_model(cfg):
    if cfg.dataset == 'ZINC':
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nfeat_node = 28
        nfeat_edge = 4
        nout = 1  # regression

    elif cfg.dataset == 'MNIST' or cfg.dataset == 'CIFAR10':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 5 if cfg.dataset == 'CIFAR10' else 3
        nfeat_edge = 1
        nout = 10

    elif 'ogbg' in cfg.dataset:
        nfeat_node = None
        nfeat_edge = None
        node_type = 'Atom'
        edge_type = 'Bond'
        if cfg.dataset == 'ogbg-moltox21':
            nout = 12
        elif cfg.dataset == 'ogbg-molhiv':
            nout = 1

    elif 'peptides' in cfg.dataset:
        # 'peptides-func' (10-task classification)
        # 'peptides-struct' (11-task regression)
        nfeat_node = None
        nfeat_edge = None
        node_type = 'Atom'
        edge_type = 'Bond'
        nout = 10 if cfg.dataset == 'peptides-func' else 11

    elif cfg.dataset == 'COCO-SP':
        nfeat_node = 14
        nfeat_edge = 2
        node_type = 'Linear'
        edge_type = 'Linear'
        nout = 81

    elif cfg.dataset == 'CSL':
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nfeat_node = 1
        nfeat_edge = 1
        nout = 10

    elif cfg.dataset == 'sr25-classify':
        nfeat_node = 2
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nout = 15

    elif cfg.dataset == 'exp-classify':
        nfeat_node = 2
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nout = 2

    elif cfg.dataset == 'TreeDataset':
        nfeat_node = 2 << cfg.depth
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nout = 2 << cfg.depth

    if cfg.metis.n_patches > 0:
        if cfg.dataset == 'TreeDataset':
            model = GraphMLPMixer4TreeNeighbour
        else:
            model = GraphMLPMixer
        return model(nfeat_node=nfeat_node,
                     nfeat_edge=nfeat_edge,
                     nhid=cfg.model.hidden_size,
                     nout=nout,
                     nlayer_gnn=cfg.model.nlayer_gnn,
                     node_type=node_type,
                     edge_type=edge_type,
                     nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                     gMHA_type=cfg.model.gMHA_type,
                     gnn_type=cfg.model.gnn_type,
                     rw_dim=cfg.pos_enc.rw_dim,
                     lap_dim=cfg.pos_enc.lap_dim,
                     pooling=cfg.model.pool,
                     dropout=cfg.train.dropout,
                     mlpmixer_dropout=cfg.train.mlpmixer_dropout,
                     n_patches=cfg.metis.n_patches,
                     patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                     graph_level=False)

    else:
        if cfg.dataset == 'TreeDataset':
            model = MPGNN4TreeNeighbour
        else:
            model = MPGNN
        return model(
            nfeat_node=nfeat_node,
            nfeat_edge=nfeat_edge,
            nhid=cfg.model.hidden_size,
            nout=nout,
            nlayer_gnn=cfg.model.nlayer_gnn,
            node_type=node_type,
            edge_type=edge_type,
            gnn_type=cfg.model.gnn_type,
            rw_dim=cfg.pos_enc.rw_dim,
            lap_dim=cfg.pos_enc.lap_dim,
            pooling=cfg.model.pool,
            dropout=cfg.train.dropout)
