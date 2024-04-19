from torch_geometric.datasets import LRGBDataset
from core.data_utils.peptides_functional import PeptidesFunctionalDataset
from core.data_utils.peptides_structural import PeptidesStructuralDataset

import numpy as np
import torch
from torch_geometric.transforms import Compose
from torch_geometric.utils import to_undirected
from core.config import cfg, update_cfg

from torch_geometric.datasets import GNNBenchmarkDataset
from torch_geometric.datasets import ZINC
from core.data_utils.sr25 import SRDataset
from core.data_utils.exp import PlanarSATPairsDataset
from core.data_utils.tree_dataset import TreeDataset
from core.transform import GraphPartitionTransform, PositionalEncodingTransform


def calculate_stats(dataset):
    num_graphs = len(dataset)
    ave_num_nodes = np.array([g.num_nodes for g in dataset]).mean()
    ave_num_edges = np.array([g.num_edges for g in dataset]).mean()
    print(
        f'# Graphs: {num_graphs}, average # nodes per graph: {ave_num_nodes}, average # edges per graph: {ave_num_edges}.')


class SuperpixelTransform(object):
    # combine position and intensity feature, ignore edge value
    def __call__(self, data):
        data.x = torch.cat([data.x, data.pos], dim=-1)
        data.edge_attr = None  # remove edge_attr
        data.edge_index = to_undirected(data.edge_index)
        return data


class CSLTransform(object):
    # combine position and intensity feature, ignore edge value
    def __call__(self, data):
        data.x = torch.zeros(data.num_nodes).long()
        return data


def create_dataset(cfg):
    pre_transform = PositionalEncodingTransform(
        rw_dim=cfg.pos_enc.rw_dim, lap_dim=cfg.pos_enc.lap_dim)

    if cfg.dataset == 'MNIST' or cfg.dataset == 'CIFAR10':
        transform_train = transform_eval = SuperpixelTransform()
    elif cfg.dataset == 'CSL':
        transform_train = transform_eval = CSLTransform()
    else:
        transform_train = transform_eval = None

    if cfg.metis.n_patches > 0:
        _transform_train = GraphPartitionTransform(n_patches=cfg.metis.n_patches,
                                                   metis=cfg.metis.enable,
                                                   drop_rate=cfg.metis.drop_rate,
                                                   num_hops=cfg.metis.num_hops,
                                                   is_directed=cfg.dataset == 'TreeDataset',
                                                   patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                                                   patch_num_diff=cfg.pos_enc.patch_num_diff)

        _transform_eval = GraphPartitionTransform(n_patches=cfg.metis.n_patches,
                                                  metis=cfg.metis.enable,
                                                  drop_rate=0.0,
                                                  num_hops=cfg.metis.num_hops,
                                                  is_directed=cfg.dataset == 'TreeDataset',
                                                  patch_rw_dim=cfg.pos_enc.patch_rw_dim,
                                                  patch_num_diff=cfg.pos_enc.patch_num_diff)

        if cfg.dataset in ['MNIST', 'CIFAR10', 'CSL']:
            transform_train = Compose([transform_train, _transform_train])
            transform_eval = Compose([transform_eval, _transform_eval])
        else:
            transform_train = _transform_train
            transform_eval = _transform_eval

    if cfg.dataset == 'ZINC':
        root = 'dataset/ZINC'
        train_dataset = ZINC(
            root, subset=True, split='train', pre_transform=pre_transform, transform=transform_train)
        val_dataset = ZINC(root, subset=True, split='val',
                           pre_transform=pre_transform, transform=transform_eval)
        test_dataset = ZINC(root, subset=True, split='test',
                            pre_transform=pre_transform, transform=transform_eval)

    elif cfg.dataset == 'MNIST' or cfg.dataset == 'CIFAR10':
        root = 'dataset'
        train_dataset = GNNBenchmarkDataset(
            root, cfg.dataset, split='train', pre_transform=pre_transform, transform=transform_train)
        val_dataset = GNNBenchmarkDataset(
            root, cfg.dataset, split='val', pre_transform=pre_transform, transform=transform_eval)
        test_dataset = GNNBenchmarkDataset(
            root, cfg.dataset, split='test', pre_transform=pre_transform, transform=transform_eval)

    elif 'ogbg' in cfg.dataset:
        from ogb.graphproppred import PygGraphPropPredDataset
        dataset = PygGraphPropPredDataset(
            cfg.dataset, 'dataset', pre_transform=pre_transform)
        split_idx = dataset.get_idx_split()
        train_dataset, val_dataset, test_dataset = dataset[split_idx['train']
                                                           ], dataset[split_idx['valid']], dataset[split_idx['test']]
        train_dataset.transform, val_dataset.transform, test_dataset.transform = transform_train, transform_eval, transform_eval

    elif cfg.dataset == 'peptides-func':
        dataset = PeptidesFunctionalDataset(
            root='dataset', pre_transform=pre_transform)
        split_idx = dataset.get_idx_split()
        train_dataset, val_dataset, test_dataset = dataset[split_idx['train']
                                                           ], dataset[split_idx['val']], dataset[split_idx['test']]
        train_dataset.transform, val_dataset.transform, test_dataset.transform = transform_train, transform_eval, transform_eval

    elif cfg.dataset == 'peptides-struct':
        dataset = PeptidesStructuralDataset(
            root='dataset', pre_transform=pre_transform)
        split_idx = dataset.get_idx_split()
        train_dataset, val_dataset, test_dataset = dataset[split_idx['train']
                                                           ], dataset[split_idx['val']], dataset[split_idx['test']]
        train_dataset.transform, val_dataset.transform, test_dataset.transform = transform_train, transform_eval, transform_eval

    elif cfg.dataset == 'COCO-SP':
        train_dataset = LRGBDataset(root='dataset', name=cfg.dataset, split='train', pre_transform=pre_transform)
        val_dataset = LRGBDataset(root='dataset', name=cfg.dataset, split='val', pre_transform=pre_transform)
        test_dataset = LRGBDataset(root='dataset', name=cfg.dataset, split='test', pre_transform=pre_transform)
        train_dataset.transform, val_dataset.transform, test_dataset.transform = transform_train, transform_eval, transform_eval

    elif cfg.dataset == 'PascalVOC-SP':
        train_dataset = LRGBDataset(root="dataset", name="PascalVOC-SP", split="train", pre_transform=pre_transform)
        val_dataset = LRGBDataset(root="dataset", name="PascalVOC-SP", split="val", pre_transform=pre_transform)
        test_dataset = LRGBDataset(root="dataset", name="PascalVOC-SP", split="test", pre_transform=pre_transform)
        train_dataset.transform, val_dataset.transform, test_dataset.transform = transform_train, transform_eval, transform_eval

    elif cfg.dataset == 'CSL':
        root = 'dataset'
        dataset = GNNBenchmarkDataset(
            root, cfg.dataset, pre_transform=pre_transform)
        return dataset, transform_train, transform_eval

    elif cfg.dataset == 'exp-classify':
        root = "dataset/EXP/"
        dataset = PlanarSATPairsDataset(root, pre_transform=pre_transform)
        return dataset, transform_train, transform_eval

    elif cfg.dataset == 'sr25-classify':
        root = 'dataset/sr25'
        dataset = SRDataset(root, pre_transform=pre_transform)
        dataset.transform = transform_eval
        dataset.data.x = dataset.data.x.long()
        # each graph is a unique class
        dataset.data.y = torch.arange(len(dataset.data.y)).long()
        dataset = [x for x in dataset]
        return dataset, dataset, dataset

    elif cfg.dataset == 'TreeDataset':
        root = 'dataset/TreeDataset'
        dataset = TreeDataset(root, cfg.depth)
        train_dataset, val_dataset, test_dataset = dataset.train, dataset.val, dataset.test
        if transform_train is not None:
            train_dataset = [transform_train(x) for x in train_dataset]
            val_dataset = [transform_train(x) for x in val_dataset]
            test_dataset = [transform_train(x) for x in test_dataset]
        return train_dataset, val_dataset, test_dataset

    else:
        print("Dataset not supported.")
        exit(1)

    torch.set_num_threads(cfg.num_workers)
    if not cfg.metis.online:
        train_dataset = [x for x in train_dataset]
    val_dataset = [x for x in val_dataset]
    test_dataset = [x for x in test_dataset]

    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    print("Generating data")

    cfg.merge_from_file('train/configs/GraphMLPMixer/zinc.yaml')
    cfg = update_cfg(cfg)
    cfg.metis.n_patches = 0
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)

    if cfg.dataset == 'CSL' or cfg.dataset == 'exp-classify':
        print('------------Dataset--------------')
        calculate_stats(train_dataset)
        print('------------------------------')
    else:
        print('------------Train--------------')
        calculate_stats(train_dataset)
        print('------------Validation--------------')
        calculate_stats(val_dataset)
        print('------------Test--------------')
        calculate_stats(test_dataset)
        print('------------------------------')
