import os
import sys
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Union

import umap
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from rich.pretty import pprint
from torch_geometric.loader import DataLoader as GraphDataLoader
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from chem_mat_data.config import Config
from chem_mat_data.data import load_graphs
from chem_mat_data import ensure_dataset

mpl.use('Agg')

# == DATASET PARAMETERS ==

# :param DATASET_PATH:
#       Path to the dataset of graph dictionaries. This should be a JSON file which contains all the individual 
#       graph dictionaries indexed by integer keys (aka the index_data_map).
DATASET_PATH: str = 'zinc250'
# :param NUM_VAL:
#      Number of validation samples to be used in the dataset. If set to a positive integer, the dataset will
#      be subsampled to this number of samples. If set to None, all samples will be used.
#      If set to a float, it will be interpreted as a fraction of the total dataset size.
#      E.g. 0.8 means 80% of the dataset will be used for training.
NUM_VAL: Union[int, float] = 0.05
# :param NUM_TEST:
#      Number of test samples to be used in the dataset. If set to a positive integer, the dataset will
#      be subsampled to this number of samples. If set to None, all samples will be used.
#      If set to a float, it will be interpreted as a fraction of the total dataset size.
#      E.g. 0.1 means 10% of the dataset will be used for testing.
NUM_TEST: Union[int, float] = 0.05
# :param PROCESSING:
#       The processing instance to be used for processing the SMILES strings into the graph dictionary objects.
PROCESSING: ProcessingBase = MoleculeProcessing()

# == MODEL PARAMETERS ==

# :param CONTRASTIVE_TAU:
#       Temperature parameter for the contrastive loss. This controls the scale of the logits in the contrastive loss.
CONTRASTIVE_TAU: float = 1.0
# :param CLUSTER_DEPTH:
#       The depth of the clustering hierarchy. This controls how many levels the binary cluster tree has and therefore 
#       also controls the number of clusters that will be considered, which will therefore always be a power of 2, or 
#       in other words, the number of clusters will be 2 ** CLUSTER_DEPTH.
CLUSTER_DEPTH: int = 5
# :param CLUSTER_TAU:
#       The temperature parameter for the clustering contrastive loss. This controls the sharpness of the distribution 
#       of the cluster assignments. Very generally speaking lower values will lead to more aggressive cluster separation.
CLUSTER_TAU: float = 0.5
# :param CLUSTERING_FACTOR:
#       The weight with which the clustering loss is considered in the overall loss function of the network.
CLUSTERING_FACTOR: float = 1.0
# :param CLUSTERING_WARMUP_EPOCHS:
#       Generally, it is not recommendable to use the clutering loss from the very beginning of the training. It makes 
#       more sense to use it only once there is already a relatively good representation trained purely on the 
#       main contrastive loss. This parameter controls how many epochs the model is only trained on the contrastive 
#       loss before turning on the clustering loss.
CLUSTERING_WARMUP_EPOCHS: int = 5
# :param EPOCHS_PER_LEVEL:
#       It is also not always recommendable to perform the clustering of all the tree levels at the same time. Therefore, 
#       the training implements a slow annealing strategy where the clustering starts at the lowest level of the cluster 
#       tree, trains for a few epochs and then moves on to the next level of the cluster tree until the given CLUSTER_DEPTH
#       is reached. This parameter controls how many epochs the models is trained per incremental level.
EPOCHS_PER_LEVEL: int = 5
# :param PROJECTION_UNITS:
#       This list controls the number of layers and the hidden units in these layers to be used in the projection head of 
#       the contastive learning scheme. The last layer of this projection head will be used to calculate the similarities on
#       on for the contrastive loss.
PROJECTION_UNITS: List[int] = [256, 128, ]
# :param CLUSTER_UNITS:
#       This list controls the number of layers and the hidden units in these layers to be used in the clustering head of
#       the contastive clustering scheme. In addition to the network structure defined here there will always be added a final
#       linear layer which will map to the required number of units to represent the cluster tree with the given CLUSTER_DEPTH.
CLUSTER_UNITS: List[int] = [256, 128, ]

# == TRAINING PARAMETERS ==

# :param NUM_EPOCHS:
#       Number of epochs to train the model for. This is the number of times the model will see the entire training dataset.
#       If set to a positive integer, the model will be trained for this number of epochs.
NUM_EPOCHS: int = 40
# :param BATCH_SIZE:
#       Batch size to be used for training the model. This is the number of samples to be processed in a single 
#       forward/backward pass.
BATCH_SIZE: int = 512
# :param BATCH_ACCUMULATE:
#       Number of batches to accumulate gradients over before performing a backward pass. This is useful for
#       simulating a larger batch size without increasing the memory usage.
BATCH_ACCUMULATE: int = 2
# :param LEARNING_RATE:
#       Learning rate to be used for training the model. This is the step size at each iteration while moving toward 
#       a minimum of a loss function.
LEARNING_RATE: float = 1e-3

# == EXPERIMENT PARAMETERS == 

__DEBUG__: bool = False


experiment = Experiment.extend(
    'contrastive_graphs.py',
    namespace=file_namespace(__file__),
    base_path=folder_path(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=False, replace=True)
def load_dataset(e: Experiment) -> Dict[int, dict]:
    """
    In this implementation we load the ZINC250 dataset using the ChemMatData database.
    """

    ## --- gettting the dataset ---
    e.log('Initializing ChemMatData config...')
    config = Config()
    config.load()
    
    e.log(f'Downloading dataset {e.DATASET_PATH}...')
    dataset_path = ensure_dataset(
        dataset_name=e.DATASET_PATH,
        config=config,
        use_cache=True,
    )
    
    ## --- loading dataset ---
    # After the dataset was downloaded from the remote server to the local machine, we can load the information contained 
    # in the dataset into memory. The dataset is a list of graph dictionaries, where each dictionary contains the information
    # about a single graph.
    e.log('Loading graphs into memory...')
    graphs: list[dict] = load_graphs(
        path=dataset_path,
    )
    
    index_data_map: Dict[int, dict] = {
        index: {
            'graph': graph,
            'smiles': graph['graph_repr'], 
            'label': 0,
            'cluster_id': 0,
        }
        for index, graph in enumerate(graphs)
    }
    
    # Set the number of node features, edge features and labels into experiment memory for 
    # for the model initialization later on.
    example_graph = e.PROCESSING.process(next(iter(index_data_map.values()))['smiles'])
    e['num_node_features'] = np.array(example_graph['node_attributes']).shape[1]
    e['num_edge_features'] = np.array(example_graph['edge_attributes']).shape[1]
    e['num_labels'] = len(set(data['label'] for data in index_data_map.values()))
    
    return index_data_map


experiment.run_if_main()