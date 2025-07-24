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

from cohi_clustering.data import load_graph_dataset
from cohi_clustering.data import data_from_graph_dict, data_list_from_graph_dicts
from cohi_clustering.data import ContrastiveMoleculeGraphDataset
from cohi_clustering.data import ContrastiveGraphDataLoader
from cohi_clustering.models import AbstractGraphModel
from cohi_clustering.models import SimCLRContrastiveGNN
from cohi_clustering.models import ClusteringCallback, ClusterLevelScheduler
from cohi_clustering.visualization import create_cluster_report_pdf

mpl.use('Agg')

# == DATASET PARAMETERS ==

# :param DATASET_PATH:
#       Path to the dataset of graph dictionaries. This should be a JSON file which contains all the individual 
#       graph dictionaries indexed by integer keys (aka the index_data_map).
DATASET_PATH: str = '/media/ssd2/Programming/cohi_clustering/cohi_clustering/experiments/results/generate_molecule_dataset/debug/dataset.json'
# :param NUM_VAL:
#      Number of validation samples to be used in the dataset. If set to a positive integer, the dataset will
#      be subsampled to this number of samples. If set to None, all samples will be used.
#      If set to a float, it will be interpreted as a fraction of the total dataset size.
#      E.g. 0.8 means 80% of the dataset will be used for training.
NUM_VAL: Union[int, float] = 0.15
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
CLUSTER_DEPTH: int = 4
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
CLUSTERING_WARMUP_EPOCHS: int = 10
# :param EPOCHS_PER_LEVEL:
#       It is also not always recommendable to perform the clustering of all the tree levels at the same time. Therefore, 
#       the training implements a slow annealing strategy where the clustering starts at the lowest level of the cluster 
#       tree, trains for a few epochs and then moves on to the next level of the cluster tree until the given CLUSTER_DEPTH
#       is reached. This parameter controls how many epochs the models is trained per incremental level.
EPOCHS_PER_LEVEL: int = 10
# :param PROJECTION_UNITS:
#       This list controls the number of layers and the hidden units in these layers to be used in the projection head of 
#       the contastive learning scheme. The last layer of this projection head will be used to calculate the similarities on
#       on for the contrastive loss.
PROJECTION_UNITS: List[int] = [ ]
# :param CLUSTER_UNITS:
#       This list controls the number of layers and the hidden units in these layers to be used in the clustering head of
#       the contastive clustering scheme. In addition to the network structure defined here there will always be added a final
#       linear layer which will map to the required number of units to represent the cluster tree with the given CLUSTER_DEPTH.
CLUSTER_UNITS: List[int] = [256, 128, ]

# == TRAINING PARAMETERS ==

# :param NUM_EPOCHS:
#       Number of epochs to train the model for. This is the number of times the model will see the entire training dataset.
#       If set to a positive integer, the model will be trained for this number of epochs.
NUM_EPOCHS: int = 60
# :param BATCH_SIZE:
#       Batch size to be used for training the model. This is the number of samples to be processed in a single 
#       forward/backward pass.
BATCH_SIZE: int = 200
# :param BATCH_ACCUMULATE:
#       Number of batches to accumulate gradients over before performing a backward pass. This is useful for
#       simulating a larger batch size without increasing the memory usage.
BATCH_ACCUMULATE: int = 2
# :param LEARNING_RATE:
#       Learning rate to be used for training the model. This is the step size at each iteration while moving toward 
#       a minimum of a loss function.
LEARNING_RATE: float = 1e-3

# == EXPERIMENT PARAMETERS == 

__DEBUG__: bool = True

experiment = Experiment(
    namespace=file_namespace(__file__),
    base_path=folder_path(__file__),
    glob=globals(),
)

@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment,
                 ) -> Dict[int, dict]:
    """
    This default implementation uses the `load_graph_dataset` function to load the dataset from a json 
    file specified by the `DATASET_PATH` parameter in the experiment.
    """
    index_data_map: dict[int, dict] = load_graph_dataset(
        path=e.DATASET_PATH,
    )
    example_graph = next(iter(index_data_map.values()))['graph']
    e['num_node_features'] = np.array(example_graph['node_attributes']).shape[1]
    e['num_edge_features'] = np.array(example_graph['edge_attributes']).shape[1]
    e['num_labels'] = len(set(data['label'] for data in index_data_map.values()))
    
    return index_data_map


@experiment.hook('train_model', default=True, replace=False)
def train_model(e: Experiment,
                index_data_map: dict[int, dict],
                train_indices: List[int],
                val_indices: List[int],
                **kwargs,
                ) -> AbstractGraphModel:
    
    ## --- setting up dataset format ---
    # As part of the arguments we get the dataset in the format of the index_data_map which contains the 
    # graph dict representations of the graphs. For the training of the models however we need the graphs 
    # in the format of the PyG Data objects. Therefore we convert them here.
    e.log('converting graph dicts to graph data objects...')
    
    index_smiles_map: dict[int, str] = {i: data['smiles'] for i, data in index_data_map.items()}
    # This special dataset class will handle the augmented views of the graphs which are necessary for the 
    # contrastive learning approach. This needs to be handled in the dataset class itself instead of the 
    # model, due to the peculiar nature of the graph data structure. In the Data representation that 
    # enters the model, doing any kind of structural modifications (which change the number of nodes or edges)
    # is basically impossible. Therefore, these kinds of augmentations have to be applied before converting 
    # the graph structure to the Data representation - namely in the dataset class.
    dataset = ContrastiveMoleculeGraphDataset(
        index_smiles_map=index_smiles_map,
        apply_augmentations=True,
        processing=e.PROCESSING,
    )
    
    dataset_train = dataset.get_subset(train_indices)
    # The ContrastiveGraphDataLoader is a custom DataLoader that handles the specific structure of the dataset
    # and collates the data into batches of graph Data objects.
    # Each element of the dataset is actually a dictionary containing the original graph and two augmented views.
    # The DataLoader will take care of collating these into *separate* batched Data objects for each key, which 
    # will then finally be passed to the model.
    loader_train = ContrastiveGraphDataLoader(
        dataset= dataset_train,
        batch_size=e.BATCH_SIZE,
        shuffle=True,
        num_workers=24,
        persistent_workers=True,
        prefetch_factor=3,
    )
    
    indices_train_reduced = random.sample(train_indices, k=int(len(train_indices) * 0.01))
    dataset_train_reduced = dataset.get_subset(indices_train_reduced)
    loader_train_reduced = ContrastiveGraphDataLoader(
        dataset=dataset_train_reduced,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=12,
    )
    
    data_dicts_val = [index_data_map[i] for i in val_indices]
    dataset_val = dataset.get_subset(val_indices)
    loader_val = ContrastiveGraphDataLoader(
        dataset=dataset_val, 
        batch_size=e.BATCH_SIZE, 
        shuffle=False,
        num_workers=12,
    )
    
    ## --- training callback ---
    
    class TrainingCallback(pl.Callback):
        
        def __init__(self):
            pl.Callback.__init__(self)
            self.prev_mapped = None
        
        def on_train_epoch_start(self, trainer, pl_module):
            
            pl_module.eval()
            device = pl_module.device
            pl_module.to('cpu')
            
            ## --- track metrics ---
            
            # Iterate through all logged values and track them with the experiment so that they become 
            # part of the experiment archive data and are available for later analysis.
            for key, value in trainer.callback_metrics.items():
                if value is not None:
                    # Track each value using the experiment's track method
                    e.track(key, value.item())
            
            ## --- validation forward pass ---
            # Here we perform the forward pass on the validation set (and a smaller part of the training set)
            # to get the embeddings and the projections for all further analysis.
            
            e.log('running training forward pass...')
            results_train: List[dict] = pl_module.forward_graph_loader(loader_train_reduced)
            
            e.log('running validation forward pass...')
            results_val: List[dict] = pl_module.forward_graph_loader(loader_val)
            
            ## --- linear evaluation performance ---
            # One metric one can look at in terms of clustering performance is the linear evaluation performance.
            # Aka how well does a linear model perform given the embeddings and trained to predict the true 
            # clustering labels...
            
            e.log('evaluation linear evaluation performance...')
            # Prepare training data
            embeddings_train = np.stack([result['embedding'] for result in results_train], axis=0)
            labels_train = np.array([index_data_map[i]['label'] for i in indices_train_reduced])

            # Prepare validation data
            embeddings_val = np.stack([result['embedding'] for result in results_val], axis=0)
            labels_val = np.array([index_data_map[i]['label'] for i in val_indices])

            if e['num_labels'] > 1:

                # Train Logistic Regression model
                e.log('training logistic regression model...')
                logistic_model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
                logistic_model.fit(embeddings_train, labels_train)

                # Evaluate the model
                e.log('evaluating logistic regression model...')
                predictions_val = logistic_model.predict(embeddings_val)
                accuracy = accuracy_score(labels_val, predictions_val)

                e.track('linear_accuracy', float(accuracy))
                e.log(f' * accuracy: {accuracy*100:.2f}%')
            
            ## --- clustering performance ---
            
            if e['num_labels'] > 1:
                
                e.log('evaluating k-means clustering performance...')
                
                kmeans = KMeans(n_clusters=e['num_labels'], random_state=0, n_init='auto')
                cluster_assignments = kmeans.fit_predict(embeddings_val)
                
                nmi = normalized_mutual_info_score(labels_val, cluster_assignments)
                e.track('nmi_raw', float(nmi))
                e.log(f' * NMI: {nmi:.4f}')
            
            ## --- umap embedding space ---
            
            e.log('plotting umap of the embeddings...')
            embeddings = np.stack([result['embedding'] for result in results_val], axis=0)
            if embeddings.shape[1] > 2:
                reducer = umap.UMAP(n_components=2, min_dist=0.0, n_neighbors=100)
                mapped = reducer.fit_transform(embeddings)
            else:
                mapped = embeddings
                
            labels = np.array([data['label'] for data in data_dicts_val])
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.scatter(mapped[:, 0], mapped[:, 1], c=labels, cmap='Spectral', s=5)
            norm = mpl.colors.Normalize(vmin=0, vmax=9)
            cmap = mpl.cm.get_cmap('Spectral', 10)  # 'Spectral' colormap with 10 discrete colors
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.arange(10))
            cbar.set_label('Digit Class')
            cbar.ax.set_yticklabels(np.arange(10))
            ax.set_title(f'UMAP Embedding of Validation Set\n'
                         f'Epoch {trainer.current_epoch}')
            
            e.track('umap_embedding', fig)
            
            ## --- umap projection space ---
            
            e.log('plotting umap of the projections...')
            projections = np.stack([result['projection'] for result in results_val], axis=0)
            reducer = umap.UMAP(n_components=2, min_dist=0.0, n_neighbors=100)
            mapped = reducer.fit_transform(projections)
            
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.scatter(mapped[:, 0], mapped[:, 1], c=labels, cmap='Spectral', s=5)
            norm = mpl.colors.Normalize(vmin=0, vmax=9)
            cmap = mpl.cm.get_cmap('Spectral', 10)  # 'Spectral' colormap with 10 discrete colors
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=np.arange(10))
            cbar.set_label('Digit Class')
            cbar.ax.set_yticklabels(np.arange(10))
            ax.set_title(f'UMAP Projection of Validation Set\n'
                         f'Epoch {trainer.current_epoch}')
            
            e.track('umap_projection', fig)
            
            ## --- tracking embedding properties ---
            
            e.log('calculating embedding properties...')
            embedding_variance = np.mean(np.std(embeddings, axis=0))
            e.track('embedding_variance', float(embedding_variance))
            
            k = min(50, len(embeddings))  # Ensure k is not larger than the number of embeddings
            knn = NearestNeighbors(n_neighbors=k+1)  # k+1 because the first neighbor is the point itself
            knn.fit(embeddings)
            
            distances, indices = knn.kneighbors(embeddings)
            
            # The first column is the distance to itself, so we start from the second column
            avg_neighbor_distance = np.mean(np.mean(distances[:, 1:], axis=1))
            e.track('avg_neighbor_distance', float(avg_neighbor_distance))
            
            #pprint(e.data)
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            ax.plot(e.data['embedding_variance'], color='orange', label='embedding variance')
            ax.plot(e.data['avg_neighbor_distance'], color='orange', ls='--', label='avg neighbor distance')
            ax.set_title(f'Embedding Properties - Validation Set\n'
                         f'Epoch {trainer.current_epoch}')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            e.track('embedding_properties', fig)
            
            ## --- evaluating clustering ---
            
            e.log('evaluating cluster assignments...')
                
            cluster_index_map = defaultdict(list)
            cluster_label_map = defaultdict(lambda: defaultdict(int))
            for index, result, label in zip(val_indices, results_val, labels):
                cluster_index = int(result['cluster_index'])
                cluster_index_map[cluster_index].append(index)
                cluster_label_map[cluster_index][label] += 1
                
            print(result['cluster_probas'].shape, result['cluster_probas'], np.sum(result['cluster_probas']))
                
            e['cluster_index_map'] = dict(cluster_index_map)

            # Sort the cluster_label_map by cluster index
            cluster_label_map = dict(sorted(cluster_label_map.items(), key=lambda item: item[0]))
            pprint(cluster_label_map)
            
            pl_module.to(device)
            pl_module.train()
    
    ## --- model initialization ---
    e.log('initializing model ...')

    model: AbstractGraphModel = SimCLRContrastiveGNN(
        in_features=e['num_node_features'],
        edge_features=e['num_edge_features'],
        cluster_depth=e.CLUSTER_DEPTH,
        cluster_tau=e.CLUSTER_TAU,
        learning_rate=e.LEARNING_RATE,
        contrastive_tau=e.CONTRASTIVE_TAU,
        projection_units=e.PROJECTION_UNITS,
        cluster_units=e.CLUSTER_UNITS,
    )
    
    ## --- model training ---
    e.log('training model ...')
    
    trainer = pl.Trainer(
        max_epochs=e.NUM_EPOCHS,
        accelerator='gpu',
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[
            TrainingCallback(),
            ClusteringCallback(warmup_epochs=e.CLUSTERING_WARMUP_EPOCHS, factor=e.CLUSTERING_FACTOR),
            ClusterLevelScheduler(warmup_epochs=e.CLUSTERING_WARMUP_EPOCHS, epochs_per_level=e.EPOCHS_PER_LEVEL),
        ],
        accumulate_grad_batches=e.BATCH_ACCUMULATE,
    )
    trainer.fit(
        model=model,
        train_dataloaders=loader_train,
        val_dataloaders=loader_val,
    )
    
    return model


@experiment.hook('evaluate_model', default=True, replace=False)
def evaluate_model(e: Experiment,
                   model: AbstractGraphModel,
                   index_data_map: dict,
                   train_indices: List[int],
                   val_indices: List[int],
                   test_indices: List[int],
                   **kwargs):
    
     ## --- setting up dataset format ---
    # As part of the arguments we get the dataset in the format of the index_data_map which contains the 
    # graph dict representations of the graphs. For the training of the models however we need the graphs 
    # in the format of the PyG Data objects. Therefore we convert them here.
    e.log('converting graph dicts to graph data objects...')
    
    index_smiles_map: dict[int, str] = {i: data['smiles'] for i, data in index_data_map.items()}
    # This special dataset class will handle the augmented views of the graphs which are necessary for the 
    # contrastive learning approach. This needs to be handled in the dataset class itself instead of the 
    # model, due to the peculiar nature of the graph data structure. In the Data representation that 
    # enters the model, doing any kind of structural modifications (which change the number of nodes or edges)
    # is basically impossible. Therefore, these kinds of augmentations have to be applied before converting 
    # the graph structure to the Data representation - namely in the dataset class.
    dataset = ContrastiveMoleculeGraphDataset(
        index_smiles_map=index_smiles_map,
        apply_augmentations=True,
        processing=e.PROCESSING,
    )
    
    dataset_train = dataset.get_subset(train_indices)
    # The ContrastiveGraphDataLoader is a custom DataLoader that handles the specific structure of the dataset
    # and collates the data into batches of graph Data objects.
    # Each element of the dataset is actually a dictionary containing the original graph and two augmented views.
    # The DataLoader will take care of collating these into *separate* batched Data objects for each key, which 
    # will then finally be passed to the model.
    loader_train = ContrastiveGraphDataLoader(
        dataset= dataset_train,
        batch_size=e.BATCH_SIZE,
        shuffle=False,
        num_workers=24,
        persistent_workers=False,
        prefetch_factor=3,
    )
    
    dataset_test = dataset.get_subset(test_indices)
    loader_test = ContrastiveGraphDataLoader(
        dataset=dataset_test, 
        batch_size=e.BATCH_SIZE, 
        shuffle=False,
        num_workers=24,
        persistent_workers=False,
        prefetch_factor=3,
    )
    
    ## --- forward pass ---
    # We run a forward pass on the training and test set to get all the embeddings and cluster assignments.
    
    e.log('running train forward pass...')
    results_train: List[dict] = model.forward_graph_loader(loader_train)
    
    e.log('running test forward pass...')
    results_test: List[dict] = model.forward_graph_loader(loader_test)

    # --- linear evaluation performance ---
    # Training a linear classifier on the training embeddings to predict the cluster assignment. The test 
    # embeddings will then be used to evaluate the performance of this classifier.
    
    # Prepare training data
    embeddings_train = np.stack([result['embedding'] for result in results_train], axis=0)
    labels_train = np.array([index_data_map[i]['label'] for i in train_indices])
    
    # Prepare validation data
    embeddings_test = np.stack([result['embedding'] for result in results_test], axis=0)
    labels_test = np.array([index_data_map[i]['label'] for i in test_indices])
    
    if e['num_labels'] > 1:
        e.log('evaluating linear evaluation performance...')
        
        # Train Logistic Regression model
        e.log('training logistic regression model...')
        logistic_model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
        logistic_model.fit(embeddings_train, labels_train)

        # Evaluate the model
        e.log('evaluating logistic regression model...')
        predictions_test = logistic_model.predict(embeddings_test)
        accuracy = accuracy_score(labels_test, predictions_test)

        e['metrics/linear_accuracy'] = float(accuracy)
        e.log(f' * accuracy: {accuracy*100:.2f}%')
    
    ## --- umap embedding --- 
    # We cast the embeddings on the test set into a 2D space by using a UMAP projection and save the visualization
    # as an artifact into the experiment folder.
    
    e.log('plotting umap of the embeddings...')
    reducer = umap.UMAP(n_components=2, min_dist=0.0, n_neighbors=100)
    mapped = reducer.fit_transform(embeddings_test)
        
    labels = np.array([index_data_map[index]['label'] for index in test_indices])
    clusters = np.array([result['cluster_index'] for result in results_test])
    
    fig, (ax_true, ax_pred) = plt.subplots(ncols=2, nrows=1, figsize=(12, 6))
    ax_true.scatter(mapped[:, 0], mapped[:, 1], c=labels, cmap='Spectral', s=5)
    ax_true.set_title('UMAP Embedding of Test Set (True Clusters)')
    ax_pred.scatter(mapped[:, 0], mapped[:, 1], c=clusters, cmap='tab20', s=5)
    ax_pred.set_title('UMAP Embedding of Test Set (Pred. Clusters)')
    #ax_pred.set_title(f'UMAP Embedding of Test Set')
    
    e.commit_fig('umap_test.png', fig)
    
    ## --- clustering report ---
    # The clustering report is supposed to be a PDF document which illustrates the clusters that were found by 
    # illustrating some examples for each of the clusters.
    e.log('building cluster report...')
    
    # The keys of this dict will be the integer indices that identify the clusters that are predicted 
    # by the network and the values are Lists that contain the data element indices for the elements 
    # associated with that cluster.
    cluster_index_map: Dict[int, List[int]] = defaultdict(list)
    for index, result in zip(test_indices, results_test):
        cluster_index = int(result['cluster_index'])
        cluster_index_map[cluster_index].append(index)
        
    # This will be the path into which we store the example images of the various elements.
    image_folder_path = os.path.join(e.path, '.examples')
    os.mkdir(image_folder_path)
        
    # This will be a list of dicts where each dict contains the information about one of the clusters 
    # which will then be compiled into the report.
    cluster_infos: List[dict] = []
    num_examples: int = 25
    for cluster_index, indices in cluster_index_map.items():
        
        num = min(num_examples, len(indices))
        example_indices = random.sample(indices, k=num)
        examples: List[dict] = []
        for index in example_indices:
            
            # ~ creating the image!
            # For graph examples, we'll have to create the visualization of the graph and save it as an 
            # image explicitly - it does not yet exist.
            image_path = os.path.join(image_folder_path, f'{index}.png')
            processing: MoleculeProcessing = e.PROCESSING
            fig, _ = processing.visualize_as_figure(
                value=index_data_map[index]['smiles'],
                width=1000,
                height=1000,
            )
            fig.savefig(image_path)
            
            examples.append({
                'index': index,
                'image_path': image_path,
            })
            
        cluster_infos.append({
            'index': cluster_index,
            'size': len(indices),
            'examples': examples,
        })

    cluster_infos = list(sorted(cluster_infos, key=lambda info: info['index']))
    report_path = os.path.join(e.path, 'test_clusters.pdf')
    create_cluster_report_pdf(
        cluster_infos=cluster_infos,
        output_path=report_path,
    )


@experiment
def experiment(e: Experiment):

    e.log('starting experiment for contrastive clustering on graphs...')

    ## --- loading dataset ---
    # First of all we need to load the dataset of graph dictionaries into the memory from the persistent file storage.
    
    e.log('loading dataset...')
    # :hook load_dataset:
    #       This hook is used to load the dataset from the persistent file storage and return it in the format 
    #       of an index data map dictionary mapping integer indices to graph dictionaries. 
    #       In addition, this hook implementation internally also has to to set the number of node features, edge features 
    #       and labels to the experiment data, which will then be needed later on to create the model.
    index_data_map: dict[int, dict] = e.apply_hook(
        'load_dataset',
    )
    
    e.log(f'loaded dataset with {len(index_data_map)} elements.')
    indices: list[int] = list(index_data_map.keys())
    
    ## --- dataset splitting ---
    # After loading the dataset we now need to split it into the training, validation and test sets. 
    
    e.log('splitting dataset...')
    if isinstance(e.NUM_TEST, int):
        num_test = e.NUM_TEST
    elif isinstance(e.NUM_TEST, float):
        num_test = int(len(indices) * e.NUM_TEST)
    
    indices_test: list[int] = random.sample(indices, k=num_test)
    indices_train = list(set(indices) - set(indices_test))
    
    if isinstance(e.NUM_VAL, int):
        num_val = e.NUM_VAL
    elif isinstance(e.NUM_VAL, float):
        num_val = int(len(indices_train) * e.NUM_VAL)
    
    indices_val: list[int] = random.sample(indices_train, k=num_val)
    indices_train = list(set(indices_train) - set(indices_val))
    e.log(f' * using {len(indices_train)} training samples')
    e.log(f' * using {len(indices_val)} validation samples')
    e.log(f' * using {len(indices_test)} test samples')
    
    ## --- model training --- 
    
    e.log('model training...')

    # :hook train_model:
    #       This hook is used to train the model on the training set and validate it on the validation set as the 
    #       training progresses. The model should be returned at the end of the training process.
    model: AbstractGraphModel = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        train_indices=indices_train,
        val_indices=indices_val,
    )
    
    ## --- model saving ---
    # After having trained the model we need to save it to the disk so that it can be used later on for inference 
    # or as the bases for some finetuning etc.
    e.log('saving model to disk...')
    model_path = os.path.join(e.base_path, 'model.ckpt')
    model.save(path=model_path)
    
    ## --- model evaluation ---
    # In the end, we want to evaluate the model on the test set to see how well the contrastive clustering approach
    # has worked. And we want to create some visualization artifacts and so on.

    e.log('STARTING MODEL EVALUATION...')
    # :hook evaluate_model:
    #       This hook is used to evaluate the model on the test set and create some additional evaluation 
    #       artifacts like the UMAP embedding of the test set, the clustering report and so on.
    e.apply_hook(
        'evaluate_model',
        model=model,
        index_data_map=index_data_map,
        train_indices=indices_train,
        val_indices=indices_val,
        test_indices=indices_test,
    )

experiment.run_if_main()