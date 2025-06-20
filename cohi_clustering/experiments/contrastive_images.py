import os
import sys
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Union

import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from rich.pretty import pprint
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from cohi_clustering.models import AbstractImageModel, SimCLRContrastiveCNN, ClusterLevelScheduler
from cohi_clustering.models import ClusteringCallback
from cohi_clustering.data import ImageDataLoader
from cohi_clustering.data import load_mnist_index_data_map
from cohi_clustering.data import data_list_from_image_dicts
from cohi_clustering.data import collate_image_data
from cohi_clustering.visualization import create_cluster_report_pdf
import umap
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from collections import Counter

mpl.use('Agg')

# == DATASET PARAMETERS ==

NUM_VAL: Union[int, float] = 0.05

NUM_TEST: Union[int, float] = 0.1

CONTRASTIVE_TAU: float = 0.3
CLUSTER_TAU: float = 1.0

EMBEDDING_UNITS: List[int] = [1024, 512]
PROJECTION_UNITS: List[int] = [2048]
CLUSTER_UNITS: List[int] = []
BATCH_ACCUMULATE: int = 8
CLUSTER_DEPTH: int = 4
CLUSTERING_WARMUP_EPOCHS: int = 5
EPOCHS_PER_LEVEL: int = 5

CLUSTERING_FACTOR: float = 1.0

# == TRAINING PARAMETERS ==

LEARNING_RATE: float = 1e-3
BATCH_SIZE: int = 128
EPOCHS: int = 50

__DEBUG__: bool = True

experiment = Experiment(
    namespace=file_namespace(__file__),
    base_path=folder_path(__file__),
    glob=globals(),
)


@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment,
                 **kwargs) -> Dict[int, dict]:
    # Loading the MNIST dataset
    mnist_path = '/media/data/Downloads/mnist_png'
    
    index_data_map: Dict = load_mnist_index_data_map(mnist_path)
    return index_data_map


@experiment.hook('train_test_split', default=True, replace=False)
def train_test_split(e: Experiment,
                     index_data_map: dict,
                     **kwargs
                     ) -> Tuple[list, list, list]:
    indices = list(index_data_map.keys())
    
    num_test = e.NUM_TEST
    if isinstance(e.NUM_TEST, float):
        num_test = int(e.NUM_TEST * len(indices))
    
    test_indices = random.sample(indices, k=num_test)
    indices = list(set(indices) - set(test_indices))
    
    num_val = e.NUM_VAL
    if isinstance(e.NUM_VAL, float):
        num_val = int(e.NUM_VAL * len(indices))
        
    val_indices = random.sample(indices, k=num_val)
    indices = list(set(indices) - set(val_indices))
    
    train_indices = indices
    
    return train_indices, val_indices, test_indices


@experiment.hook('train_model', default=True, replace=False)
def train_model(e: Experiment,
                index_data_map: dict, 
                train_indices: List[int],
                val_indices: List[int],
                **kwargs,
                ) -> AbstractImageModel:
    
    data_list_train = data_list_from_image_dicts([index_data_map[i] for i in train_indices])
    train_loader = ImageDataLoader(data_list_train, shuffle=True, batch_size=e.BATCH_SIZE)
    
    train_indices_reduced = random.sample(train_indices, k=int(0.1 * len(train_indices)))
    data_list_train_reduced = data_list_from_image_dicts([index_data_map[i] for i in train_indices_reduced])
    train_reduced_loader = ImageDataLoader(
        data_list_train_reduced,
        shuffle=False, 
        batch_size=500
    )
    
    image_dicts_val = [index_data_map[i] for i in val_indices]
    val_data_list = data_list_from_image_dicts(image_dicts_val)
    val_loader = ImageDataLoader(val_data_list, batch_size=500)
    
    class TrainingCallback(pl.Callback):
        
        def __init__(self):
            pl.Callback.__init__(self)
            self.prev_mapped = None
        
        def on_train_epoch_start(self, trainer, pl_module):
            
            pl_module.eval()
            device = pl_module.device
            pl_module.to('cpu')
            
            # TRACK METRICS
            # -------------
            
            # Iterate through all logged values and track them with the experiment
            for key, value in trainer.callback_metrics.items():
                if value is not None:
                    # Track each value using the experiment's track method
                    e.track(key, value.item())
            
            # VALIDATION FORWARD PASS
            # -----------------------
            
            e.log('running training forward pass...')
            results_train: List[dict] = pl_module.forward_image_dicts(train_reduced_loader)
            
            e.log('running validation forward pass...')
            results_val: List[dict] = pl_module.forward_image_dicts(val_loader)
            
            # LINEAR EVALUATION PERFORMANCE
            # -----------------------------
            
            e.log('evaluation linear evaluation performance...')
            # Prepare training data
            embeddings_train = np.stack([result['embedding'] for result in results_train], axis=0)
            labels_train = np.array([index_data_map[i]['label'] for i in train_indices_reduced])

            # Train Logistic Regression model
            e.log('training logistic regression model...')
            logistic_model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
            logistic_model.fit(embeddings_train, labels_train)

            # Prepare validation data
            embeddings_val = np.stack([result['embedding'] for result in results_val], axis=0)
            labels_val = np.array([index_data_map[i]['label'] for i in val_indices])

            # Evaluate the model
            e.log('evaluating logistic regression model...')
            predictions_val = logistic_model.predict(embeddings_val)
            accuracy = accuracy_score(labels_val, predictions_val)

            e.track('linear_accuracy', float(accuracy))
            e.log(f' * accuracy: {accuracy*100:.2f}%')
            
            # CLUSTERING PERFORMANCE
            # ----------------------
            
            e.log('evaluating k-means clustering performance...')
            
            kmeans = KMeans(n_clusters=10, random_state=0, n_init='auto')
            cluster_assignments = kmeans.fit_predict(embeddings_val)
            
            nmi = normalized_mutual_info_score(labels_val, cluster_assignments)
            e.track('nmi_raw', float(nmi))
            e.log(f' * NMI: {nmi:.4f}')
            
            # UMAP EMBEDDING SPACE
            # --------------------
            
            e.log('plotting umap of the embeddings...')
            embeddings = np.stack([result['embedding'] for result in results_val], axis=0)
            if embeddings.shape[1] > 2:
                reducer = umap.UMAP(n_components=2, min_dist=0.0, n_neighbors=100)
                mapped = reducer.fit_transform(embeddings)
            else:
                mapped = embeddings
                
            labels = np.array([data['label'] for data in image_dicts_val])
            
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
            
            # UMAP PROJECTION SPACE
            # ---------------------
            
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
            
            # TRACKING EMBEDDING PROPERTIES
            # -----------------------------
            
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
            
            # EVALUATING CLUSTERING
            # ---------------------
            
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
    
    e.log('setting up the model...')
    model = SimCLRContrastiveCNN(
        input_shape=e['input_shape'],
        resnet_units=64,
        embedding_units=e.EMBEDDING_UNITS,
        projection_units=e.PROJECTION_UNITS,
        cluster_units=e.CLUSTER_UNITS,
        cluster_depth=e.CLUSTER_DEPTH,
        contrastive_factor=1.0,
        contrastive_tau=e.CONTRASTIVE_TAU,
        cluster_factor=1.0,
        cluster_tau=e.CLUSTER_TAU,
        learning_rate=e.LEARNING_RATE,
    )
    
    data_list_train = data_list_from_image_dicts([index_data_map[i] for i in train_indices])
    train_loader = ImageDataLoader(data_list_train, shuffle=True, batch_size=e.BATCH_SIZE)
    
    e.log('starting model training...')    
    trainer = pl.Trainer(
        max_epochs=e.EPOCHS, 
        callbacks=[
            TrainingCallback(), 
            ClusteringCallback(warmup_epochs=e.CLUSTERING_WARMUP_EPOCHS, factor=e.CLUSTERING_FACTOR),
            ClusterLevelScheduler(warmup_epochs=e.CLUSTERING_WARMUP_EPOCHS, epochs_per_level=e.EPOCHS_PER_LEVEL),
        ],
        accumulate_grad_batches=e.BATCH_ACCUMULATE,
    )
    trainer.fit(model, train_dataloaders=[train_loader])
    
    print(f'Model is in train mode: {model.training}')
    
    model.eval()
    return model


@experiment.hook('evaluate_model', default=True, replace=False)
def evaluate_model(e: Experiment,
                   model: AbstractImageModel,
                   index_data_map: dict,
                   train_indices: List[int],
                   val_indices: List[int],
                   test_indices: List[int],
                   **kwargs):
    
    # ~ forward pass
    image_dicts_train = [index_data_map[i] for i in train_indices]
    train_data_list = data_list_from_image_dicts(image_dicts_train)
    train_loader = ImageDataLoader(train_data_list, shuffle=False, batch_size=500)
    
    e.log('running train forward pass...')
    results_train: List[dict] = model.forward_image_dicts(train_loader)
    
    image_dicts_test = [index_data_map[i] for i in test_indices]
    test_data_list = data_list_from_image_dicts(image_dicts_test)
    test_loader = ImageDataLoader(test_data_list, shuffle=False, batch_size=500)
    
    e.log('running test forward pass...')
    results_test: List[dict] = model.forward_image_dicts(test_loader)

    # ~ linear evaluation
    e.log('evaluating linear evaluation performance...')
    
    # Prepare training data
    embeddings_train = np.stack([result['embedding'] for result in results_train], axis=0)
    labels_train = np.array([index_data_map[i]['label'] for i in train_indices])
    
    # Train Logistic Regression model
    e.log('training logistic regression model...')
    logistic_model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
    logistic_model.fit(embeddings_train, labels_train)

    # Prepare validation data
    embeddings_test = np.stack([result['embedding'] for result in results_test], axis=0)
    labels_test = np.array([index_data_map[i]['label'] for i in test_indices])

    # Evaluate the model
    e.log('evaluating logistic regression model...')
    predictions_test = logistic_model.predict(embeddings_test)
    accuracy = accuracy_score(labels_test, predictions_test)

    e['metrics/linear_accuracy'] = float(accuracy)
    e.log(f' * accuracy: {accuracy*100:.2f}%')
    
    # ~ clustering report
    e.log('building cluster report...')
    #clusters = list(set([result['cluster_index'] for result in results_test]))
    
    # The keys of this dict will be the integer indices that identify the clusters that are predicted 
    # by the network and the values are Lists that contain the data element indices for the elements 
    # associated with that cluster.
    cluster_index_map: Dict[int, List[int]] = defaultdict(list)
    for index, result in zip(test_indices, results_test):
        cluster_index = int(result['cluster_index'])
        cluster_index_map[cluster_index].append(index)
        
    # This will be a list of dicts where each dict contains the information about one of the clusters 
    # which will then be compiled into the report.
    cluster_infos: List[dict] = []
    num_examples: int = 25
    for cluster_index, indices in cluster_index_map.items():
        
        num = min(num_examples, len(indices))
        example_indices = random.sample(indices, k=num)
        examples: List[dict] = []
        for index in example_indices:
            examples.append({
                'index': index,
                'image_path': index_data_map[index]['image_path'],
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
    
    e.log('experiments starting...')

    e.log('loading dataset...')
    index_data_map: Dict[int, dict] = e.apply_hook(
        'load_dataset'
    )
    e.log(f'loaded dataset of {len(index_data_map)} elements')
    example_data: dict = next(iter(index_data_map.values()))
    input_shape = example_data['image'].shape
    e.log(f'input shape: {input_shape}')
    e['input_shape'] = input_shape
    
    e.log('splitting dataset...')
    train_indices, val_indices, test_indices = e.apply_hook(
        'train_test_split',
        index_data_map=index_data_map
    )
    e.log(f' * train: {len(train_indices)}')
    e.log(f' * val: {len(val_indices)} ({e.NUM_VAL})')
    e.log(f' * test: {len(test_indices)} ({e.NUM_TEST})')
    e['indices/train'] = train_indices
    e['indices/val'] = val_indices
    e['indices/test'] = test_indices
    
    model = e.apply_hook(
        'train_model',
        index_data_map=index_data_map,
        train_indices=train_indices,
        val_indices=val_indices,
    )
    
    # ~ evaluating clustering on the test set
    
    e.log('Training finished. evaluating model...')
    e.apply_hook(
        'evaluate_model',
        model=model,
        index_data_map=index_data_map,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
    
experiment.run_if_main()