import os
import json
import torch
import random
import numpy as np
from typing import Dict, List, Optional
from PIL import Image
from copy import deepcopy

import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch.utils.data import DataLoader
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.loader.dataloader import Collater
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing
from vgd_counterfactuals.generate.molecules import get_valid_bond_removals
from rdkit.Chem import RWMol

# ===========
# IMAGE DATA
# ===========

def load_mnist_index_data_map(mnist_path: str, load_images: bool = True) -> Dict[int, dict]:
    """
    Loads the information of the MNIST dataset at the given ``mnist_path`` into an index_data_map structure
    
    This function assumes that the given ``mnist_path`` is an absolute path to a folder containing the 
    MNIST dataset of handwritten digits. The folder structure should be as follows:
    
    - training
        - 0
        - 1
        - ...
        - 9
    - testing
        - 0
        - 1
        - ...
        - 9
    
    Each individual folder should contain the PNG images of the respective digits.    
    
    :param mnist_path: The absolute path to the folder containing the MNIST dataset.
    :param load_images: If set to True, the function will load the images into the index_data_map. If set
        to False, the function will only store the paths to the images.
    
    :returns: A dictionary mapping the integer index of the dataset element to a dictionary containing 
        additional information about that element, including the label and the path to the image file.
    """
    index_data_map: Dict[int, dict] = {}
    index: int = 0
    
    # ~ train elements
    training_path = os.path.join(mnist_path, 'training')
    training_folders = os.listdir(training_path)
    for folder_name in sorted(training_folders):
        label = int(folder_name)
        folder_path = os.path.join(training_path, folder_name)
        file_names = os.listdir(folder_path)
        for file_name in sorted(file_names):
            file_path = os.path.join(folder_path, file_name)
            index_data_map[index] = {
                'label': label,
                'image_path': file_path,
                'split': 'train'
            }
            if load_images:
                image = Image.open(file_path).convert('L')
                array = np.array(image)
                array = np.expand_dims(array, axis=0)
                index_data_map[index]['image'] = array
            
            index += 1
            
    # ~ test elements
    testing_path = os.path.join(mnist_path, 'testing')
    testing_folders = os.listdir(testing_path)
    for folder_name in sorted(testing_folders):
        label = int(folder_name)
        folder_path = os.path.join(testing_path, folder_name)
        file_names = os.listdir(folder_path)
        for file_name in sorted(file_names):
            file_path = os.path.join(folder_path, file_name)
            index_data_map[index] = {
                'label': label,
                'image_path': file_path,
                'split': 'test'
            }
            if load_images:
                image = Image.open(file_path).convert('L')
                array = np.array(image)
                array = np.expand_dims(array, axis=0)
                index_data_map[index]['image'] = array
            
            index += 1
            
    return index_data_map



class ImageData:
    
    def __init__(self, x: torch.Tensor, y: torch.Tensor, batch_size: int = 1, single: bool = True, **kwargs):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.single = single
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def to(self, device):
        for key, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(device))
                
        return self
        
    @classmethod
    def from_list(cls, data_list):
        
        keys = ['x', 'y']
        for key, value in vars(data_list[0]).items():
            if isinstance(value, torch.Tensor) and key not in keys:
                keys.append(key)
        
        key_value_dict = {}
        for key in keys:
            key_value_dict[key] = torch.stack([getattr(data, key) for data in data_list], dim=0)
            
        return cls(**key_value_dict, batch_size=len(data_list), single=False)
        

def collate_image_data(batch: List[ImageData]):
    data = ImageData.from_list(batch)
    return data


def data_from_image_dict(image_dict: dict) -> ImageData:
    
    x = torch.tensor(image_dict['image'], dtype=torch.float32)
    
    # If the label is not given, we will use 0 as a mock value.
    if 'label' in image_dict:
        y = torch.tensor(image_dict['label'])
    else:
        y = torch.tensor(0, dtype=torch.float32)

    return ImageData(x=x, y=y)


def data_list_from_image_dicts(image_dicts: List[dict]) -> List[ImageData]:
    
    # In this list, we'll store the ImageData elements
    data_list: List[ImageData] = []
    
    for image_dict in image_dicts:
        data = data_from_image_dict(image_dict)
        data_list.append(data)
        
    return data_list


class ImageDataLoader(DataLoader):
    
    def __init__(self, *args, **kwargs):
        DataLoader.__init__(self, *args, collate_fn=collate_image_data, **kwargs)


# ===========
# GRAPH DATA
# ===========


def load_graph_dataset(path: str) -> dict[int, dict]:
    """
    Loads a graph dataset from a JSON file at the given ``path``.

    The function expects the file at ``path`` to contain a JSON object mapping integer indices to graph dictionaries.
    Each graph dictionary should contain all necessary attributes for graph construction and analysis.

    :param path: The absolute path to the JSON file containing the graph dataset.
    :returns: A dictionary mapping integer indices to graph dictionaries.
    """
    with open(path, 'r') as file:
        index_data_map: dict[int, dict] = json.load(file)

    return index_data_map


def data_from_graph_dict(graph: dict) -> GraphData:
    """
    Converts a graph dictionary into a ``GraphData`` object.

    The input dictionary should contain the following keys:
    - 'node_attributes': Node feature matrix (list or array)
    - 'edge_index': Edge indices (list or array)
    - 'edge_attributes': Edge feature matrix (list or array)
    - 'graph_labels': Graph-level labels (list or array)

    All attributes are converted to PyTorch tensors with appropriate data types.

    :param graph: Dictionary containing graph attributes.
    :returns: A ``GraphData`` object with the specified attributes.
    """
    data = GraphData(
        x=torch.tensor(graph['node_attributes'], dtype=torch.float32),
        edge_index=torch.tensor(graph['edge_indices'], dtype=torch.long).t().contiguous(),
        edge_attr=torch.tensor(graph['edge_attributes'], dtype=torch.float32),
        y=torch.tensor(graph['graph_labels'], dtype=torch.float32),
    )
    
    return data


def data_list_from_graph_dicts(graphs: List[dict]) -> List[GraphData]:
    """
    Converts a list of graph dictionaries into a list of ``GraphData`` objects.

    Each graph dictionary in the input list is processed using ``data_from_graph_dict`` to create a corresponding
    ``GraphData`` object. The resulting objects are collected into a list.

    :param graphs: List of graph dictionaries, each containing graph attributes.
    :returns: List of ``GraphData`` objects constructed from the input dictionaries.
    """
    data_list = []
    for graph in graphs:
        data = data_from_graph_dict(graph)
        data_list.append(data)
        
    return data_list


class ContrastiveMoleculeGraphDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for contrastive learning on molecular graphs using SMILES strings.

    This dataset generates augmented graph views for each molecule, enabling contrastive learning approaches
    such as SimCLR or MoCo for graph-structured data. Augmentations are performed by applying random graph
    transformations (e.g., node removal) to the molecular graph representation.

    **Features:**
    - Accepts a list of SMILES strings representing molecules.
    - Applies graph augmentations using a configurable set of transformation functions.
    - Returns the original graph and two randomly augmented views for each sample when ``apply_augmentations`` is True.
    - Integrates with PyTorch and PyTorch Geometric for graph data processing.
    - Uses a processing pipeline (``MoleculeProcessing``) to convert SMILES to graph dictionaries.

    :param smiles_list: List of SMILES strings representing molecules in the dataset.
    :param apply_augmentations: If True, returns two augmented graph views for each sample (for contrastive learning).
    :param processing: MoleculeProcessing instance used to convert SMILES strings to graph dictionaries.
    :param transforms: List of string names of graph transformation functions to use for augmentation.

    **Example Usage:**
    .. code-block:: python

        from cohi_clustering.data import ContrastiveMoleculeGraphDataset
        dataset = ContrastiveMoleculeGraphDataset(
            smiles_list=[...],
            apply_augmentations=True,
            transforms=['remove_node']
        )
        data, data_aug_1, data_aug_2 = dataset[0]

    **Notes:**
    - Transformation functions must be defined as methods of the class and referenced by name in ``transforms``.
    - Augmentations are sampled randomly for each sample when ``apply_augmentations`` is True.
    - The dataset implements all required methods for PyTorch compatibility (``__len__``, ``__getitem__``).
    - For deterministic augmentations, use the underscore-prefixed transform methods directly.
    """
    
    def __init__(self, 
                 index_smiles_map: Dict[int, str], 
                 apply_augmentations: bool = True,
                 processing: MoleculeProcessing = MoleculeProcessing(),
                 transforms: list[str] = ['remove_edge', 'feature_noise'],
                 ):
        
        # This data structure maps string names of the various transforms that are *available* to the 
        # actual dataset. The keys are the string names of the transforms and the value are the 
        # actual transformation functions. So this map can be used to retreive the transformation 
        # function by its name.
        self.TRANSFORM_MAP: dict[int, callable] = {
            'remove_edge': self.transform_remove_edge,
            'feature_noise': self.transform_feature_noise,
        }
        
        ## --- properties ---
        self.index_smiles_map = index_smiles_map
        self.apply_augmentations = apply_augmentations
        self.processing = processing
        self.transforms = transforms
        
        ## --- computed properties ---
        self.indices: list[int] = list(index_smiles_map.keys())
        # This list will contain the transformation function that should be used when using the dataset.
        # It will be a list of callable functions (methods of the class) which take the molecule / graph 
        # as input and are supposed to return the transformed graph dict.     
        self.transform_funcs: list[callable] = [
            self.TRANSFORM_MAP[transform] 
            for transform in self.transforms 
        ]
        
    ## --- utility methods ---
    # These are functions which are needed to implement some common functionality.
        
    def mol_get_isolated_atoms(self, mol: Chem.Mol) -> list[int]:
        
        isolated_atoms = []
        for atom in mol.GetAtoms():
            if atom.GetDegree() == 1:
                isolated_atoms.append(atom.GetIdx())
                
        return isolated_atoms
    
    def get_subset(self, indices: List[int]) -> 'ContrastiveMoleculeGraphDataset':

        _index_smiles_map: dict[int, str] = {i: self.index_smiles_map[i] for i in indices}
        return ContrastiveMoleculeGraphDataset(
            index_smiles_map=_index_smiles_map,
            apply_augmentations=self.apply_augmentations,
            processing=self.processing,
            transforms=self.transforms,
        )
        
    ## --- transform functions ---
    # This section defines the transform functions that can be applied to the graphs in the dataset to get 
    # the augmented views for the contrastive learning.
    # For each transformation there are two methods defined: The method starting with the underscore is 
    # the deterministic version which accepts parameters that define how to transform is applied. The 
    # version without the underscore already applies the randomization to these parameters to yield a 
    # consistent interface.
    
    # def _transform_remove_edge(self, 
    #                           mol: Chem.Mol,
    #                           graph: dict,
    #                           atom_index: int
    #                           ) -> dict:
        
    #     # Create a mutable copy of the molecule
    #     rw_mol = RWMol(mol)

    #     # Remove the atom at the specified index
    #     rw_mol.RemoveAtom(atom_index)
        
    #     new_mol = rw_mol.GetMol()
    #     #rdmolops.Kekulize(new_mol, clearAromaticFlags=True)
    #     Chem.SanitizeMol(new_mol, catchErrors=True)

    #     # Generate a new graph dict from the modified molecule
    #     new_smiles = Chem.MolToSmiles(new_mol)
    #     assert new_smiles is not None, (
    #         'Failed to convert modified molecule to SMILES!'
    #     )
    #     assert '.' not in new_smiles, (
    #         'SMILES should not contain disconnected components (indicated by ".") after transformation!'
    #     )
        
    #     # Generate the graph dict represnetation to be removed from the molecule
    #     new_graph = self.processing.process(value=new_smiles)

    #     return new_graph
    
    def transform_remove_edge(self,
                              mol: Chem.Mol,
                              graph: dict,
                              ) -> dict:
        
        removals: list[dict] = get_valid_bond_removals(mol)
        
        if removals:
            removal = random.choice(removals)
            new_smiles = removal['value']
        else:
            new_smiles = Chem.MolToSmiles(mol)
        
        new_graph = self.processing.process(value=new_smiles)
        return new_graph
    
    def transform_feature_noise(self,
                                mol: Chem.Mol,
                                graph: dict,
                                noise_level: float = 0.01,
                                ) -> dict:
        
        node_attributes = np.array(graph['node_attributes'], dtype=np.float32)
        # Add Gaussian noise to the node attributes
        noise = np.random.normal(0, noise_level, node_attributes.shape)
        
        new_graph = deepcopy(graph)
        new_graph['node_attributes'] = node_attributes + noise
        
        return new_graph
    
    ## --- implement "Dataset" ---
    # The implementation of these methods is required to make the class work as a valid PyTorch Dataset.
        
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index: int) -> GraphData:
        
        ## --- creating graph representations ---
        # There we create all the various graph representations from the SMILES string given by the dataset
        # including the RDKIT mol object and the processed graph dict.
        smiles: str = self.index_smiles_map[self.indices[index]]
        mol: Chem.Mol = Chem.MolFromSmiles(smiles)
        assert mol is not None, f'Invalid SMILES string: {smiles}'
        
        graph: dict = self.processing.process(value=smiles)
        data: GraphData = data_from_graph_dict(graph)
        
        result = {
            'data': data,
            'aug_1': None,
            'aug_2': None,
        }
        
        if self.apply_augmentations:
            
            # Here we randomly sample two transform function from the list of available transforms 
            # and apply them to the graph.
            transform_1, transform_2 = random.choices(
                self.transform_funcs,
                k=2,
            )
            
            # apply first augmentation
            graph_aug_1 = transform_1(
                mol=mol,
                graph=graph,
            )
            data_aug_1 = data_from_graph_dict(graph_aug_1)
            result['aug_1'] = data_aug_1
            
            # apply second augmentation
            graph_aug_2 = transform_2(
                mol=mol,
                graph=graph,
            )
            data_aug_2 = data_from_graph_dict(graph_aug_2)
            result['aug_2'] = data_aug_2
            
        return result
    
    
class ContrastiveGraphDataLoader(torch.utils.data.DataLoader):
    """
    Custom DataLoader for the ContrastiveMoleculeGraphDataset.

    This DataLoader is designed to work with the ContrastiveMoleculeGraphDataset and provides a collate function
    that handles the specific structure of the dataset, including the original graph and its augmented views.
    
    Specifically, each element in the ConstrastiveMoleculeGraphDataset is a dictionary containing multiple different 
    graph Data objects, namely the original graph and two augmented views. This DataLoader is supposed to take care 
    of the collating process to ensure that these objects are batched correctly in their individual batched 
    graph Data representations. The collate function should therefore also return a dictionary with the same keys as 
    the individual elements of the dataset, but with the values now being the batched graph Data objects.
    
    :param dataset: The ContrastiveMoleculeGraphDataset instance to load data from.
    :param batch_size: Number of samples per batch.
    :param shuffle: Whether to shuffle the dataset at every epoch.
    :param num_workers: Number of subprocesses to use for data loading.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, collate_fn=self.collate_fn, **kwargs)
        
        # We use the Collater class from PyTorch Geometric to handle the collating process itself
        # This class simply takes a list of graph Data objects and collates them into as single 
        # batched Data object.
        self.geometric_collater = Collater(self.dataset, )
        
    def collate_fn(self, batch: dict):
        
        # dynamically infer the keys from the first element of the batch.
        # To make this more robust we could use the minimal common keys of all elements in the batch here.
        keys: list[str] = list(batch[0].keys())
        
        batches: dict[str, GraphData] = {}
        for key in keys:
            batches[key] = Batch.from_data_list([data[key] for data in batch])

        return batches
