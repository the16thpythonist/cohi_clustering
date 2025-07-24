import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
from torch_geometric.data import Data as GraphData

from cohi_clustering.data import load_mnist_index_data_map
from cohi_clustering.data import ImageData, collate_image_data
from cohi_clustering.data import ContrastiveMoleculeGraphDataset
from cohi_clustering.data import ContrastiveGraphDataLoader

from .util import ARTIFACTS_PATH


def test_load_mnist_index_data_map():   
    """
    If it generally works to load the MNIST dataset from the file system.
    """
    index_data_map: dict = load_mnist_index_data_map('/media/data/Downloads/mnist_png')
    assert isinstance(index_data_map, dict), "Should return a dictionary"
    assert len(index_data_map) > 0
    print(f'number of elements: {len(index_data_map)}')
    
    example_data = index_data_map[0]
    
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    image = example_data['image']
    image = image.squeeze()  # Fix shape for matplotlib
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    fig_path = os.path.join(ARTIFACTS_PATH, 'test_load_mnist_index_data_map.png')
    fig.savefig(fig_path)
    

class TestImageData:
    
    def test_collate_image_data_with_dataloader(self):
        """
        If it is possible to use the ``DataLoader`` class to obtain the collated/batched versions
        of a list of ImageData objects.
        """
        # ~ testing normal functionality
        data1 = ImageData(x=torch.rand(1, 28, 28), y=torch.tensor(0))
        data2 = ImageData(x=torch.rand(1, 28, 28), y=torch.tensor(1))
        
        data_list = [data1, data2]
        data_loader = DataLoader(data_list, batch_size=16, collate_fn=collate_image_data)
        
        for data in data_loader:
            print('data', type(data), data)
            assert isinstance(data, ImageData)
            assert data.x.shape[0] == 2
            assert data.y.shape[0] == 2
            
        # ~ testing dynamically attaching attributes
        data1 = ImageData(x=torch.rand(1, 28, 28), y=torch.tensor(0), split=torch.tensor(0))
        data2 = ImageData(x=torch.rand(1, 28, 28), y=torch.tensor(1), split=torch.tensor(1))
        
        data_list = [data1, data2]
        data_loader = DataLoader(data_list, batch_size=16, collate_fn=collate_image_data)
        
        for data in data_loader:
            assert isinstance(data, ImageData)
            # Since we added the additionaly "split" attribute for the individual elements, we also
            # need to find it in the collated data object.
            assert hasattr(data, 'split')
            # The first dimension should be the batch dimension here as well!
            assert data.split.shape[0] == 2
            
    def test_moving_image_data_to_device_works(self):
        """
        It if is possible to use the ``ImageData.to`` method to move the tensors contained in the 
        ImageData object to a different device.
        """
        # First of all testing if it works to move the image data to the cuda device.
        image_data = ImageData(x=torch.rand(1, 28, 28), y=torch.tensor(0))
        image_data = image_data.to('cuda')
        assert image_data.x.device.type == 'cuda'
        assert image_data.y.device.type == 'cuda'
        
        # Now test if I can put it back to the cpu.
        image_data = image_data.to('cpu')
        assert image_data.x.device.type == 'cpu'
        assert image_data.y.device.type == 'cpu'
        
        
class TestConstrastiveMoleculeGraphDataset:
    
    def test_basically_works_withoug_augmentations(self):
        
        index_smiles_map = {0: 'CCO', 1: 'CCN', 2: 'CCC'}
        
        dataset = ContrastiveMoleculeGraphDataset(
            index_smiles_map=index_smiles_map,
            apply_augmentations=False,
        )
        for data in dataset:
            
            # without having the augmentations enabled, each element of the dataset should be a single dict
            # containing the graph data
            assert isinstance(data, dict)
            assert isinstance(data['data'], GraphData)
            assert data['aug_1'] is None
            assert data['aug_2'] is None
            
    def test_basically_works_with_augmentations(self):
        
        index_smiles_map = {0: 'CCO', 1: 'CCN', 2: 'CCC'}
        
        dataset = ContrastiveMoleculeGraphDataset(
            index_smiles_map=index_smiles_map,
            apply_augmentations=True,
        )
        for data in dataset:
            
            # with the augmentations enabled, each element of the dataset should be a tuple of three Data objects
            # representing the original graph and two augmented versions.
            assert isinstance(data, dict)
            assert isinstance(data['data'], GraphData)
            assert isinstance(data['aug_1'], GraphData)
            assert isinstance(data['aug_2'], GraphData)
            

class TestContrastiveGraphDataLoader():
    
    def test_basically_works(self):
        
        ## --- setup dataset ---
        # To test the DataLoader, we first need to setup the dataset
        
        index_smiles_map = {0: 'CCO', 1: 'CCNCCC', 2: 'CCC', 3: 'CCCC'}
        dataset = ContrastiveMoleculeGraphDataset(
            index_smiles_map=index_smiles_map,
            apply_augmentations=True,
        )
        
        ## --- setup DataLoader ---
        # Now we can setup the DataLoader to load the dataset in batches.
        
        loader = ContrastiveGraphDataLoader(
            dataset=dataset,
            batch_size=2,
            shuffle=False,  # Shuffle true made issues before
            drop_last=True,
            num_workers=0,
        )
        batch = next(iter(loader))
        assert isinstance(batch, dict)
        for key, value in batch.items():
            assert isinstance(value, GraphData)
            # Batch size should be 2 as we set it 
            batch_size = np.max(value.batch.numpy()) + 1
            assert batch_size == 2