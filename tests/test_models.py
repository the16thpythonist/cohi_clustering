import os
import tempfile

import torch
from torch.utils.data import DataLoader

from cohi_clustering.testing import create_mock_image_dicts
from cohi_clustering.data import data_list_from_image_dicts
from cohi_clustering.data import collate_image_data
from cohi_clustering.models import ContrastiveCNN


class TestContrastiveCNN:
    
    def test_construction_basically_works(self):
        
        model = ContrastiveCNN(
            input_shape=(1, 28, 28),
            resnet_units=64,
            embedding_units=[128, 256],
            projection_units=[256, 1024],
            contrastive_factor=1.0,
        )
        assert isinstance(model, ContrastiveCNN)
        
    def test_forward_basically_works(self):
        """
        If it generally works to do a forward pass with the model.
        """
        # Setting up the model
        model = ContrastiveCNN(
            input_shape=(1, 28, 28),
            resnet_units=64,
            embedding_units=[128, 256],
            projection_units=[256, 1024],
            contrastive_factor=1.0,
        )
        
        # Setting up the mock input data
        image_dicts = create_mock_image_dicts(num_images=10)
        data_list = data_list_from_image_dicts(image_dicts)
        data_loader = DataLoader(data_list, batch_size=16, collate_fn=collate_image_data)
        data = next(iter(data_loader))
        
        # performing a forward pass
        result = model(data)
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert isinstance(result['embedding'], torch.Tensor)
        
    def test_model_saving_and_loading_works(self):
        """
        If saving and loading the model to and from the disk works and if the loaded model 
        produces the same results as the original model.
        """
        # Setting up the model
        model = ContrastiveCNN(
            input_shape=(1, 28, 28),
            resnet_units=64,
            embedding_units=[128, 256],
            projection_units=[256, 1024],
            contrastive_factor=1.0,
        )
        model.eval()
        
        # Setting up the mock input data
        image_dicts = create_mock_image_dicts(num_images=10)
        data_list = data_list_from_image_dicts(image_dicts)
        data_loader = DataLoader(data_list, batch_size=16, collate_fn=collate_image_data)
        data = next(iter(data_loader))
        
        # performing a forward pass
        result = model(data)
        assert isinstance(result, dict)
        assert 'embedding' in result
        assert isinstance(result['embedding'], torch.Tensor)
        
        with tempfile.TemporaryDirectory() as tmp_path:
            model_path = os.path.join(tmp_path, 'model.ckpt')
            
            # Saving the model
            model.save(model_path)
            assert os.path.exists(model_path)
            
            # Loading the model
            loaded_model = ContrastiveCNN.load(model_path)
            loaded_model.eval()
            assert isinstance(loaded_model, ContrastiveCNN)
            
            # Performing a forward pass
            result_loaded = loaded_model(data)
            assert isinstance(result_loaded, dict)
            assert 'embedding' in result_loaded
            
            # Checking if the hyperparameters are the same
            print('result original', result['embedding'])
            print('result loaded', result_loaded['embedding'])
            
            assert torch.allclose(result['embedding'], result_loaded['embedding'])