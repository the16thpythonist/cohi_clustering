import os
import tempfile

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader

from cohi_clustering.testing import create_mock_image_dicts
from cohi_clustering.testing import create_mock_graph_dicts
from cohi_clustering.data import data_list_from_image_dicts
from cohi_clustering.data import collate_image_data
from cohi_clustering.data import data_list_from_graph_dicts
from cohi_clustering.models import SimCLRContrastiveCNN
from cohi_clustering.models import SimCLRContrastiveGNN


class TestContrastiveCNN:
    
    def test_construction_basically_works(self):
        
        model = SimCLRContrastiveCNN(
            input_shape=(1, 28, 28),
            resnet_units=64,
            embedding_units=[128, 256],
            projection_units=[256, 1024],
            contrastive_factor=1.0,
        )
        assert isinstance(model, SimCLRContrastiveCNN)
        
    def test_forward_basically_works(self):
        """
        If it generally works to do a forward pass with the model.
        """
        # Setting up the model
        model = SimCLRContrastiveCNN(
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
        model = SimCLRContrastiveCNN(
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
            loaded_model = SimCLRContrastiveCNN.load(model_path)
            loaded_model.eval()
            assert isinstance(loaded_model, SimCLRContrastiveCNN)
            
            # Performing a forward pass
            result_loaded = loaded_model(data)
            assert isinstance(result_loaded, dict)
            assert 'embedding' in result_loaded
            
            # Checking if the hyperparameters are the same
            print('result original', result['embedding'])
            print('result loaded', result_loaded['embedding'])
            
            assert torch.allclose(result['embedding'], result_loaded['embedding'])
            
            
class TestContrastiveGNN:
    
    def test_basically_works(self):
        """
        If it is possible to run a forward pass with the model and if the model is able to perform 
        a forward pass without an error.
        """
        
        batch_size = 5
        
        ## --- generating graphs --- 
        # At first we need to generate some mock input data to pass to the model later on.
        in_features = 10
        edge_features = 5
        graph_dicts = create_mock_graph_dicts(
            num_graphs=10,
            in_features=in_features,
            edge_features=edge_features,
        )
        
        data_list: list[GraphData] = data_list_from_graph_dicts([data['graph'] for data in graph_dicts])
        data_loader: GraphDataLoader = GraphDataLoader(data_list, batch_size=batch_size, shuffle=True)
        batch = next(iter(data_loader))
        
        ## --- model forward pass ---
        # Then we perform one model forward pass to generally check if the model is working or not.
        model = SimCLRContrastiveGNN(
            in_features=in_features,
            edge_features=edge_features,
        )
        
        result = model(batch)
        assert isinstance(result, dict)
        assert 'graph_embedding' in result
        assert len(result['graph_embedding']) == batch_size
    
    def test_saving_and_loading_basically_works(self):
        """
        If it is possible to save the model to the disk and load it again and it still gives the 
        same results.
        """
        
        batch_size = 5
        
        ## --- generating graphs --- 
        # At first we need to generate some mock input data to pass to the model later on.
        in_features = 10
        edge_features = 5
        graph_dicts = create_mock_graph_dicts(
            num_graphs=10,
            in_features=in_features,
            edge_features=edge_features,
        )
        
        data_list: list[GraphData] = data_list_from_graph_dicts([data['graph'] for data in graph_dicts])
        data_loader: GraphDataLoader = GraphDataLoader(data_list, batch_size=batch_size, shuffle=True)
        batch = next(iter(data_loader))
        
        ## --- model forward pass ---
        # Then we perform one model forward pass to generally check if the model is working or not.
        model = SimCLRContrastiveGNN(
            in_features=in_features,
            edge_features=edge_features,
        )
        model.eval()
        
        result = model(batch)
        assert isinstance(result, dict)
        
        ## --- saving and loading the model ---
        # We use the functions that the model defines to save it as a checkpoint first and then load it again.
        
        with tempfile.TemporaryDirectory() as temp_path:
            
            model_path = os.path.join(temp_path, 'model.ckpt')
            
            # Saving the model
            model.save(model_path)
            assert os.path.exists(model_path)
            
            # Loading the model
            loaded_model = SimCLRContrastiveGNN.load(model_path)
            assert isinstance(loaded_model, SimCLRContrastiveGNN)
            
        ## --- forward pass on loaded model ---
        # Then we do a forward pass on the loaded model and check if the results are the same as with the 
        # original model.
        
        result_loaded = loaded_model(batch)
        assert isinstance(result_loaded, dict)
        
        for key in ['graph_embedding', 'graph_clustering']:
            assert key in result
            assert key in result_loaded
            assert torch.allclose(result[key], result_loaded[key])