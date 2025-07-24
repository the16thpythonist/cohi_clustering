import numpy as np
from typing import List


def create_mock_image_dicts(num_images: int,
                            image_shape: tuple = (1, 28, 28)
                            ) -> List[dict]:
    """
    Creates a list of ``num_images`` image dict representations with random mock data with the 
    shape ``image_shape`` for testing purposes.
    
    :param num_images: The number of image dicts to create
    :param image_shape: The shape of the individual images to create which should be a tuple 
        (num_channels, height, width)
    
    :returns: A list of dicts
    """
    image_dicts: List[dict] = []
    for _ in range(num_images):
        image_dict = {
            'image_path': None,
            'image': np.random.rand(*image_shape),
            'label': np.random.randint(0, 10),
        }
        image_dicts.append(image_dict)
        
    return image_dicts


def create_mock_graph_dicts(num_graphs: int,
                            in_features: int = 10,
                            edge_features: int = 5,
                            num_node_range: tuple = (5, 15),
                            ) -> List[dict]:
    
    graph_dicts: List[dict] = []
    for _ in range(num_graphs):
        
        graph: dict[str, np.ndarray] = {}
        
        num_nodes = np.random.randint(*num_node_range)
        node_indices = np.arange(num_nodes)
        graph['node_indices'] = node_indices
        graph['node_attributes'] = np.random.rand(num_nodes, in_features)
        graph['edge_indices'] = np.array(list(zip(node_indices[:-1], node_indices[1:])), dtype=np.int32)
        graph['edge_attributes'] = np.random.rand(num_nodes - 1, edge_features)
        graph['graph_labels'] = np.expand_dims(np.random.randint(0, 10), axis=0)
        
        graph_dict = {
            'graph': graph,
            'label': np.random.randint(0, 10),
        }
        graph_dicts.append(graph_dict)
        
    return graph_dicts