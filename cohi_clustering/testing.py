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