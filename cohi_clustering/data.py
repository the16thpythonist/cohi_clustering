import os
import torch
import numpy as np
from typing import Dict, List, Optional
from PIL import Image


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
                image = np.array(Image.open(file_path))
                index_data_map[index]['image'] = image
            
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
                image = np.array(Image.open(file_path))
                index_data_map[index]['image'] = image
            
            index += 1
            
    return index_data_map



class ImageData:
    
    def __init__(self, x: torch.Tensor, y: torch.Tensor, single: bool = True, **kwargs):
        self.x = x
        self.y = y
        self.single = True
        
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
            
        return cls(**key_value_dict, single=False)
        

def collate_image_data(batch: List[ImageData]):
    data = ImageData.from_list(batch)
    return data


def data_from_image_dict(image_dict: dict) -> ImageData:
    
    x = torch.tensor(image_dict['image'])
    
    # If the label is not given, we will use 0 as a mock value.
    if 'label' in image_dict:
        y = torch.tensor(image_dict['label'])
    else:
        y = torch.tensor(0)

    return ImageData(x=x, y=y)


def data_list_from_image_dicts(image_dicts: List[dict]) -> List[ImageData]:
    
    # In this list, we'll store the ImageData elements
    data_list: List[ImageData] = []
    
    for image_dict in image_dicts:
        data = data_from_image_dict(image_dict)
        data_list.append(data)
        
    return data_list