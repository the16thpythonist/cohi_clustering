import os
from typing import List, Dict

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

from cohi_clustering.data import load_mnist_index_data_map


experiment = Experiment(
    namespace=file_namespace(__file__),
    base_path=folder_path(__file__),
    glob=globals(),
)

@experiment.hook('load_dataset', default=True, replace=False)
def load_dataset(e: Experiment,
                 **kwargs) -> Dict[int, dict]:
    # Loading the MNIST dataset
    mnist_path = '/home/jonas/Downloads/mnist_png'
    
    index_data_map: Dict = load_mnist_index_data_map(mnist_path)


@experiment
def experiment(e: Experiment):
    
    e.log('experiments starting...')
    
    
experiment.run_if_main()