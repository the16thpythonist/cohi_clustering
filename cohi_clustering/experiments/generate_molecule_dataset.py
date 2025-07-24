import os
import csv
import json
import random
from typing import List
from rdkit import Chem
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path
from vgd_counterfactuals.generate.molecules import get_neighborhood
from visual_graph_datasets.processing.base import ProcessingBase
from visual_graph_datasets.processing.molecules import MoleculeProcessing

from cohi_clustering.utils import CustomEncoder

mpl.use('Agg')

# == GENERATION PARAMETERS ==

# :param CORE_FRAGMENTS:
#       List of core fragments to be used in the dataset generation. Each of these fragments will be used 
#       as the base template for generating variants through single atom modifications.
CORE_FRAGMENTS: List[str] = [
    # 'C(Cl)1=C(Cl)C=C2C=C3C=C(Cl)C(Cl)=CC3=CC2=C1',
    # 'O=C1C2=C(N=CN2C)N(C)C(N1C)=O',
    # 'C1C(O)C(/C=C/CCC)C(CCCCC(N)C)C1=O',
    'C1=CC=C2C=CC(CO)=CC2=C1',
    'C(Cl)1=C(Cl)C=C2C=CC=CC2=C1',
    'N1=C(C)C=C2C=CC=C(I)C2=C1',
    'C1=C(CCCC)C=C2C=CC=CC2=C1',
    'CCCCO',
    'CC(CCN)CC'
]

# :param NEIGHBORHOOD_RADIUS:
#       Number of modification steps for variant generation. Radius 1 means that each core fragment will be 
#       modified only by using singular graph edits, while radius 2 means that the modifications will be 
#       applied to the already modified fragments recursively and so on.
NEIGHBORHOOD_RADIUS: int = 2

# :param SUBSAMPLE_SIZE:
#       If set to a positive integer, the variants of each original core fragment will be subsampled to maximum 
#       this number of variants. If set to None, all generated variants will be kept.
SUBSAMPLE_SIZE: Optional[int] = 1000

# :param PROCESSING_CLASS:
#       The processing instance to be used for processing the generated dataset of smiles strings into 
#       graph dictionary objects.
PROCESSING: ProcessingBase = MoleculeProcessing()

# == DATASET PARAMETERS ==

__DEBUG__: bool = True

experiment = Experiment(
    namespace=file_namespace(__file__),
    base_path=folder_path(__file__),
    glob=globals(),
)

@experiment.hook('generate_molecule_variants', default=True, replace=False)
def generate_molecule_variants(e: Experiment,
                               smiles: str,
                               radius: int,
                               **kwargs
                               ) -> List[str]:
    
    # This list will hold all the generated variants to be returned in the end, including 
    # the different neighborhood radii.
    variants: set[str] = {smiles, }
    
    for _ in range(radius):
        
        variants_copy = variants.copy()
        
        for smiles in variants_copy:
            neighbors: list[dict] = get_neighborhood(
                smiles=smiles,
                use_bond_additions=True,
                fix_protonation=False,
            )
        
            # The actual list of neighbors returene by the get_neighborhood function is a list of dicts where each dict 
            # contains also additional information about the modification and so on. So here we only need the actual 
            # SMILES which is stored under the 'value' key.
            for neighbor in neighbors:
                variants.add(neighbor['value'])

    return variants


@experiment.hook('save_dataset', default=True, replace=False)
def save_dataset(e: Experiment,
                 index_data_map: dict[int, dict],
                 **kwargs
                 ) -> None:

    ## --- saving as CSV file ---
    # First we want to save the dataset as a CSV file containing the SMILES strings and other meta information.
    e.log('saving dataset as CSV file of SMILES...')
    
    csv_path: str = os.path.join(e.path, 'dataset.csv')
    with open(csv_path, 'w') as file:
        
        dict_writer = csv.DictWriter(file, fieldnames=['smiles', 'original', 'cluster_id'])
        dict_writer.writeheader()

        for index, data in index_data_map.items():
            dict_writer.writerow({
                'smiles': data['smiles'],
                'original': data['orginal'],
                'cluster_id': data['cluster_id']
            })
    
    ## --- saving graph representations ---
    # Finally we want to save the full index_data_map as a JSON file containing the full graph representations
    # This will be what we will need to load later on when training the models.
    e.log('saving dataset as JSON file of graph dicts...')
    
    json_path: str = os.path.join(e.path, 'dataset.json')
    with open(json_path, 'w') as file:
        json.dump(index_data_map, file, cls=CustomEncoder)


@experiment
def experiment(e: Experiment):

    e.log(f'starting experiment to generate molecule dataset...')
    e.log(f' * based on {len(e.CORE_FRAGMENTS)} core fragments')
    e.log(f' * neighborhood radius: {e.NEIGHBORHOOD_RADIUS}')
    
    ## --- generating the dataset ---
    
    # This list will contain the dictionaries where each dict represents one molecule element. 
    # The dicts themselves contain the information about the SMILES representation of the moleucle 
    # as well as other information such as the original core fragment it was derived from.
    dataset: list[dict] = []
    
    e.log('generating dataset...')
    for fragment_id, fragment_smiles in enumerate(e.CORE_FRAGMENTS):
        
        e.log(f'  * generating variants for core fragment: {fragment_smiles} (ID: {fragment_id})')
        
        # :hook generate_molecule_variants:
        #       Given the SMILES representation of an original molecule fragment, this hook will generate 
        #       a set of variants by applying single atom modifications up to the specified neighborhood radius.
        #       returns a list of SMILES strings representing the generated variants.
        variants: list[str] = e.apply_hook(
            'generate_molecule_variants',
            smiles=fragment_smiles,
            radius=e.NEIGHBORHOOD_RADIUS,
        )
        e.log(f'   generated {len(variants)} variants')
        
        if e.SUBSAMPLE_SIZE is not None and len(variants) > e.SUBSAMPLE_SIZE:
            e.log(f'   subsampling to {e.SUBSAMPLE_SIZE} variants')
            # If the number of generated variants exceeds the subsample size, we will randomly sample 
            # the variants to the specified size.
            variants = random.sample(variants, e.SUBSAMPLE_SIZE)
            
        for variant_smiles in variants:
            data = {
                'smiles': variant_smiles,
                'orginal': fragment_smiles,
                'cluster_id': fragment_id,
                'label': fragment_id,
            }
            dataset.append(data)

    e.log(f'generated dataset with {len(dataset)} molecules overall')

    ## --- converting the dataset ---
    # The dataset will not be useful just with the SMILES string representations, we first need to convert those into 
    # generic graph dict representations which we can then save to the disk.
    
    e.log('converting dataset to graph dicts...')
    
    processing: MoleculeProcessing = e.PROCESSING
    
    # This data structure will contain the unique indices of the elements as the keys and the graph dict representations 
    # as the values.
    index_data_map: dict[int, dict] = {}

    for index, data in enumerate(dataset):
        
        graph = processing.process(
            value=data['smiles'],
            graph_labels=[data['cluster_id']]
        )
        
        index_data_map[index] = {
            **data,
            'graph': graph
        }
    
    ## --- saving the dataset ---
    # Finally after having created the dataset we need to save it to the disk in a structured way, such as a JSON 
    # file containing all the graph dict representations.
    
    e.log('saving dataset to disk...')
    # :hook save_dataset:
    #       This hook will save the dataset to the disk in a structured manner.
    e.apply_hook(
        'save_dataset', 
        index_data_map=index_data_map
    )    


experiment.run_if_main()
