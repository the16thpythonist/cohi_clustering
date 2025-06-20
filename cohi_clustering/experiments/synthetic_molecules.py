import os
from typing import List
from rdkit import Chem

from pycomex.functional.experiment import Experiment
from pycomex.utils import file_namespace, folder_path

# == EXPERIMENT PARAMETERS ==

CORE_FRAGMENTS: List[str] = ['c1ccccc1']
"""List of SMILES strings representing the base fragments."""

NEIGHBORHOOD_RADIUS: int = 1
"""Number of modification steps for variant generation."""

SIDE_GROUPS: List[str] = ['F', 'Cl', 'Br', 'O', 'N']
"""SMILES strings of simple side groups which may be attached to the fragments."""

__DEBUG__: bool = True

experiment = Experiment(
    namespace=file_namespace(__file__),
    base_path=folder_path(__file__),
    glob=globals(),
)


def get_variants(e: Experiment,
                 fragment_smiles: str,
                 radius: int,
                 side_groups: List[str] | None = None,
                 **kwargs
                 ) -> List[str]:
    """Return a list of variant SMILES for the given core fragment."""
    if side_groups is None:
        side_groups = getattr(e, 'SIDE_GROUPS', SIDE_GROUPS)
    mol = Chem.MolFromSmiles(fragment_smiles)

    candidate_atoms = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1 and atom.GetTotalNumHs() > 0
    ]

    mol = Chem.AddHs(mol)

    variants: List[str] = []
    for atom_idx in candidate_atoms:
        for group in side_groups:
            editable = Chem.RWMol(mol)
            atom = editable.GetAtomWithIdx(atom_idx)
            neighbors = [
                n.GetIdx() for n in atom.GetNeighbors() if n.GetAtomicNum() == 1
            ]
            if not neighbors:
                continue
            h_idx = neighbors[0]
            editable.RemoveBond(atom_idx, h_idx)
            editable.RemoveAtom(h_idx)
            new_atom = Chem.Atom(group)
            new_idx = editable.AddAtom(new_atom)
            editable.AddBond(atom_idx, new_idx, order=Chem.BondType.SINGLE)
            new_mol = editable.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
            except Chem.SanitizeException:
                continue
            new_mol = Chem.RemoveHs(new_mol)
            variants.append(Chem.MolToSmiles(new_mol))

    return list(sorted(set(variants)))


@experiment.hook('get_variants', default=True, replace=False)
def _hook_get_variants(e: Experiment,
                       fragment_smiles: str,
                       radius: int,
                       **kwargs
                       ) -> List[str]:
    return get_variants(e, fragment_smiles, radius, **kwargs)


@experiment
def experiment(e: Experiment):
    import pandas as pd

    rows: List[dict] = []
    for i, frag in enumerate(e.CORE_FRAGMENTS):
        name = f'fragment_{i}'
        variants = e.apply_hook(
            'get_variants',
            fragment_smiles=frag,
            radius=e.NEIGHBORHOOD_RADIUS,
        )
        for smi in variants:
            rows.append({'fragment': name, 'smiles': smi})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(e.path, 'dataset.csv')
    df.to_csv(csv_path, index=False)
    e['dataset_path'] = csv_path


experiment.run_if_main()
