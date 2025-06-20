from cohi_clustering.experiments.synthetic_molecules import get_variants


def test_get_variants_produces_molecules():
    variants = get_variants(None, 'C', 1)
    assert isinstance(variants, list)
    assert len(variants) > 0
