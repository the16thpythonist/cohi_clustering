import os
from cohi_clustering.visualization import create_cluster_report_pdf

from .util import ASSETS_PATH, ARTIFACTS_PATH


def test_create_cluster_report_basically_works():
    
    report_path = os.path.join(ARTIFACTS_PATH, 'test_create_cluster_report_basically_works.pdf')
    image_path = os.path.join(ASSETS_PATH, 'placeholder.png')
    cluster_infos = [
        {
            'index': 0,
            'examples': [{'image_path': image_path}] * 10
        },
        {
            'index': 1,
            'examples': [{'image_path': image_path}] * 10
        },
    ]
    create_cluster_report_pdf(cluster_infos, output_path=report_path)