from typing import Dict, List

import jinja2 as j2
from weasyprint import HTML, CSS
from cohi_clustering.utils import TEMPLATE_ENV


def create_cluster_report_pdf(cluster_infos: List[dict], 
                              output_path: str,
                              header_color: str = 'lightgray'):
    """
    Creates a multi-page PDF report containing information about the clusters that is specified in the 
    ``cluster_infos`` list. The report will be saved to the absolute file path specified in ``output_path``.
    
    ``cluster_infos`` is supposed to be a list consisting of dictionaries - one dict per cluster. Each of 
    those dicts should contain the following keys that represent information about that cluster.
    
    - "index": The unique integer identifier index of the cluster
    - "size": The number of samples associated with the cluster. 
    - "examples": A list of dictionaries, each containing the following information about the examples
      for that cluster.
        - "image_path": The absolute path to the image file that represents the example such that it 
          can be copied into the report.
    
    """
    html_template: j2.Template = TEMPLATE_ENV.get_template('cluster_report.html.j2')
    html_content: str = html_template.render({'cluster_infos': cluster_infos})
    html = HTML(string=html_content)
    
    css_template: j2.Template = TEMPLATE_ENV.get_template('cluster_report.css.j2')
    css_content: str = css_template.render({'header_color': header_color})
    css = CSS(string=css_content)
    
    html.write_pdf(output_path, stylesheets=[css])