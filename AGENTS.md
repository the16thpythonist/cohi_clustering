# Coding Agent instructions

# Guide for AI Agents

## Overview

This project contains research on hierarchical contrastive clustering. The general idea of contrastive 
clustering is to perform unsupervised clustering of a dataset by maximinizing the latent similarity of 
positive samples obtained through data augmentation while minimizing the latent similarity of
negative samples in a batch. Hierarchical contrastive clustering extends this idea by performing creating 
a binary cluster tree structure where the leaves are the individual cluster assignments but the 
internal structure is able to capture the hierarchical relationships between the clusters (e.g. cars and 
boats being examples of a larger "vehicles" cluster).

Specifically, this repository contains a re-implementation of a paper called [Hierarchical Contrastive Clustering](https://arxiv.org/abs/2303.03389) with the goal of applying the method that was created for image data to graph structured 
data and graph neural networks.

## Project Structure

- `/cohi_clustering`: Python source files
    - `/cohi_clustering/experiments`: Pycomex experiment files
    - `/cohi_clustering/templates`: Jinja2 templates
- `/tests`: Pytest unit tests which are names "tests_" plus the name of the source python file
    - `/tests/assets`: Additional files etc. which are needed by some of the unittests
    - `/tests/artifacts`: Temp folder in which the tests save their results

## Documentation

Type hints should be used wherever possible.
Every function/method should be properly documented by a Docstring using the **ReStructuredText** documentation style.
The doc strings should start with a brief summary of the function, followed by the parameters as illustrated by this example:

```python
def multiply(a: float, b: float) -> float:
    """
    Returns the product of ``a`` and ``b``.

    :param a: first float input value
    :param b: second float input value
    
    :returns: The float result
    """
    return a * b
```

## Computational Experiments

This project uses the [pycomex](https://github.com/the16thpythonist/pycomex) micro-framework for implementing and executing computational experiments.
A computational experiment can be defined according to the following example:

```python
from pycomex.functional.experiment import Experiment
from pycomex.utils import folder_path, file_namespace

# :param PARAM1:
#       Description for the parameter...
PARAM1: int = 100

__DEBUG__ = True # enables/disables debug mode

experiment = Experiment(
    base_path=folder_path(__file__),
    namespace=file_namespace(__file__),
    glob=globals()
)

@experiment.hook('util_function')
def util_function(e: Experiment, param: int):
    return param ** 2

@experiment
def experiment(e: Experiment):
    
    # automatically pushes to log file as well as to the stdout
    # parameters are accessed as attributes of the Experiment object
    e.log(f'this is a parameter value: {e.PARAM1}')

    # Store values into the experiment automatically
    e['value'] = value

    # The e.path contains the path to the artifacts folder of the 
    # current experiment run.
    e.log('artifacts path: {}')

    # Easily store figures to the artifacts folder
    fig: plt.Figure = ...
    e.commit_fig(fig, 'figure1.png')

    # using hooks instead of plain functions
    result = e.apply_hook(
        'util_function',
        param=20,
    )


experiment.run_if_main()
```

## Code Convention

1. functions with many parameters should be split like this:

```python
def function(arg1: List[int],
             arg2: List[float],
             **kwargs
             ) -> float:
    # ...

```

2. Strings that appear inside the code should use single quotes instead of double quotes, e.g. `'this is a string'`.

## Testing

Unittests use `pytest` in the `/tests` folder with this command

```bash
pytest -q -m "not localonly"
```

## Pull Requests / Contributing

Pull Requests should always start with a small summary of the changes and a list of the changed files.
Additionally a PR should contain a small summary of the tests results.