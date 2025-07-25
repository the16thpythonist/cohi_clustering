|made-with-python| |python-version| |version|

.. |made-with-python| image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
   :target: https://www.python.org/

.. |python-version| image:: https://img.shields.io/badge/Python-3.8.0-green.svg
   :target: https://www.python.org/

.. |version| image:: https://img.shields.io/badge/version-0.1.0-orange.svg
   :target: https://www.python.org/

=================
⭐ Project Title
=================

.. important::

    Write a short project description or use the abstract of your publication !

=========================
📦 Installation by Source
=========================

.. code-block:: console

    git clone https://github.com/the16thpythonist/cohi_clustering

Install using ``pip``:

.. code-block:: console

    cd cohi_clustering
    python3 -m pip install .

**Check the installation.** Afterwards, you can check the install by invoking the CLI:

.. code-block:: console

    python3 -m cohi_clustering.cli --version
    python3 -m cohi_clustering.cli --help


=========================
📦 Installation by Source
=========================

.. important:: 

    delete this section if your code is not to be published as a python package

Install the latest stable release using ``pip``

.. code-block::

    pip3 install cohi_clustering

============
🚀 Quckstart
============

.. important:: 

    Use this section to create a minimal example of how to use the code in this repository. If your repository is mainly based on a number 
    of scripts, you could show how the most important scripts can be executed and what the most important parameters are. If your code is rather 
    used as a library you can write a simple code block that shows how to use the features of that library.

.. code-block:: python

    # The following code is just an example and not executable
    from cohi_clustering.dataset import Dataset
    from cohi_clustering.compute import Computation

    dataset = Dataest('name')
    computation = Computation(dataset)
    result = computation.compute()
    print(result)


============================
🧪 Computational Experiments
============================

This project makes use of the PyComex_ microframework for the implementation and management of computational experiments. 

All the computational experiments defined in this pacakge are accessible through a command line interface. To access the 
experimentation CLI, use the ``exp`` command like this:

.. code-block:: console

    python3 -m cohi_clustering.cli exp --help

**List of all experiments. ** To show a list of all available experiments, use the ``list`` command.

.. code-block:: console

    python3 -m cohi_clustering.cli exp list

**Experiment information. ** To show more information for a specific experiment, use the ``info`` command with 
one of the names from the list. This command will list additional information such as the full experiment description
and a list of parameters.

.. code-block:: console

    python3 -m cohi_clustering.cli exp info [experiment_name]

**Run an experiment. ** You can start the execution of an experiment with the ``run`` command. However, 
be aware that the execution of any experiment will most likely take a lot of time.

.. code-block:: console

    python3 -m cohi_clustering.cli exp run [experiment_name]

Each experiment will create a new archive folder, which will contain all the artifacts (such as visual
examples and the raw data) created during the runtime. The location of this archive folder can be found
from the output generated by the experiment execution.

==============
📖 Referencing
==============

.. note:: 

    delete this section if there is no publication to be cited yet

If you use, extend or otherwise reference our work, please cite the corresponding paper as follows:

.. code-block:: bibtex

    @article{
        title={Your Publication title},
        author={Mustermann, Max and Doe, John},
        journal={arxiv},
        year={2023},
    }


==========
🤝 Credits
==========

We thank the following packages, institutions and individuals for their significant impact on this package.

* PyComex_ is a micro framework which simplifies the setup, processing and management of computational
  experiments. It is also used to auto-generate the command line interface that can be used to interact
  with these experiments.

.. _PyComex: https://github.com/the16thpythonist/pycomex.git
.. _Cookiecutter: https://github.com/cookiecutter/cookiecutter