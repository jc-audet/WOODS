.. WOODS documentation master file, created by
   sphinx-quickstart on Tue Nov  2 19:46:03 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

WOODS
=================================

.. image:: assets/banner.png
    :width: 600
    :align: center

WOODS is a project aimed at investigating the implications of Out-of-Distribution generalization problems in sequential data along with it’s possible solution. To that goal, we offer a DomainBed-like suite to test domain generalization algorithms on our WILDS-like set of sequential data benchmarks inspired from real world problems of a wide array of common modalities in modern machine learning.


.. toctree::
   :maxdepth: 1
   :caption: Getting started:

   installation.md
   downloading_datasets.md
   running_a_sweep.md
   adding_an_algorithm.md
   adding_a_dataset.md
   contributing.md


API Documentation
---------------------
.. toctree::
    :caption: API documentation
    :hidden:
    :maxdepth: 3

    WOODS <woods>

.. autosummary::

   woods
   woods.objectives
   woods.datasets
   woods.hyperparams
   woods.train
   woods.models
   woods.model_selection
   woods.command_launchers
   woods.utils

.. autosummary::

   woods.scripts
   woods.scripts.main
   woods.scripts.download
   woods.scripts.hparams_sweep
   woods.scripts.compile_results
   woods.scripts.visualize_results
   woods.scripts.fetch_and_preprocess
   


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
