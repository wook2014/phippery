
.. _sec_install_intro:

============
Installation
============

.. _sec_installation_phippery:

^^^^^^^^^^
Python API
^^^^^^^^^^

Installation of the ``phippery`` python package is available through

.. code-block::

  » pip install phippery

.. seealso:: for more information about how to contribute
  please see the :ref:`Development <sec_dev_intro>` page.


.. _sec_installation_phipflow:

^^^^^^^^^^^^^^^^^^^
Alignments Pipeline
^^^^^^^^^^^^^^^^^^^

The root `phippery Dockerfile <https://github.com/matsengrp/phippery/blob/1.2.0/Dockerfile>`_
installs phippery, Bowtie 1 and SAMtools. The separate
`PhIP-Flow V1.12 Nextflow configuration <https://github.com/matsengrp/phip-flow/blob/V1.12/nextflow.config>`_
uses ``quay.io/hdc-workflows/phippery:1.2.0`` for alignment and most other
tasks, with a separate image for edgeR and BEER.

.. note::
    Run Nextflow on the host with ``-profile docker`` to launch the
    tasks in their configured containers.

To install
`docker desktop <https://www.docker.com/products/docker-desktop>`_ 
(Desktop is just fine), 
on most unix OS:

.. code-block:: bash

   » curl -fsSL https://get.docker.com -o get-docker.sh && sudo sh get-docker.sh

To test the Docker install:

.. code-block:: bash

   » docker -v
   Docker version 20.10.1, build 831ebea
   » docker run docker/whalesay cowsay boo

To install
`Nextflow <https://www.nextflow.io/docs/latest/getstarted.html>`_
on most Unix OS:

.. code-block:: bash

   » curl -s https://get.nextflow.io | bash 

To test the Nextflow install:

.. code-block:: bash

   » nextflow -v
   nextflow version 20.04.1.5335


^^^^^^^^^^^^^
Streamlit App
^^^^^^^^^^^^^

We reccomend running streamlit through the 
`quay public image <https://quay.io/repository/matsengrp/phip-viz?tab=info>`_
we maintain.

Simply navigate to a directory where you have some ``.phip``
binary datasets and run

.. code-block::
 
   » docker run -p 8501:8501 -v $PWD:/app/data/ quay.io/matsengrp/phip-viz


for local installation

.. code-block::  

  » git clone https://github.com/matsengrp/phip-viz.git
  » (cd phip-viz && pip install -r requirements.txt)
