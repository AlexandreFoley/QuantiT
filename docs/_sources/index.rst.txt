.. QuantiT documentation master file, created by
   sphinx-quickstart on Mon Jan 11 15:04:13 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to QuantiT's documentation!
===================================
You can refer to the doxygen for an automatically formatted documentation.
This is formatted manually, and is a WIP.


QuantiT
-------

QuantiT is an open-source library that implements tensors with conservation laws, selection rules and algorithms to manipulate such tensors. Such tensors are commonly used in tensor network computations to exploit the symmetries of a problem in order to increase the efficiency and the accuracy of a simulation.

QuantiT is implemented in C++ and offers a python interface as well. Because its tensor implementation is backed by pytorch's tensors, computation can easily be done on any backend supported by pytorch, such as GPUs and CPUs.

Why Tensor Networks?
--------------------

Tensor networks allow us to work with a compressed representation of wave functions and operators. A well-designed tensor network allows us to work and optimize the compressed representation without doing a complete decompression. It has proven successful so far: there are many commonly used networks that are applied to problems for which storing a complete representation of a wave function would require several orders of magnitude more memory than there is on earth.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   
   Installation
   Roadmap1
   Pythoninterface
   CPPinterface
   Doxygen <../doxygen/html/index.html#http://>




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`



