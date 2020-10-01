Overview of Package
===================
This project formed part of the journal paper `An Artificial Intelligence Approach 
to Optimal Control of Sub-Nanosecond SOA-Based Optical Switches <https://ieeexplore.ieee.org/document/9124678?arnumber=9124678>`_. 
All data (`<https://doi.org/10.5522/04/12356696.v1>`_) 
used in the paper for PSO were generated with this code. The PSO algorithm is used 
to optimise the driving signal for SOA switches both in simulation and experiment. 

This ``soa`` package can be used to quickly and easily play around with different SOA simulations
and PSO hyperparameters.


Getting Started
===============
Follow the :doc:`instructions <Install>` to install this project, then have a look 
at the :doc:`tutorial <Tutorial>`. This project is not actively maintained, therefore
the best way to get started is to follow the simple tutorial example(s) to become familiar
with using some basic parts of the code, then have a look inside the code to understand
how the different methods and functions work. You can then edit the code yourself
according to what you want to investigate with respect to PSO and SOA drive signal
optimisation.

Unfortunately the Python code for this project is not the most clean, however
there are not too many lines of complex interdependencies/file structures to deal with and the
code is commented sufficiently such that you should be able to read through it
and adapt it as you wish. Intermediate-strong Python programmers are encouraged
to completely re-write parts they consider poorly implemented.


Free Software
=============
This is free software; you can redistribute it and/or modify it under the terms
of the Apache License 2.0. Contributions and forks are welcome - see the :doc:`contribute guide <Contribute>`. 
Contact cwfparsonson@gmail.com for questions.





Documentation
===============================

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Install
   Tutorial
   Contribute
   License
   Citing



Index
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
