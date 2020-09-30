Install
=======

Open Git Bash. Change the current working directory to the location where you want
to clone this `GitHub <https://github.com/cwfparsonson/soa_driving>`_ project, and run::

    $ git clone https://github.com/cwfparsonson/soa_driving.git

In ``soa_driving/pso``, install the required packages with either conda::

    $ conda install --file requirements/default.txt

or pip::

    $ pip install -r requirements/default.txt

Still in ``soa_driving/pso``, make the ``soa`` python module importable from anywhere
on your machine::

    $ python setup.py develop


You should then be able to import the soa module into your Python script from any directory
on your machine::

    >>> import soa
