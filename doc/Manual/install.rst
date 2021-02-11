.. _install:

Installation
============

Qubefit has been tested on both macOS as well as Linux (Red Hat
7). To install the code clone or download the latest version from
github: `qubefit <https://github.com/mneeleman/qubefit>`_.
After downloading the code (and unzipping if necessary), open up a
terminal window, navigate to the directory where the code lives, which contains the setup.py file, and type::

  python setup.py install
  
If everything goes correctly, this installs the code. To test if the installation was
successful, see if the GUI loads. Use the command line to put yourself Inside the examples folder ``qubefit/examples`` and then type::

  qubemom WolfeDiskCube.fits

and ::

    qfgui WolfeDiskSetup.py

If this brings up the two GUIs such as shown in the section :ref:`guis`, the code was installed correctly.

Dependencies
------------

Qubefit has been tested on python 3.8 and depends heavily on the
following packages, where the version numbers are the version used for
testing the code:

  * ``numpy (v1.19.2)``
  * ``scipy (v1.5.2)``
  * ``astropy (v4.0.2)``
  * ``matplotlib (v3.3.2)``
  * ``h5py (2.10.0)``
  * ``scikit-image (v0.17.2)``
  * ``tqdm (v4.50.2)``
  * ``emcee (v3.0.2)``
  * ``corner (v2.1.0)``

The first four packages are standard packages that you probably
already have installed on your computer. For these the version number
is probably not particularly important except for ``numpy``, which pre
v.1.15 will throw an error because of certain missing functions.

The following four packages can simply be installed using conda
e.g.,::

   conda install h5py
   conda install scikit-image
   conda install tqdm
   conda install -c conda-forge emcee

It should be noted that the top three packages are normally installed in the full installation of conda (v4.9.2). Alternatively, these programs can also be installed using pip. See the repsective websites for each program: `h5py <https://docs.h5py.org/en/stable/build.html>`_, `scikit image <https://scikit-image.org/>`_, `tqdm <https://github.com/tqdm/tqdm>`_,
`emcee <https://emcee.readthedocs.io/en/stable/>`_.
``scikit-image`` package is only needed to generate masks, and if you
want to make your own masks, it does not need to be installed. The
``tqdm`` package provides the progress bar used in the MCMC chain. 

.. warning::
   It is very important to use version 3.X or higher for
   ``emcee``, because version 2.X or earlier of this code will result
   in errors. Further details are given in the
   documentation for `emcee v3.0 <https://emcee.readthedocs.io/en/v3.0.0/user/upgrade/>`_.

Although the ``corner`` package is not required to run the fitting
routine, it is recommended for displaying the output of the fitting
routine, and is necessary to create the diagnostic plots. It can be installed using pip, see the documentation for `corner <https://corner.readthedocs.io/en/latest/>`_.

