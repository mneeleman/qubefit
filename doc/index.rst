.. qubefit documentation master file, created by
   sphinx-quickstart on Mon Dec 28 13:40:01 2020.

Qubefit
===================================
Qubefit is a python-based code designed to fit models to astronomical observations that yield three dimensional 'data cubes'. These 'data cubes' consist out of two spatial directions and one spectral, and are the end products of both radio / (sub-)millimeter interferometry observations, as well as observations with integral field units in the optical and near-infrared. Qubefit was specifically designed to model the emission from high redshift galaxies, for which the emission is only barely resolved and the resolution of the observations, which is known as the point spread function (PSF) or the beam, substantially affects the emission (often called beam-smearing).

.. image:: ./Fig/QubeFitLogo.png
   :align: center
   :height: 200pt

Currently there are several packages available that can model the emission from high redshift **disk** galaxies. However, for some high redshift galaxies this might not be a good assumption. Some galaxies might not show any sign of rotation, while other galaxies are actually multiple merging clumps. Modeling these galaxies with disks could bias observations by erroneously fitting disks to objects that might not be disks at all. Qubefit has several non-disk like models, and other user-defined models can easily be added into the code. This can help assess if galaxy emission truly arises from disks or if other configurations can also reproduce the observed emission.

Qubefit uses a fully Bayesian framework to explore the multi-dimension parameter space and find the best fit parameters and uncertainties. It does this by applying the affine invariant Markov Chain Monte Carlo (MCMC) ensemble sampler via the `emcee <https://emcee.readthedocs.io/en/stable/>`_ package. Although this fully Bayesian approach comes with a price in computational speed, it is relatively robust to the choice of priors and initial guesses.

Within the code, there are several convenience functions defined to assess the results of the fitting routine, as well as to assess the kinematics of the data, such as moment images and positon-velocity diagrams. Tutorials on how to fit a model to a data cube, as well as how to use the kinematics functions are given in the tutorial section in the menu. These are good places to start after you have installed qubefit. 
	    
Citing the code
=======================================
If you use this code in your work, please cite the following paper: ``link to paper``. Here is the bibtex reference for this work::

  Put bibtex reference here

License
=======================================
Qubefit is free software made available under the MIT License, and comes with no guarantee whatsover. For details see the ``LICENSE``.

Bugs and Suggestions
=======================================
If you spot any bugs (there are plenty), have suggestions or comments, and/or wish to contribute to the code, please let me know at neeleman-at-mpia.de.

Documentation
=======================================

.. toctree::
   :maxdepth: 1
   :caption: User Manual:

   Manual/install
   Manual/runfit
   Manual/exfile
   Manual/gui
   Manual/models
   Manual/newmodel

.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   Tutorials/ThinDisk.ipynb
   Tutorials/Moments.ipynb

.. toctree::
   :maxdepth: 1
   :caption: API:

   API/qube
   API/qubefit
   API/qfmodels
