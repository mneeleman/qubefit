.. _runfit:

Running the Fitting Routine
=======================================
This page will go through the steps needed to fit an existing model to a data cube. If you wish to define your own model you can have a look at :ref:`newmodel`. In short, to fit an existing  model to a data cube requires two steps. The first step is to setup a :ref:`qubefit` instance. This instance should include all of the information needed to run the fitting procedure, such as the data cube, the model, and priors of the parameters of the model. The second step is running the Markov Chain Monte Carlo (MCMC) routine. This step is computationally expensive, but requires little input from the user.


Initialize the qubefit instance
--------------------------------------
The first step is to initialize a :ref:`qubefit` instance and populate all of the information within this instance to successfully run the fitting routine. In theory this can be done line-by-line interactively, but in practice it is much easier to create a setup file to do this (see the :ref:`exfile`). The code requires the following keys to be defined (with a short description). This list is here provided for reference, and it is strongly advised to use a setup file, and the associated helper functions, to populate these keys as shown in the :ref:`exfile`.

  * **data**: the data cube from the observations
  * **shape**: the shape/dimensions of the data cube
  * **variance**: the variance of the data cube with the same dimensions as the data
  * **maskarray**: The mask used in the fitting procedure, which is a Boolean array with the same size as the data.
  * **kernel**: the point spread function or beam of the observations convolved with the line spread function of the instrument. Should be a three dimensional cube smaller than the data.
  * **kernelarea**: Float that contains the size of the point spread function or beam (used for normalization).
  * **probmethod**: String describing the method used to calculate the probability and likelihood function.
  * **modelname**: The model to fit. String name should be a function defined in qfmodels.py.
  * **intensityprofile**: List of three strings describing the intensity profile in the radial, azimuthal and axial direction. Profiles should be defined, within underscores, in qfmodels.py
  * **velocityprofile**: List of three strings describing the velocity profile in the radial, azimuthal and axial direction. Profiles should be defined, within underscores, in qfmodels.py
  * **dispersionprofile**: List of three strings describing the dispersion profile in the radial, azimuthal and axial direction. Profiles should be defined, within underscores, in qfmodels.py
  * **par**: Array with the numeric values needed for the model. These values are in intrinsic units.
  * **mcmcpar**: Subset of the **par** key for the parameters that are part of the MCMC routine, i.e., those parameters that are note held fixed.
  * **mcmcmap**: Map that denotes what position in the MCMC chain corresponds to what parameter.
  * **initpar**: dictionary that contains the initial parameter values in the given units, conversion to internal units, flag determining of the parameters should be kept fixed and the chosen prior (see the :ref:`exfile`).

Running the MCMC procedure
--------------------------------------
After the :ref:`qubefit` instance has been properly initialized with all of the required information, the fitting routine can be run. This is done with a single function call::

   sampler = QubeS.run_mcmc(nwalkers=100, nruns=1000, nproc=6)

The `run_mcmc` function calls ``emcee`` and has three keywords that can be set. The first two are the number of walkers (**nwalkers**) and the number of steps (**nsteps**). These numbers should be large enough such that the chains are converged. More details on this are given in the `emcee documentation on convergence <https://emcee.readthedocs.io/en/stable/tutorials/autocorr/#autocorr>`_. The last keyword sets the number of processes that will be opened by the code. Emcee allows for parallelization, **nproc** sets the numer of parallel processes that will be openend. This number should be smaller than the number of available computation cores on your system (preferably one or two less to allow for other system functions, or much less if you share the system with other users).

Although this single function can be easily called by itself, it might be useful to embed this into a small helper function that initiates the model, runs the chains, and saves the model. By default the chains will be saved to a numpy array and stored as part of the QubeFit instance, but one can store the *emcee* sampler information into other formats as well depending on your preferences.
