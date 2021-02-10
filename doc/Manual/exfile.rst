.. _exfile:

Example Setup File
===============================================
To successfully run the fitting routine, you have to initialize a :ref:`qubefit` instance and populate all of the information within this instance. In theory this can be done line-by-line interactively, but in practice it is much easier to create a setup file to do this. The setup file we use is the ThinDiskSetup.py ``(link this)``. Here we will go line-by-line to explain each of the lines in this setup file. It is important to note that some of these lines of code might not be needed for your data or model, or you might need to add some additional information for the code to run successful. However, this example file is probably a good starting point for the setup file for your project.

Structure
-------------------
.. code-block::

   import numpy as np
   import astropy.units as u
   from qubefit.qubefit import QubeFit

   def set_model():
       ...
   return QubeS

The first thing to note is the overall structure of the setup file. The setup is done within a function named *set_model()* . This structure and naming convention is important **if** you wish to use the  :ref:`guis` to look at the source, because the GUIs will explicitly look for this function name within the setup file. The function returns `QubeS`, which is a :ref:`qubefit` instance that contains all of the needed information to run the fitting routine.

Loading the Data
-----------------------------------
.. code-block::

   DataFile = './examples/WolfeDiskCube.fits'
   Qube = QubeFit.from_fits(DataFile)
   Qube.file = DataFile

In these lines the `data` and `header` keys are set using the task `from_fits` ``(link this)``. The final line sets the `file` key in the :ref:`qubefit` instance. This line is only used to assign a title to the GUI window.

.. note::
   Depending on the header in the fits file, the rest frequency of the line might need to be (re)set here, i.e., Qube.header['restfrq'] = freq_in_Hz, where freq_in_Hz is the value of the redshifted frequency of the line.

Trimming the Data Cube
------------------------------------------------------
.. code-block::

   center, sz, chan = [507, 513], [45, 45], [15, 40]
   xindex = (center[0] - sz[0], center[0] + sz[0] + 1)
   yindex = (center[1] - sz[1], center[1] + sz[1] + 1)
   zindex = (chan[0], chan[1])
   QubeS = Qube.get_slice(xindex=xindex, yindex=yindex, zindex=zindex)

It is crucial to trim the data cube to the smallest possible region, because the largest time sink is convolving the model with the beam/PSF. The convolution time scales roughly linearly with the size of the cube, so large data cubes will unnecessarily slow down the code. In this example, the new trimmed cube will be centered at pixel position (507, 513), have a physical size of 90 by 90 pixels (45 pixels on each side of the center), and covers the wavelength (frequency) channels between 15 and 39 inclusive.

Note that the `get_slice` routine copies (technically it is a deep copy) the structure of the :ref:`qubefit` instance and anything that has been previously added, so QubeS is a full instance of :ref:`qubefit`. This routine will also update the `header` and `shape` keys to reflect the new size of the data.

Calculating the Variance
-------------------------------------------
.. code-block::

   QSig = Qube.calculate_sigma()
   QSig = np.tile(QSig[chan[0]: chan[1], np.newaxis, np.newaxis],
                  (1, 2 * sz[1] + 1, 2 * sz[0] + 1))
   QubeS.variance = np.square(QSig)

These lines will load the `variance` key into the :ref:`qubefit` instance. For interferometry data, such as ALMA, this can be calculated from the data cube by fitting a Gaussian to data that contains no signal. For optical observations, the uncertainties or variances per pixel are often stored as a separate fits file, which can be loaded in directly with the `from_fits` routine. The `variance` key requires a simple numpy array with the same dimensions as the `data` key. In our case the output of the routine `calculate_sigma` is a single uncertainty per channel (wavelength slice), so we have to tile the data into a full array with the same size as the trimmed data cube. Finally we take the square (to get the variance) and load this array into the `variance` key.

Note that we need to use the full data cube to calculate the uncertainties, not the trimmed one. This is because the trimmed data cube does not have enough pixels without signal to accurately estimate the uncertainty of each channel.

Defining the Kernel
----------------------------------------------------
.. code-block::
   
  QubeS.create_gaussiankernel(channels=[0], LSFSigma=0.1)

This line of code populates the `kernel` key of the :ref:`qubefit` instance. The kernel is the shape of the beam (or point spread function, PSF). For most cases, the beam or PSF can be approximated by a Gaussian, and the above code will generate such a kernel from the beam parameters defined by the fits header.

.. note::
   For optical data the "beam" parameters will need to be updated to the values corresponding to the spatial resolution or seeing of the observations. This can be done by setting the corresponding values in the header: QubeS.header['BMAJ'] = psf_in_degrees and QubeS.header['BMIN'] = psf_in_degrees, where psf_in_degrees is the size of the point spread function (i.e., the seeing) in degrees.

After creating a 2D kernel, the kernel is convolved with the line spread function of the instrument. For ALMA observations, the line spread function is often negligible, because the channels are Hanning smoothed to much coarser resolution. Setting the width of the line spread function to something small, like 0.1 times the channel width, will make a correct 3D kernel.

Setting the Mask
------------------------------------------------------
.. code-block::

   QubeS.create_maskarray(sigma=3, fmaskgrow=0.01)

This code populates the `mask` key in the QubeS object. This mask is a simple array of ones and zeros that has the same size as the data cube. A one means to include a pixel in the fitting procedure, whereas a zero means to not include the pixel. The mask array is stored in QubeS.mask, so if you want to use your own custom mask, you would need to set this keyword simply as: QubeS.mask = your_mask_array, where your_mask_array is an array of equal size as the data cube and filled in with ones and zeros.

Defining the Model
-------------------------------------------------------
.. code-block::
   
  QubeS.modelname = 'ThinDisk'
  QubeS.intensityprofile[0] = 'Exponential'
  QubeS.velocityprofile[0] = 'Constant'
  QubeS.dispersionprofile[0] = 'Constant'

Here we are defining which model to use. Note that the models that come with qubefit package are described on the page :ref:`models` . You could also think about :ref:`newmodel`, which is part of the strength of the qubefit package.

In this example we will use the ThinDisk model. Within the ThinDisk model, we can also set several profiles for the intensity, velocity and the dispersion. In this case we assume a simple exponential profile for the emission and both a constant velocity and dispersion profile across the disk. Options available are described in detail in :ref:`models`. 

Parameters and Priors
-------------------------------------------------------
.. code-block::
   
   PDict = {'Xcen': {'Value': 45.0, 'Unit': u.pix, 'Fixed': False,
                     'Conversion': None,
                     'Dist': 'uniform', 'Dloc': 35, 'Dscale': 20},
                     ..}
   QubeS.load_initialparameters(PDict)

The next thing to load into the :ref:`qubefit` instance are the parameters for the model. The parameters are stored in a nested dictionary. For each parameter in the dictionary, 7 keys need to be defined that will determine the initial value of the parameter and its prior. The function `load_initialparameters` populates several keys in the :ref:`qubefit` instance, namely `initpar`, `par`, `mcmcpar`, `mcmcmap` and `mcmcdim`. Although these could be set individually, it is **highly** recommended to use the load_initialparameters keyword, to make sure the mapping gets done correctly. The structure of the dictionary for each parameter is as follows:

  * **Value**: The initial value of the parameter in whatever unit you specify.
  * **Unit**: The unit of the parameter. Any unit can be used as long as you apply the correct conversion to the native units of the cube with the **Conversion** key.
  * **Fixed**: if sets to True, the code will keep this parameter fixed during the fitting routine.  
  * **Conversion**: This will convert the **Value** parameter into the native units of the data cube. For instance, the velocities are often wanted in units of km/s, but the native units of the cube are pixels (in the spectral direction). Note that this conversion can also be used to convert degrees into radians (for angles).
  * **Dist**: This is the prior distribution for the parameter. The valid distributions here are those defined in the `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ module. A large list of valid priors are allowed, but beware that not all will give reasonable results. One should be very careful selecting these distributions, and when in doubt take the least constraining, which is often an uniformed (here called `uniform`) prior.
  * **Dloc**: This sets the location of the distribution and is equal to the `loc` parameter in the functions defined in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. Look at these pages to see what this parameter means for the distribution that you have chosen. For example, in the `uniformed` prior this corresponds to the lower bound of the acceptable range.
  * **Dscale**: This sets the scale of the distribution and is equal to the `scale` parameter in the functions defined in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_. Look at these pages to see what this parameter means for the distribution that you have chosen. For example, in the `uniformed` prior this corresponds to the range of the acceptable values starting at **Dloc**. Therefore in our example, all `Xcen` values between 35 and 55 pixels have equal probability, and outside this range the probability is zero.


Making the Model
-------------------------------------
.. code-block::

  QubeS.create_model()

We are now finally in a postion to create a model cube. This is done with the above command. The model will be stored `model` key, and it will have the same dimension as the data cube. The model cube has also been convolved with the kernel (Beam/PSF). If you want to make a model that is not convolved with the kernel, you can set the keyword convolve to False, i.e., `QubeS.create_model(convolve=False)`.
