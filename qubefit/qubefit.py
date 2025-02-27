"""
Main class for the qubefit fitting program.

This class is based on the qube class and inherits all of its functions. The
additional functions defined here are to define a model, run the fitting
routine and analyze the results.
"""
import numpy as np
from scipy.special import erf
import emcee
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from scipy.stats import norm, uniform, kstest
from skimage import measure
import os
if os.getcwd() == '/Users/mneelema/Packages/qubefit/qubefit':
    from qube import Qube
    from qfmodels import *
    from qfutils import make_cornerfigure, make_mcmcchainfigure, make_2dmodelfigure
else:
    from qubefit.qube import Qube
    from qubefit.qfmodels import *
    from qubefit.qfutils import make_cornerfigure, make_mcmcchainfigure, make_2dmodelfigure
from multiprocessing import Pool
import h5py
from astropy.modeling.models import Gaussian2D, Sersic2D
import json
from astropy.modeling import Fittable2DModel, Parameter
from scipy.special import gammaincinv


class QubeFit(Qube):
    """
    Initiate the Qubefit class.

    Class that is used for fitting data cubes. This class is based on the
    Qube class and inherits all of its functions. The additional functions
    defined here are to define a model, run the fitting routine and analyze
    the results.

    When it is initiated, the following data attributes are defined and
    can be explcitely set: modelname, probmethod, intensityprofile,
    velocity profile, and dispersionprofile.

    Parameters
    ----------
        modelname : STRING, optional
            The name of the model used for the fitting. The default is
            'ThinDisk'.
        probmethod : STRING, optional
            The name of the probability function to use. The default is
            'ChiSq'.
        intensityprofile : LIST, optional
            List of three strings with the names of the profiles to use for
            the intesity of the model. The three strings correspond to the
            three dimensions used to describe the model. For a cylindrical
            coordinate system this is r, phi and z in that order.
            The default is ['Exponential', None, 'Exponential'].
        velocityprofile : LIST, optional
            List of three strings with the names of the profiles to use for
            the velocity of the model. The three strings correspond to the
            three dimensions used to describe the model. For a cylindrical
            coordinate system this is r, phi and z in that order.
            The default is ['Constant', None, None].
        dispersionprofile : TYPE, optional
            List of three strings with the names of the profiles to use for
            the disperdsion of the model. The three strings correspond to the
            three dimensions used to describe the model. For a cylindrical
            coordinate system this is r, phi and z in that order.
            The default is ['Constant', None, None].
    """
    def __init__(self, modelname='ThinDisk', probmethod='ChiSq', intensityprofile=None, velocityprofile=None,
                 dispersionprofile=None):
        """Initiate the qubefit class."""
        super().__init__()
        self.modelname = modelname
        if intensityprofile is None:
            self.intensityprofile = ['Exponential', None, 'Exponential']
        else:
            self.intensityprofile = intensityprofile
        if velocityprofile is None:
            self.velocityprofile = ['Constant', None, None]
        else:
            self.velocityprofile = velocityprofile
        if dispersionprofile is None:
            self.dispersionprofile = ['Constant', None, None]
        else:
            self.dispersionprofile = dispersionprofile
        self.probmethod = probmethod
        self.kernel = []
        self.kernelarea = 0.
        self.initpar = None
        self.par = {}
        self.mcmcpar = []
        self.mcmcmap = []
        self.priordist = []
        self.mcmcdim = 0
        self.model = np.array([])
        self.residual = np.array([])
        self.variance = np.array([])
        self.mcmcarray = None
        self.mcmclnprob = None
        self.chainpar = None
        self.maskarray = []

    def create_gaussiankernel(self, channels=None, lsf_sigma=None, kernelsize=4):
        """
        Create a Gaussian kernel.

        This method will generate a Gaussian kernel from the data
        stored in the Qube. It will look for the beam keyword and generate
        a gaussian kernel from the shape of the synthesized beam given here.
        Several different kernels can be returned.
        1) A list of 3D kernels -one for each spectral channel- that
        takes into account both a varying LSF and a varying PSF
        (set this by selecting a range of channels and LSFSigma)
        2) A list of 2D kernels -one for each spectral bin- that
        takes into account a varying PSF, but no LSF
        (set this by selecting a range of channels and LSFSigma=None)
        3) A 3D kernel that has constant PSF and LSF
        Only the last option is currently implemented in the code and should
        therefore be used (i.e., specify a single channel not a list).

        This method will populate the kernel and kernelarea data
        attributes of the qubefit instance.

        Parameters
        ----------
        channels : LIST, NUMPY.ARRAY or INT, optional
            The channels to use to create the beam. If this is not specified,
            then the center of the cube will be used. If multiple channels
            are specified, then a list of kernels will be given, one for
            each channel. This is useful when the kernel/PSF changes with the
            channel. This option, however, is currently NOT implemented in
            the code and therefore just a single channel should be specified.
            The default is None.
        lsf_sigma : FLOAT or NUMPY.ARRAY, optional
            The width of the line spread function. Note that this is the
            sigma or square root of the variace, NOT the FWHM!. The unit for
            this value is pixels. If this is not specfied, then the LSF is
            assumed to be neglible, which is often ok for ALMA data, which has
            been averaged over many channels.
            The default is None.
        kernelsize : FLOAT, optional
            The size of the kernel in terms of the sigma of the major axis
            in pixels (bsig). The actual size of the kernel will be a cube
            that has a size for the two spatial dimensions: 2 (n * bsig) + 1,
            where n is the kernelsize. The default is 4.

        Raises
        ------
        AttributeError
            Function will raise an attribute error if the beam has not been
            propertly defined.

        Returns
        -------
        None : NoneType
        """
        # Check that the beam attribute has been defined
        if not hasattr(self, 'beam'):
            raise AttributeError('Beam attribute must be defined to create' +
                                 'gaussian kernel')
        # define some parameters for the beam
        bmaj = self.beam['BMAJ'] / np.sqrt(8 * np.log(2)) / np.abs(self.header['CDELT1'])
        bmin = self.beam['BMIN'] / np.sqrt(8 * np.log(2)) / np.abs(self.header['CDELT1'])
        bpa = self.beam['BPA']
        theta = np.pi / 2. + np.radians(bpa)
        kernel_area = bmaj * bmin * 2 * np.pi
        # Here decide which channel(s) to use for generating the kernel
        try:
            len(bmaj)
        except TypeError:
            bmaj, bmin, bpa, theta, kernel_area = [bmaj], [bmin], [bpa], [theta], [kernel_area]
        if channels is None:
            channels = [len(bmaj) // 2]
        if type(channels) is int:
            channels = [channels]
        # create an array of LSFSigma (if needed)
        if lsf_sigma is not None and type(lsf_sigma) is float:
            lsf_sigma = np.full(len(bmaj), lsf_sigma)
        # create a (list of) 2D kernel(s)
        kernel = ()
        for ii in channels:
            xsize = 2 * np.ceil(kernelsize*bmaj[ii]) + 1
            ysize = 2 * np.ceil(kernelsize*bmaj[ii]) + 1
            twod_kernel = Gaussian2DKernel(bmaj[ii], bmin[ii], theta=theta[ii], x_size=xsize, y_size=ysize).array
            # apply the line-spread function (if wanted)
            if lsf_sigma is None:
                kernel = kernel + (twod_kernel, )
            else:
                lsf_kernel = Gaussian1DKernel(lsf_sigma[ii]).array
                temp_kernel = np.zeros(lsf_kernel.shape + twod_kernel.shape)
                temp_kernel[lsf_kernel.shape[0] // 2, :, :] = twod_kernel
                threed_kernel = convolve(temp_kernel, lsf_kernel[np.newaxis, np.newaxis, ...])
                kernel = kernel + (threed_kernel, )
        # select the kernel areas
        kernel_area = [kernel_area[x] for x in channels]
        # if a single channel is given then remove the list and force the
        # single kernel to be 3D.
        if len(kernel) == 1:
            kernel = kernel[0]
            kernel_area = kernel_area[0]
            if kernel.ndim == 2:
                kernel = np.array([kernel, ])
        # now assign the following attributes.
        self.kernel = kernel
        self.kernelarea = kernel_area

    def load_initialparameters(self, parameters):
        """
        Load the intial parameter dictionary into the qubefit instance.

        This method will load in the initial parameters from a
        dictionary. The dictionary is transfered into a data attribute,
        in addition several other attributes are created which are needed
        for the model and the MCMC fitting procedure.

        This method will set the initpar, par, mcmcpar, mcmcmap,
        priordist and mcmcdim data attributes. The initpar attribute is a
        simple copy of the parameter dictionary. The par data attribute
        contains the value of the parameters converted into intrinsic units.
        mcmcpar and mcmcmap are the values and names of the parameters not
        held fixed during the fitting procedure and mcmcdim are the number of
        free parameters. Finally priordist is the prior distribution of each
        parameter that is not held fixed (see scipy.stats).

        Parameters
        ----------
        parameters : DICT
            The dictionary describes several important features for each
            parameter of the model. A detailed description is given in the
            online documentation. In general, it contains a 'Value' and 'Unit',
            key, which are in physically interesting units.
            A 'Conversion' to intrinsic units of the cube. A 'Fixed' key to
            allow a parameter to be kept fixed, and finally a 'Dist' keyword,
            which describes the priors of the parameters (using the keys:
            'Dloc and 'Dscale').

        Returns
        -------
        None : NoneType

        """
        self.initpar = parameters
        for key in self.initpar.keys():

            if self.initpar[key]['Conversion'] is not None:
                self.par[key] = (self.initpar[key]['Value'] *
                                 self.initpar[key]['Unit'] /
                                 self.initpar[key]['Conversion']).value
            else:
                self.par[key] = self.initpar[key]['Value']

            if not self.initpar[key]['Fixed']:
                self.mcmcpar.append(self.par[key])
                self.mcmcmap.append(key)
                self.priordist.append(eval(self.initpar[key]['Dist'])
                                      (loc=self.initpar[key]['Dloc'],
                                      scale=self.initpar[key]['Dscale']))
        # store the number of free variables of the mcmc process
        self.mcmcdim = len(self.mcmcpar)

    def create_model(self, do_convolve=True):
        """
        Create the model cube with the given parameters and model.

        This method will take the stored parameters (in self.par) and
        will generate the requested model. NOTE: currently no checking is
        done that the parameters are actually defined for the model that is
        requested. This will likely result in a rather benign AttributeError.

        This method will set the model data attribute using the parameters
        stored in the par data attribute and the model defined elsewhere.

        Parameters
        ----------
        do_convolve : BOOLEAN, optional
            If set, the model will be convolved with the beam.
            The default is True.

        Returns
        -------
        None : NoneType
        """
        kwargs = self.__define_kwargs__()
        kwargs['convolve'] = do_convolve
        self.model = __create_model__(**kwargs)
        if 'data' in self.__define_kwargs__().keys():
            self.residual = self.data - self.model

    def run_mcmc(self, nwalkers=50, nsteps=100, nproc=None, init_frac=0.02,
                 filename=None, return_sampler=False):
        """
        Run the MCMC process with emcee.

        This method is the heart of QubeFit. It will run an MCMC process on a
        predefined model and will return the resulting chain of the posterior
        PDFs of each individual parameter that was varied. It saves the
        results in the mcmcarray and mcmclnprob data attributes, but it is
        HIGHLY RECOMMENDED to also save the outputs into a HDF5 file.

        Parameters
        ----------
        nwalkers : INTEGER, optional
            The number of walkers to use in the MCMC chain. The default is 50.
        nsteps : INTEGER, optional
            The number of steps to make in the MCMC chain. The default is 100.
        nproc : INTEGER, optional
            The number of parallel processes to use. If set to None, the code
            will determine the optimum number of processes to spawn. Set this
            number to limit to load of this code on your system.
            The default is None.
        init_frac: FLOAT, optional
            The fraction to use to randomize the initial distribution of
            walker away from the chosen initial value. The default is 0.02.
        filename : STRING, optional
            If set, the chain will be saved into a file with this file name.
            The file format is HDF5, and if not directly specified, this
            extension will be appended to the filename. The default is None.
        return_sampler : BOOLEAN, optional
            If set, this will return the emcee ensemble sampler.
            The default is False.

        Raises
        ------
        ValueError
            A ValueError will be raised if the initial probability is
            infinity. This likely is an indication that the initial
            parameters fall outside the range defined by the priors.

        Yields
        ------
        sampler : EMCEE.ENSEMBLESAMPLER
            The ensemble sampler returned by the emcee function call can be
            returned, if wanted.
        """
        # load the hdf5 backend if filename is specified
        if filename is not None:
            if filename[-5:] != '.hdf5':
                filename = filename + '.hdf5'
            backend = emcee.backends.HDFBackend(filename)
        else:
            backend = None

        # intiate the model (redo if already done)
        self.create_model()

        # create the bootstrap array (do not redo if already done)
        if not hasattr(self, 'maskarray'):
            self.create_maskarray()

        # define the keyword arguments for the model fitting function
        kwargs = self.__define_kwargs__()

        # calculate the intial probability, if it is too small (i.e., 0) then
        # the code will exit with an error
        initprob = __lnprob__(self.mcmcpar, **kwargs)
        if np.isinf(initprob):
            raise ValueError('Initial parameters yield zero probability.' +
                             ' Please choose other initial parameters.')
        print('Intial probability is: {}'.format(initprob))

        # define the sampler
        os.environ["OMP_NUM_THREADS"] = "1"
        with Pool(nproc) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, self.mcmcdim, __lnprob__, pool=pool, backend=backend,
                                            kwargs=kwargs)

            # initiate the walkers
            p0 = [(1 + init_frac * np.random.rand(self.mcmcdim)) * self.mcmcpar for _walker in range(nwalkers)]

            # run the mcmc chain
            sampler.run_mcmc(p0, nsteps, progress=True)

        # Now store the results into the structure (transpose is needed to agree with hdf5 structure)
        self.mcmcarray = np.transpose(sampler.chain, axes=(1, 0, 2))
        self.mcmclnprob = np.transpose(sampler.lnprobability, axes=(1, 0))

        # return the sampler (for testing of emcee)
        if return_sampler:
            return sampler

    def get_chainresults(self, filename=None, burnin=0.3, reload_model=True,
                         load_best=False):
        """
        Get the results from the MCMC run either from file or memory.

        Calling this method will generate a dictionary with the median
        values, and 1, 2, 3  sigma ranges of the parameters. These have been
        converted into their original units using the conversions in the
        the initial parameter attribute (initpar).

        If not present, this method will populate the self.mcmcarray and
        self.mcmclnprob attributes with the values in the file. It will
        also populate the self.chainpar data attribute with a dictionary
        with the median parameters, uncertainty and unit conversions.

        Parameters
        ----------
        filename : STRING, optional
            The name of the HDF5 file to load. If the filename is set to None,
            the assumption is that the qubefit instance already has the
            MCMC chain and log probability loaded into their respective
            instances. If this is not the case, the code will exit with an
            AttributeError. The default is None.
        burnin : FLOAT, optional
            The fraction of runs to discard at the start of the MCMC chain.
            This is necessary as the chain might not have converged in the
            very beginning. One should check the chain to make sure that this
            value is correct. The default is 0.3.
        reload_model : BOOLEAN, optional
            reload the model cube with the updated parameters. The default
            parameters to use are the median values of each parameter.
            The default is True.
        load_best : BOOLEAN, optional
            If set, then instead of the median parameters, the combination
            of parameters that yielded the highest probability will be
            chosen. The default is False.

        Raises
        ------
        AttributeError
            An attribute error will be raised if the needed mcmcarray and
            lnprobability keys are not set in the qubefit instance and no
            filename is given.

        Returns
        -------
        None : NoneType
        """
        # if filename is given, then load the HDF5 file.
        if filename is not None:
            file = h5py.File(filename, 'r')
            self.mcmcarray = file['mcmc']['chain'][()]
            self.mcmclnprob = file['mcmc']['log_prob'][()]
        else:
            if self.mcmcarray is None or self.mcmclnprob is None:
                raise AttributeError('get_chainresults: mcmcarray and/or mcmclnprob is not defined')

        # get the burnin value below which to ditch the data
        burninvalue = int(np.ceil(self.mcmcarray.shape[0] * burnin))
        self.mcmcarray = self.mcmcarray[burninvalue:, :, :]
        self.mcmclnprob = self.mcmclnprob[burninvalue:, :]

        # sigma ranges
        perc = (1 / 2 + 1 / 2 * erf(np.arange(-3, 4, 1) / np.sqrt(2))) * 100

        # find the index with the highest probability
        best_val_idx = np.unravel_index(self.mcmclnprob.argmax(),
                                        self.mcmclnprob.shape)

        # create the output parameters from the initial parameter conversions.
        par, med_arr, best_arr = {}, {}, {}
        for key in self.initpar.keys():

            # get the intrinsic units of the parameters
            if self.initpar[key]['Conversion'] is not None:
                unit = self.initpar[key]['Unit'] / self.initpar[key]['Conversion'].unit
            else:
                unit = self.initpar[key]['Unit']

            # now calculate median, etc. only if it was not held fixed
            if not self.initpar[key]['Fixed']:
                idx = self.mcmcmap.index(key)
                values = np.percentile(self.mcmcarray[:, :, idx], perc) * unit
                med_arr.update({key: values[3].value})
                best_value = self.mcmcarray[best_val_idx[0], best_val_idx[1], idx] * unit
                best_arr.update({key: best_value.value})
                if self.initpar[key]['Conversion'] is not None:
                    values = values * self.initpar[key]['Conversion']
                    best_value = best_value * self.initpar[key]['Conversion']
            else:
                values = (np.array([0, 0, 0, self.initpar[key]['Value'],
                                    0, 0, 0]) *
                          self.initpar[key]['Unit'])
                best_value = (self.initpar[key]['Value'] *
                              self.initpar[key]['Unit'])

            par.update({key: {'Median': values[3].value,
                              '1Sigma': [values[2].value, values[4].value],
                              '2Sigma': [values[1].value, values[5].value],
                              '3Sigma': [values[0].value, values[6].value],
                              'Best': best_value.value, 'Unit': values[0].unit,
                              'Conversion': self.initpar[key]['Conversion']}})
        par.update({'MedianArray': med_arr, 'BestArray': best_arr})
        self.chainpar = par

        # reload the model with the median or best array
        if reload_model:
            if load_best:
                self.update_parameters(best_arr)
            else:
                self.update_parameters(med_arr)
            self.create_model()

    def update_parameters(self, parameters):
        """
        Update the parameters key with the given parameters.

        This method is a simple wrapper to update one or multiple parameters
        in the par data attribute. These parameters should be given in
        intrinsic units.

        Parameters
        ----------
        parameters : DICT
            dictionary of parameters that will be read into the par keyword.

        Returns
        -------
        None : NoneType
        """
        for key in parameters.keys():
            self.par[key] = parameters[key]

    def calculate_chisquared(self, reduced=True, adjust_for_kernel=True):
        """
        Calcuate the (reduced) chi-squared statistic.

        This method will calculate the (reduced) chi-squared statistic for
        the given model, data and pre-defined mask. The value can be adjusted
        for the oversampling of the beam using a simplified assumption, which
        is described in detail in the reference paper.

        Parameters
        ----------
        reduced : BOOLEAN, optional
            If set, this will calculate the reduced chi-square statistic, by
            dividing by the degrees of freedom and optionally by the
            adjustment factor to account for the oversampled beam.
            The default is True.
        adjust_for_kernel : BOOLEAN, optional
            If set, it will apply an adjustement factor to the reduced
            chi-suared statistic to account for oversampling the beam. If the
            mask is only sparsely sampled, this adjustment factor should be
            set to False. The default is True.

        Raises
        ------
        AttributeError
            This method will return an AttributeError if no mask has yet been
            defined.

        Returns
        -------
        chisq : FLOAT
            The (reduced) chi-square statistic of the model within the mask.
        """
        # check to see if a mask is defined
        if not hasattr(self, 'maskarray'):
            raise AttributeError('Need to define a mask for calculating the likelihood function')
        # get the residual and variance
        residual = self.residual
        variance = self.variance
        residualsample = residual[np.where(self.maskarray)]
        variancesample = variance[np.where(self.maskarray)]
        # calculate the chi-squared value
        chisq = np.sum(np.square(residualsample) / variancesample)
        if adjust_for_kernel:
            chisq = chisq / (self.kernelarea * np.log(2) / np.pi)
        if reduced:
            if adjust_for_kernel:
                indep_samples = len(residualsample) / (self.kernelarea * np.log(2) / np.pi)
            else:
                indep_samples = len(residualsample)
            dof = indep_samples - len(self.mcmcpar)
            chisq = chisq / dof
        return chisq

    def calculate_ksprobability(self):
        """
        Calculate the Kolmogorov-Smirnov probability of the model
        """
        # check to see if a mask is defined
        if not hasattr(self, 'maskarray'):
            raise AttributeError('Need to define a mask for calculating the likelihood function')
        # get the residual and variance
        residual = self.residual
        residualsample = residual[np.where(self.maskarray)]
        # calculate the ks probability using kstest of scipy.stats
        scale = np.median(np.sqrt(self.variance))
        sigmadist_theo = norm(scale=scale)
        residualsample = residualsample - np.mean(residualsample)
        prob = kstest(residualsample, sigmadist_theo.cdf)[1]
        return prob

    def create_maskarray(self, sampling=2., bootstrapsamples=200,
                         regular=None, sigma=None, nblobs=1, fmaskgrow=0.01):
        """
        Generate the mask for fitting.

        This will generate the mask array. It should only be done
        once so that the mask does not change between runs of the MCMC
        chain, which could/will result in slightly different probabilities
        for the same parameters. The mask returned is either the same size
        as the data cube (filled with np.NaN and 1's for data to
        ignore/include) or an array with size self.data.size by
        bootstrapsamples.
        The first is used for directly calculating the KS
        or ChiSquared value of the given data point, while the second is used
        for the bootstrap method of both functions.

        Several mask options can be specified. It should be noted that these
        masks are muliplicative. This method will set the maskarrray data
        attribute.

        Parameters
        ----------
        sampling : FLOAT, optional
            The number of samples to choose per beam/kernel area. This keyword
            is only needed for the sparse sampling methods and for bootstrap
            arrays. The default is 2.

        bootstrapsamples : INT, optional
            The number of bootstrap samples to generate. This keyword is only
            needed if the probability method is set to 'BootKS' or
            'BootChiSq'. The default is 200.

        regular : TUPLE, optional
            If a two element tuple is given, then a regular grid will be
            generated that is specified by the number of samplings per
            kernelarea which will go through the given tuple. The default is
            None.

        sigma : FLOAT, optional
            If given, a mask is created that will just consider values above
            the specified sigma (based on the calculated variance). Only the
            n largest blobs will be considered (specified optionally by
            nblob). It will grow this mask by convolving with the kernel
            to include also adjacent 'zero' values.

        nblobs : INT, optional
            The number of blobs to consider in the sigma-mask method. The
            blobs are ordered in size, where the largest blob which is the
            background is ignored. If set to 1, this will select only the
            largest non-background blob. The default is 1.

        fmaskgrow : FLOAT, optional
            The fraction of the peak value which marks the limit of where to
            cut of the growth factor. If set to 1, the mask is not grown.
            This value allows to include some adjacent zero values to be
            included.

        Returns
        -------
        None : NoneType
        """
        method = self.probmethod
        # bootstrap array
        if method == 'BootKS' or method == 'BootChiSq':
            # create the mask array
            size = (int(sampling * self.data.size / self.kernelarea), bootstrapsamples)
            self.maskarray = np.random.randint(self.data.size, size=size)
        # regular array
        elif method == 'KS' or method == 'ChiSq' or method == 'RedChiSq':
            # masks are cumulative / multiplicative
            self.maskarray = np.ones_like(self.data)
            if regular is not None:
                self.maskarray *= __get_regulargrid__(self.data.shape, self.kernelarea, sampling, regular)
            if sigma is not None:
                self.maskarray *= __get_sigmamask__(self.data, self.variance, self.kernel, sigma, nblobs, fmaskgrow)
        # not defined method
        else:
            raise ValueError('qubefit: Probability method is not defined: {}'.format(method))

    def fit_2dmodel(self, initpar, modelname='gaussian2d', mcmcfilename=None, nwalkers=50, nsteps=100, sampling=2,
                    regular=(), rms=None, sigma=2, nblobs=1, nproc=7, plotroot='./Test', outfile='./Test_fit.json'):
        """
        Fits a 2d model to the data.

        Will work on a single image or a single channel for a cube.
        First a check will be done if the data shape is 2d, if not
        an error will be raised.
        The fit will take into account the beam smearing of the data.
        The model that it will take is a simple combination of 2D models defined
        in the module astropy.modeling.models. Currently models that have
        been implemented are Gaussian2D and Sersic2D. Each of these
        models take their own parameters (no check is done that the
        correct parameters are defined).
        The approach is a simple minimzation of the 2D model using the
        MCMC approach that is similar the the main qubefit function.

        Parameters
        ----------
        initpar : DICT
            dictionary with the initial guesses, priors and 'fixed'
            booleans for the parameters of the model. An example of a single
            component dictionary for the Gaussian2D is -
            {'x': {'Value': 128, 'Fixed': False, 'Conversion': None, 'Unit': u.pix,
                   'Dist': 'uniform', 'Dloc': 123, 'Dscale': 10},
             'y': {'Value': 128, 'Fixed': False, 'Conversion': None, 'Unit': u.pix,
                   'Dist': 'uniform', 'Dloc': 123, 'Dscale': 10},
             'A': {'Value': 3E-3, 'Fixed': False, 'Conversion': None, 'Unit': u.Jy,
                   'Dist': 'uniform', 'Dloc': 0, 'Dscale': 1E-1},
             'sig_x': {'Value': 8, 'Fixed': False, 'Conversion': None, 'Unit': u.pix,
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 50},
             'sig_y': {'Value': 4, 'Fixed': False, 'Conversion': None, 'Unit': u.pix,
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 50},
             'theta': {'Value': 90, 'Fixed': False,
                       'Conversion': (180 * u.deg) / (np.pi * u.rad), 'Unit': u.deg,
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 180}}
        modelname: STRING
            Name of the 2D model to use. Currently implmented are: 'gaussian2d', 'sersic2d'
            or 'coresersic2d'. The latter is a sersic profile with a core defined by
            a 2DSersic profile with the same X and Y position. Parameters are named _1 for
            the main components and _2 for the core components.
            The default is 'gaussian2d'
        mcmcfilename: STRING
            Name of the output file that contains the MCMC string. If set
            to None, then no MCMC output file is generated and only the
            final output file with the parameters is written out using the
            keyword outfile.
        nwalkers: INT
            Number of walkers in the emcee process
        nsteps: INT
            Number of steps in the emcee process
        sampling: INT
            Number of approximate measurments per beam.
            The default value is 2.
        regular: TUPLE
            Tuple that specified the center of the regular grid.
            The default is the center of the data array.
        rms: FLOAT
            Noise rms of the image. If set to none, the rms will be calculated
            from the input image.
        sigma: FLOAT
            The value (in rms of the data cube) to use as the cutoff
            for the mask. The mask is grown slightly to include surrounding
            pixels. The default is 2.
        nblobs: INT
            Number of distinct regions to include. That is, the number of
            distinct sources to include in the data This number is best to
            be set via trial and error, by looking at the mask beause sometimes
            two sources are close enough together to form a single blob.
            The default is 1.
        nproc: INT
            Number of processes to spawn. Should be less than the number of cores
            in the system for optimal run times.
        plotroot: STRING
            The root name for the QA plots that are generated.
            The default is './Test'
        outfile: STRING
            The outputfile with all of the parameters from the fitting.
            The default is './Test_fit.json'

        Returns
        -------
            None : NoneType
        """
        if self.data.ndim != 2:
            raise IOError('data needs to be 2D for this fitting program. If you want to fit a single channel '
                          'please use get_slice to select the channel and try again')
        self.load_initialparameters(initpar)
        self.create_gaussiankernel(lsf_sigma=None, kernelsize=4)
        self.kernel = self.kernel[0, :]
        if rms is None:
            self.variance = np.full_like(self.data, self.calculate_sigma() ** 2)
        else:
            self.variance = np.full_like(self.data, rms ** 2)
        if regular is ():
            regular = (int(self.data.shape[1] / 2), int(self.data.shape[0] / 2))
        self.create_maskarray(sampling=sampling, regular=regular, sigma=sigma, nblobs=nblobs)
        kwargs = {'mcmcmap': self.mcmcmap, 'data': self.data, 'initpar': initpar,
                  'beam_area': self.beam['BAREA_PIX'], 'par': self.par,
                  'kernel': self.kernel, 'variance': self.variance, 'maskarray': self.maskarray,
                  '2dmodelname': '__' + modelname + '__'}
        # load the hdf5 backend if filename is specified
        if mcmcfilename is not None:
            if mcmcfilename[-5:] != '.hdf5':
                mcmcfilename = mcmcfilename + '.hdf5'
            backend = emcee.backends.HDFBackend(mcmcfilename)
        else:
            backend = None
        # calculate the intial probability, if it is too small then exit
        init_prob = __get_2dposterior__(self.mcmcpar, **kwargs)
        if np.isinf(init_prob):
            raise ValueError('Initial parameters yield zero probability.' +
                             ' Please choose other initial parameters.')
        print('Intial probability is: {}'.format(init_prob))
        # define the sampler
        os.environ["OMP_NUM_THREADS"] = "1"
        with Pool(nproc) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, len(self.mcmcmap), __get_2dposterior__, pool=pool,
                                            backend=backend, kwargs=kwargs)
            # initiate the walkers
            p0 = [(1 + 0.02 * np.random.rand(len(self.mcmcmap))) * self.mcmcpar for _walker in range(nwalkers)]
            # run the mcmc chain
            sampler.run_mcmc(p0, nsteps, progress=True)
        # Store the results
        self.mcmcarray = np.transpose(sampler.chain, axes=(1, 0, 2))
        self.mcmclnprob = np.transpose(sampler.lnprobability, axes=(1, 0))
        # create the chain plot QA
        make_mcmcchainfigure(self, plotfig=plotroot + '_mcmcchain.pdf')
        self.get_chainresults(reload_model=False)
        self.update_parameters(self.chainpar['MedianArray'])
        self.model = eval(kwargs['2dmodelname'])(**kwargs)
        self.residual = self.data - self.model
        # create the corner plot QA
        make_cornerfigure(self, plotfig=plotroot + '_corner.pdf')
        # create the data, model and residual QA
        make_2dmodelfigure(self, plotfig=plotroot + '_2dmodelfit.pdf')
        for x in self.chainpar:
            if x == 'MedianArray' or x == 'BestArray':
                continue
            if self.chainpar[x]['Unit'] is not None:
                self.chainpar[x]['Unit'] = self.chainpar[x]['Unit'].to_string()
            if self.chainpar[x]['Conversion'] is not None:
                self.chainpar[x]['Conversion'] = self.chainpar[x]['Conversion'].to_string()
        with open(outfile, "w") as of:
            json.dump(self.chainpar, of, indent=4)

    def __calculate_loglikelihood__(self):
        """
        Calculate the log likelihood.

        wrapper function to calculate the log likelihood of the model over
        the given mask and the defined likelihood/probability method.
        """
        # generate a mask array (if not defined)
        if not hasattr(self, 'maskarray'):
            raise ValueError('Need to define a mask for calculating the likelihood function')
        # get the dictionary needed to get the probability
        kwargs = self.__define_kwargs__()
        # calculate the probability with __ln_prob__()
        lnlike = __lnprob__(self.mcmcpar,  **kwargs)
        return lnlike

    def __define_kwargs__(self):
        """
        Define the kwargs dictionary.

        This will generate a simple dictionary of all of the needed
        information to create a model, and run the emcee process. This
        extra step is necessary because the multithreading capability of
        emcee cannot handle the original structure.
        """
        mkeys = ['modelname', 'intensityprofile', 'velocityprofile', 'dispersionprofile']
        m_string = {}
        for mkey in mkeys:
            m_string[mkey] = getattr(self, mkey, None)
        kwargs = {'mstring': m_string}
        akeys = ['mcmcmap', 'par', 'shape', 'data', 'kernel', 'variance', 'maskarray', 'initpar', 'probmethod',
                 'kernelarea']
        for akey in akeys:
            kwargs[akey] = getattr(self, akey, None)
        kwargs['convolve'] = True
        return kwargs


#####################################################################
# THIS IS THE END OF THE CLASS SPECIFIC DEFINITIONS. BELOW ARE SOME #
# "UNDER-THE-HOOD" functions and definitions                        #


def __lnprob__(parameters, **kwargs):
    """
    Calculate the log probability (posterior) of the data and model.

    This will calculate the log-probability of the model compared
    to the data. It will either use the ks-test approach to calculate the
    probability that the residual is consistent with a Gaussian (if a
    single variance is given in self.variance) or a chi-squared probability
    is calculated (if self.variance is an array the size of the data).

    NOTE: Currently the prior distributions are not multiplied through instead
    they are taken as either -inf or 1. This is ok for uninformed priors but
    NOT correct for other priors.
    """
    # log of the priors
    lnprior = __get_priorlnprob__(parameters, **kwargs)
    if not np.isfinite(lnprior):
        return lnprior
    else:
        # update the parameters according to the predefined map
        __update_parameters__(parameters, **kwargs)
        # create a model for the cube
        model = __create_model__(**kwargs)
        # calculate the residual
        residual = kwargs['data'] - model
        # calculate the probability
        lnprob = __get_lnprob__(residual, **kwargs)
        return lnprob


def __get_priorlnprob__(parameters, **kwargs):
    """
    Calculate the prior log probability.

    Calculate prior probability distribution function based on the
    distribution defined for each parameter.
    """
    lnprior = []
    for parameter, key in zip(parameters, kwargs['mcmcmap']):
        if kwargs['initpar'][key]['Conversion'] is not None:
            loc = (kwargs['initpar'][key]['Dloc'] * kwargs['initpar'][key]['Unit'] /
                   kwargs['initpar'][key]['Conversion']).value
            scale = (kwargs['initpar'][key]['Dscale'] * kwargs['initpar'][key]['Unit'] /
                     kwargs['initpar'][key]['Conversion']).value
        else:
            loc = kwargs['initpar'][key]['Dloc']
            scale = kwargs['initpar'][key]['Dscale']
        log_prior = eval(kwargs['initpar'][key]['Dist']).logpdf(parameter, loc=loc, scale=scale) + np.log(scale)
        lnprior.append(log_prior)
    return np.sum(lnprior)


def __get_lnprob__(residual, **kwargs):
    """
    Calculate the probability of the model (not including priors).

    This function will calculate the probability for the residual. Several
    approaches can be defined using the kwargs['ProbMethod']. The most stable
    and recommended approach is the standard chi-squared. One can also try
    the bootstrap methods, which have been tested less, but appear to work
    decent.

    'ChiSq' likelihood function is calculated using the approach:
    ChiSquared = np.sum(np.square(residual) / variance +
                        np.log(2 * pi * variance))
    Here the second part is really not necessary as it is a constant term and
    the log likelihood is insensitive to constant additive factors. We assume
    that the PSF/beam is Nyquist sampled, meaning that we have 2 independent
    measurements per FWHM (this corresponds to 4/ln(2) independent measurements
    per kernelarea. The approximate size of the independent measurement is
    therefore kernelarea / 4/ln(2). The chi-squared function should be divided
    by this to get the log likelihood funtion:

    -0.5 * ChiSquared / Nyquist, where Nyquist = kernelarea / (4 / ln(2))
    """
    # the bootstrap methods
    if kwargs['probmethod'] == 'BootKS' or kwargs['probmethod'] == 'BootChiSq':
        p_arr = []
        for bootstrap in np.arange(kwargs['maskarray'].shape[1]):
            # get indices
            indices = kwargs['maskarray'][:, bootstrap]
            residualsample = residual.flatten()[indices]
            if kwargs['probmethod'] == 'BootKS':
                scale = np.median(np.sqrt(kwargs['variance']))
                sigmadist_theo = norm(scale=scale)
                residualsample = residualsample - np.mean(residualsample)
                prob = kstest(residualsample, sigmadist_theo.cdf)[1]
                if prob <= 0.:
                    p_arr.append(-np.inf)
                else:
                    p_arr.append(np.log(prob))
            elif kwargs['probmethod'] == 'BootChiSq':
                variancesample = kwargs['variance'].flatten()[indices]
                chisq = np.sum(np.square(residualsample) / variancesample)
                p_arr.append(-0.5 * chisq)
            else:
                raise ValueError('Bootstrap method not defined.')
        # return the median log-probability
        return np.median(p_arr)
    else:
        residualsample = residual[np.where(kwargs['maskarray'])]
        variancesample = kwargs['variance'][np.where(kwargs['maskarray'])]
        if kwargs['probmethod'] == 'KS':
            scale = np.median(np.sqrt(variancesample))
            sigmadist_theo = norm(scale=scale)
            residualsample = residualsample - np.mean(residualsample)
            prob = kstest(residualsample, sigmadist_theo.cdf)[1]
            if prob <= 0.:
                lnprob = -np.inf
            else:
                lnprob = np.log(prob)
        elif kwargs['probmethod'] == 'ChiSq':
            # chisq = np.sum(np.square(residualsample) / variancesample)
            chisq = np.sum(np.square(residualsample) / variancesample +
                           np.log(2 * np.pi * variancesample))
            nyquist = kwargs['kernelarea'] * np.log(2) / 4
            lnprob = -0.5 * chisq / nyquist
        else:
            raise ValueError('Probability method not defined.')
        return lnprob


def __update_parameters__(parameters, **kwargs):
    """
    Update the parameters in the kwargs dictionary.

    This will update the parameters with the values from the MCMC
    chain parameters. This should be the first call during running of the
    MCMC chain when the probability is calculated.
    """
    for parameter, key in zip(parameters, kwargs['mcmcmap']):
        kwargs['par'][key] = parameter


def __create_model__(**kwargs):
    """
    Create the model from the parameters.

    This function will take the stored kwargs and will generate the
    model. NOTE: currently no checking is done that the parameters are
    actually defined for the model that is requested. This will likely
    result in a rather benign AttributeError.
    """
    modeln = kwargs['mstring']['modelname']
    model = eval(modeln)(**kwargs)
    return model


def __get_regulargrid__(shape, kernelarea, sampling, center):
    """
    Create a regular grid through tuple 'center'.

    Creates a regular grid of values for the given array shape. Such that
    there are approximately n samplings per kernelarea, where n is given by
    the sampling parameter. The array will be sampled at center.
    """
    # approximate scalelength for the points in pixels
    r = np.sqrt(kernelarea * 4 * np.log(2) / np.pi) / sampling
    # xarray
    tx = np.zeros(shape[-1])
    xidx = np.arange(center[0] - r * shape[-1], center[0] + r * shape[-1], r)
    xidx = xidx[(xidx > 0) & (xidx < shape[-1])].astype(int)
    tx[xidx] = 1
    # yarray
    ty = np.zeros(shape[-2])
    yidx = np.arange(center[1] - r * shape[-2], center[1] + r * shape[-2], r)
    yidx = yidx[(yidx > 0) & (yidx < shape[-2])].astype(int)
    ty[yidx] = 1
    if len(shape) == 3:
        x = np.tile(tx[np.newaxis, np.newaxis, :], (shape[-3], shape[-2], 1))
        y = np.tile(ty[np.newaxis, :, np.newaxis], (shape[-3], 1, shape[-1]))
    else:
        x = np.tile(tx[np.newaxis, :], (shape[-2], 1))
        y = np.tile(tx[:, np.newaxis], (1, shape[-1]))
    # final mask
    mask = x * y
    return mask


def __get_sigmamask__(data, variance, kernel, sigma, nblobs, fmaskgrow):
    """
    Create a mask based on the variance of the cube.

    The mask will be cut above the given sigma level. After this only the
    n - largest blobs will be considered. Finally the mask can be grown to
    include some buffer of values below the sigma cutoff.
    """
    # create the blobs and sort them
    lab = measure.label(data > sigma * np.sqrt(variance))
    bins = np.bincount(lab.flatten())
    args = np.argsort(bins)[::-1][1:1+nblobs]
    # add the blobs to the mask
    mask = np.zeros_like(data)
    for arg in args:
        mask[np.where(lab == arg)] = 1
    # grow the mask
    if fmaskgrow != 1:
        mask = convolve(mask, kernel)
        mask = np.where(mask < fmaskgrow, 0, 1)
    return mask


def __get_2dposterior__(mcmc_pars, **kwargs):
    # calculate the log prior:
    prior_lnprob = __get_priorlnprob__(mcmc_pars, **kwargs)
    if not np.isfinite(prior_lnprob):
        return prior_lnprob
    # update the parameters
    __update_parameters__(mcmc_pars, **kwargs)
    # create the model and sample
    model = eval(kwargs['2dmodelname'])(**kwargs)
    model_sample = model[np.where(kwargs['maskarray'])] * 1E3
    data_sample = kwargs['data'][np.where(kwargs['maskarray'])] * 1E3
    variance_sample = kwargs['variance'][np.where(kwargs['maskarray'])] * 1E6
    # calculate the log-likelihood
    ln_chisq = np.nansum(np.square(data_sample - model_sample) / variance_sample + np.log(2 * np.pi * variance_sample))
    nyquist = kwargs['beam_area'] * np.log(2) / 4
    lnprob = -0.5 * ln_chisq / nyquist
    return lnprob + prior_lnprob


def __gaussian2d__(**kwargs):
    idx = np.indices(kwargs['data'].shape)
    a, x, y = kwargs['par']['A'], kwargs['par']['x'], kwargs['par']['y']
    sig_x, sig_y, theta = kwargs['par']['sig_x'], kwargs['par']['sig_y'], kwargs['par']['theta']
    return convolve(Gaussian2D(a, x, y, sig_x, sig_y, theta)(idx[1], idx[0]), kwargs['kernel'])


def __sersic2d__(**kwargs):
    idx = np.indices(kwargs['data'].shape)
    a, x, y, n = kwargs['par']['A'], kwargs['par']['x'], kwargs['par']['y'], kwargs['par']['n']
    r, e, theta = kwargs['par']['r'], kwargs['par']['e'], kwargs['par']['theta']
    return convolve(Sersic2D(a, r, n, x, y, e, theta)(idx[1], idx[0]), kwargs['kernel'])


def __coresersic2d__(**kwargs):
    idx = np.indices(kwargs['data'].shape)
    a, x, y, n = kwargs['par']['A'], kwargs['par']['x'], kwargs['par']['y'], kwargs['par']['n']
    r, e, theta = kwargs['par']['r'], kwargs['par']['e'], kwargs['par']['theta']
    r_b, gamma = kwargs['par']['r_b'], kwargs['par']['gamma']
    return convolve(CoreSersic2D(a, r, n, x, y, e, theta, r_b, gamma)(idx[1], idx[0]), kwargs['kernel'])


class CoreSersic2D(Fittable2DModel):
    amplitude = Parameter(default=1, description="Surface brightness at r_eff")
    r_eff = Parameter(default=1, description="Effective (half-light) radius")
    n = Parameter(default=4, description="Sersic Index")
    x_0 = Parameter(default=0, description="X position of the center")
    y_0 = Parameter(default=0, description="Y position of the center")
    ellip = Parameter(default=0, description="Ellipticity")
    theta = Parameter(default=0.0, description="Rotation angle either as a float (in radians) or a |Quantity| angle")
    r_b = Parameter(default=0.5, description="Break radius")
    gamma = Parameter(default=0.0, description="Inner power slope")

    @classmethod
    def evaluate(cls, x, y, amplitude, r_eff, n, x_0, y_0, ellip, theta, r_b, gamma, c=0):
        """Two dimensional CoredSersic profile function."""
        bn = gammaincinv(2.0 * n, 0.5)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        x_maj = np.abs((x - x_0) * cos_theta + (y - y_0) * sin_theta)
        x_min = np.abs(-(x - x_0) * sin_theta + (y - y_0) * cos_theta)
        b = (1 - ellip) * r_eff
        expon = 2.0 + c
        inv_expon = 1.0 / expon
        z = ((x_maj / r_eff) ** expon + (x_min / b) ** expon) ** inv_expon
        return np.where(r_eff * z < r_b, amplitude * (r_b / (r_eff * z + 1E-10))**gamma,
                        amplitude * np.exp(-bn * ((z ** (1 / n)) - (r_b / r_eff) ** (1 / n))))
