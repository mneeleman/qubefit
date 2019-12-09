# modules
import numpy as np
from scipy.special import erf
import emcee
from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from scipy.stats import norm, uniform, kstest
from skimage import measure
from qubefit.qube import Qube
from qubefit.qfmodels import *


class QubeFit(Qube):

    def __init__(self, modelname='ThinDisk',
                 intensityprofile=['Exponential', None, 'Delta'],
                 velocityprofile=['Constant', None, None],
                 dispersionprofile=['Constant', None, None]):

        """ initiate with the characteristics of the wanted model. Current
        choices are:
        modelname: 'ThinDisk'
        radialprofile: 'Exponential'
        heightprofile: 'Delta'
        velocityprofile: 'Constant'
        dispersionprofile: 'Constant'
        """

        self.modelname = modelname
        self.intensityprofile = intensityprofile
        self.velocityprofile = velocityprofile
        self.dispersionprofile = dispersionprofile

    def create_gaussiankernel(self, channels=None, LSFSigma=None,
                              kernelsize=4):

        """ This function will generate a Gaussian kernel from the data
        stored in the Qube. It will look for the beam keyword and generate
        a gaussian kernel from the shape of the synthesized beam given here.
        Several different kernels can be returned.
        1) A list of 3D kernels -one for each spectral bin- that
        takes into account both a (varying) LSF and a (varying) PSF
        2) A list of 2D kernels -one for each spectral bin- that
        takes into account a varying PSF, but no LSF
        3) A 3D kernel that has constant PSF and LSF
        """

        # quick check that the beam attribute has been defined
        if not hasattr(self, 'beam'):
            raise ValueError('Beam must be defined to create gaussian kernel')

        # define some parameters for the beam
        Bmaj = (self.beam['BMAJ'] / np.sqrt(8 * np.log(2)) /
                np.abs(self.header['CDELT1']))
        Bmin = (self.beam['BMIN'] / np.sqrt(8 * np.log(2)) /
                np.abs(self.header['CDELT1']))
        Bpa = self.beam['BPA']
        theta = np.pi / 2. + np.radians(Bpa)

        # Here decide which channel(s) to use for generating the kernel
        if channels is None:
            channels = np.arange(len(Bmaj))

        # create an array of LSFSigma (if needed)
        if LSFSigma is not None and type(LSFSigma) is float:
            LSFSigma = np.full(len(Bmaj), LSFSigma)

        Kernel = ()
        for ii in channels:
            xsize = 2 * np.ceil(kernelsize*Bmaj[ii]) + 1
            ysize = 2 * np.ceil(kernelsize*Bmaj[ii]) + 1
            TwoDKernel = Gaussian2DKernel(Bmaj[ii], Bmin[ii], theta=theta[ii],
                                          x_size=xsize, y_size=ysize).array

        if LSFSigma is None:
            Kernel = Kernel + (TwoDKernel,)
        else:
            LSFKernel = Gaussian1DKernel(LSFSigma[ii]).array
            TKernel = np.zeros(LSFKernel.shape + TwoDKernel.shape)
            TKernel[LSFKernel.shape[0] // 2, :, :] = TwoDKernel
            ThreeDKernel = convolve(TKernel, LSFKernel[np.newaxis,
                                                       np.newaxis, ...])
            Kernel = Kernel+(ThreeDKernel,)

        # now remove the list if it is just one element
        if len(Kernel) == 1:
            Kernel = Kernel[0]

        # now assign this as an attribute to the Qube
        self.kernel = Kernel

        # also calculate the area of the kernel
        kernelarea = Bmaj * Bmin * 2 * np.pi
        if channels is not None:
            kernelarea = kernelarea[channels]
        self.kernelarea = kernelarea

    def load_initialparameters(self, Parameters):

        """ This function will load in the initial parameters from a
        dictionary. The dictionary is transfered into a keyword, as well
        as a parameter keyword that contains all of the needed parameters
        for the model. Finally a parameter array is created of just values,
        together with a mapping array and distribution lists all of which
        are used in the MCMC process.
        """

        self.initpar = Parameters
        self.par = {}
        self.mcmcpar = []
        self.mcmcmap = []
        self.priordist = []
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

    def create_model(self, convolve=True):

        """ This function will take the stored parameters (in self.par) and
        will generate the requested model. NOTE: currently no checking is
        done that the parameters are actually defined for the model that is
        requested. This will likely result in a rather benign AttributeError.

        keywords
        --------
        convolve: Boolean (True|False)
                  Determines if the convolution is done or not. Default(True)
        """

        kwargs = self.__define_kwargs__()
        kwargs['convolve'] = convolve
        self.model = __create_model__(**kwargs)

    def run_mcmc(self, nwalkers=50, nruns=100, threads=10, fancy=True,
                 tofile=False, filename='./chain.dat'):

        """ This is the heart of QubeFit. It will run an MCMC process on a
        predefined model and will return the resulting chain of the posterior
        PDFs of each individual parameter that was varied.
        """

        # intiate the model (redo if already done)
        self.create_model()

        # create the bootstrap array (do not redo if already done)
        if not hasattr(self, 'maskarray'):
            self.__create_maskarray__()

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
        sampler = emcee.EnsembleSampler(nwalkers, self.mcmcdim,
                                        __lnprob__, kwargs=kwargs,
                                        threads=threads)

        # initiate the walkers
        p0 = [(1 + 0.02 * np.random.rand(self.mcmcdim)) * self.mcmcpar
              for walker in range(nwalkers)]

        ######################
        # run the mcmc chain #

        # progress bar definition
        if fancy:
            widgets = [Percentage(), ' ', Bar(), ' ', AdaptiveETA()]
            pbar = ProgressBar(widgets=widgets, maxval=nruns)
            pbar.start()
        else:
            print('Starting the MCMC chain')
            width = 30

        # writing to file
        if tofile:
            f = open(filename, "w")
            f.close()
            storechain = False
        else:
            storechain = True

        # here is the actual iterative part of the MCMC
        for i, result in enumerate(sampler.sample(p0, iterations=nruns,
                                                  storechain=storechain)):
            # add to file (if set)
            if tofile:
                position = result[0]
                lnprob = result[1]
                f = open("chain.dat", "a")
                for k in range(position.shape[0]):
                    print
                    f.write("{0:4d} {1:s} {2:4f}\n".format(k,
                            " ".join(position[k].astype(str)),
                            lnprob[k]))
                f.close()

            # print the progress bar
            if fancy:
                pbar.update(i + 1)
            else:
                n = int((width+1) * float(i) / nruns)
                print("[{0}{1}]".format('#' * n, ' ' * (width - n)), end='\r')

        # finish things up:
        if fancy:
            pbar.finish()
        else:
            print("[###########DONE!")

        # Now store the results into an array

        if not tofile:
            self.mcmcarray = sampler.chain

        return sampler

    def get_chainresults(self, chain, burnin=0.0):

        """ calling this function will generate a dictionary with the median
        values, and 1, 2, 3  sigma ranges of the parameters. These have been
        converted into their original units as given in the initial parameters.
        """

        # if file is given for string convert to numpy array
        if type(chain) == str:
            chain = np.load(chain)

        # get the burnin value below which to ditch the data
        burninvalue = int(np.ceil(chain.shape[1]*burnin))

        # sigma ranges
        perc = (1 / 2 + 1 / 2 * erf(np.arange(-3, 4, 1) / np.sqrt(2))) * 100

        # find the index with the highest probability
        BestValIdx = np.unravel_index(chain[:, :, -1].argmax(),
                                      chain[:, :, -1].shape)

        # create the output dictionary from the initial input dictionary
        par, MedArr, BestArr = {}, {}, {}
        for key in self.initpar.keys():

            # get unit of the cube values
            if self.initpar[key]['Conversion'] is not None:
                Unit = (self.initpar[key]['Unit'] /
                        self.initpar[key]['Conversion'].unit)
            else:
                Unit = self.initpar[key]['Unit']

            # now calculate median, etc. only if it was not held fixed
            if not self.initpar[key]['Fixed']:
                idx = self.mcmcmap.index(key)
                Data = chain[:, burninvalue:, idx]
                Values = np.percentile(Data, perc) * Unit
                MedArr.update({key: Values[3].value})
                BestValue = chain[BestValIdx[0], BestValIdx[1], idx] * Unit
                BestArr.update({key: BestValue.value})
                if self.initpar[key]['Conversion'] is not None:
                    Values = Values * self.initpar[key]['Conversion']
                    BestValue = BestValue * self.initpar[key]['Conversion']
            else:
                Values = (np.array([0, 0, 0, self.initpar[key]['Value'],
                                    0, 0, 0]) *
                          self.initpar[key]['Unit'])
                BestValue = (self.initpar[key]['Value'] *
                             self.initpar[key]['Unit'])

            par.update({key: {'Median': Values[3].value,
                              '1Sigma': [Values[2].value, Values[4].value],
                              '2Sigma': [Values[1].value, Values[5].value],
                              '3Sigma': [Values[0].value, Values[6].value],
                              'Best': BestValue.value, 'Unit': Values[0].unit,
                              'Conversion': self.initpar[key]['Conversion']}})
        par.update({'MedianArray': MedArr, 'BestArray': BestArr})

        self.chainpar = par

    def update_parameters(self, parameters):

        """ This will update the 'par' keyword using the values given in
        parameters (here parameters is a dictionary)
        """

        for key in parameters.keys():
            self.par[key] = parameters[key]

    def calculate_probability(self):

        """ this will calculate a goodness-of-fit either a KS test or
        reduced chi-squared
        """

        # generate a bootstraparray (if needed)
        if not hasattr(self, 'maskarray'):
            self.__create_maskarray__()

        # get the dictionary needed to get the probability
        kwargs = self.__define_kwargs__()

        # calculate the probability with __ln_prob__()
        pval = __lnprob__(self.mcmcpar,  **kwargs)

        return pval

    def create_maskarray(self, sampling=2., bootstrapsamples=200,
                         method='BootChiSq', mask=None, regular=None,
                         sigma=None, nblobs=1, fmaskgrow=0.1):

        """ This will generate the mask array. It should only be done
        once so that the mask does not change between runs of the MCMC
        chain, which could/will result in slightly different probabilities
        for the same parameters. The mask returned is either the same size
        as the data cube (filled with np.NaN and 1's for data to
        ignore/include) or an array with size self.data.size by
        bootstrapsamples.
        The first is used for directly calculating the KS
        or ChiSquared value of the given data point, while the second is used
        for the bootstrap method of both functions.

        Several mask options can be specified.

        keywords:
        ---------
        sampling (float|default: 2.):
            The number of samples to choose per beam/kernel area

        bootstrapsamples (int|default: 200):
            The number of bootstrap samples to generate.

        method (string|default: 'ChiSq'):
            Specifies which method to use
            to calculate the probability for this masked array. It will also
            determine which array needs to be returned. Currently allowed
            values are: 'BootKS', 'BootChiSq', 'KS', ChiSq', 'RedChiSq'

        mask (np.array|default: None):
            If specified it will take this mask as the mask array,
            while setting the method that is used with this mask.
            The mask should have 0 (False) where not wanted and 1(True)
            if wanted.

        regular(tuple|default: None):
            If a two element tuple is given, then a regular grid will be
            generated that is specified by the number of samplings per
            kernelarea which will go through the given tuple.

        sigma(float|default: None):
            If given, a mask is created that will just consider values above
            the specified sigma (based on the calculated variance). Only the
            n largest blobs will be considered (specified optionally by
            nblobs). It will grow this mask by convolving with the kernel
            to include also adjacent 'zero' values.

        nblobs(int|default: 1):
            number of blobs to consider in the sigma-mask method

        fmaskgrow (float|default: 0.1):
            fraction of the peak value which marks the limit of where to cut
            of the growth factor. If set to 1, the mask is not grown.

        returns:
        -------
        sets the following keys:
            self.maskarray
            self.probmethod
        """

        # populate the PropMethod key
        self.probmethod = method

        # bootstrap array
        if method == 'BootKS' or method == 'BootChiSq':

            # create the mask array
            size = (int(sampling * self.data.size / self.kernelarea),
                    bootstrapsamples)
            self.maskarray = np.random.randint(self.data.size, size=size)

        # regular array
        elif method == 'KS' or method == 'ChiSq' or method == 'RedChiSq':

            # masks are cumulative / multiplicative
            self.maskarray = np.ones_like(self.data)

            if mask is not None:
                self.maskarray *= mask

            if regular is not None:
                self.maskarray *= __get_regulargrid__(self.data.shape,
                                                      self.kernelarea,
                                                      sampling, regular)
            if sigma is not None:
                self.maskarray *= __get_sigmamask__(self.data,
                                                    self.variance,
                                                    self.kernel,
                                                    sigma, nblobs, fmaskgrow)

        # not defined method
        else:
            raise ValueError('qubefit: Probability method is not defined: ' +
                             '{}'.format(method))

    def __define_kwargs__(self):

        """ This will generate a simple dictionary of all of the needed
        information to create a model, and run the emcee process. This
        extra step is necessary because the multithreading capability of
        emcee cannot handle the original structure.
        """

        mkeys = ['modelname', 'intensityprofile',
                 'velocityprofile', 'dispersionprofile']

        MString = {}
        for mkey in mkeys:
            MString[mkey] = getattr(self, mkey, None)

        akeys = ['mcmcmap', 'par', 'shape', 'data', 'kernel',
                 'variance', 'maskarray', 'initpar', 'probmethod',
                 'kernelarea']

        kwargs = {'mstring': MString}
        for akey in akeys:
            kwargs[akey] = getattr(self, akey, None)
        kwargs['convolve'] = True

        return kwargs


#####################################################################
# THIS IS THE END OF THE CLASS SPECIFIC DEFINITIONS. BELOW ARE SOME #
# "UNDER-THE-HOOD" functions and definitions                        #


def __lnprob__(parameters, **kwargs):

    """ This will calculate the log-probability of the model compared
    to the data. It will either use the ks-test approach to calculate the
    probability that the residual is consistent with a Gaussian (if a
    single variance is given in self.variance) or a chi-squared probability
    is calculated (if self.variance is an array the size of the data).
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

    """ Calculate prior probability distribution function based on the
    distribution defined for each parameter.
    """

    lnprior = []
    for parameter, key in zip(parameters, kwargs['mcmcmap']):
        if kwargs['initpar'][key]['Conversion'] is not None:
            Loc = (kwargs['initpar'][key]['Dloc'] *
                   kwargs['initpar'][key]['Unit'] /
                   kwargs['initpar'][key]['Conversion']).value
            Scale = (kwargs['initpar'][key]['Dscale'] *
                     kwargs['initpar'][key]['Unit'] /
                     kwargs['initpar'][key]['Conversion']).value
        else:
            Loc = kwargs['initpar'][key]['Dloc']
            Scale = kwargs['initpar'][key]['Dscale']

        lnprior.append(eval(kwargs['initpar'][key]['Dist']).logpdf(parameter,
                       loc=Loc, scale=Scale))

    return np.sum(lnprior)


def __get_lnprob__(residual, **kwargs):

    """ This function will calculate the probability for the residual. Several
    approaches can be defined using the kwargs['ProbMethod']. This keyword is
    set when the mask array is specified (see self.create_maskarray).

    'Chi-Squared' likelihood function is calulated using the approach:
    ChiSquared = np.sum(np.square(data - model) / variance +
                        np.log(2 * pi * variance))
    Here the second part is really not necessary as it is a constant term and
    the log likelihood is insensitive to constant additive factors. We assume
    that the PSF/beam is Nyquist sampled, meaning that we have 
    
    -0.5 * [(data - model)**2 / variance +  np.log(2 * np.pi * variancesample)]
    * Nyquist**2
    The 

    NOTE: Currently the prior distributions are not multiplied through instead
    they are taken as either -inf or 1. This is ok for uniformed priors but
    NOT correct for other priors.
    """

    # the bootstrap methods
    Boot = (kwargs['probmethod'] == 'BootKS' or
            kwargs['probmethod'] == 'BootChiSq')
    if Boot:
        PArr = []
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
                    PArr.append(-np.inf)
                else:
                    PArr.append(np.log(prob))
            elif kwargs['probmethod'] == 'BootChiSq':
                variancesample = kwargs['variance'].flatten()[indices]
                chisq = np.sum(np.square(residualsample) / variancesample)
                PArr.append(-0.5 * chisq)
            else:
                raise ValueError('Bootstrap method not defined.')

        # return the median log-probability
        return np.median(PArr)

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
            lnprob = -0.5 * chisq / kwargs['kernelarea'] * 2**2
        elif kwargs['probmethod'] == 'RedChiSq':
            # chisq = np.sum(np.square(residualsample) / variancesample)
            chisq = np.sum(np.square(residualsample) / variancesample +
                           np.log(2 * np.pi * variancesample))
            dof = len(residualsample) - len(kwargs['mcmcpar'])
            lnprob = -0.5 * chisq / dof
        else:
            raise ValueError('Bootstrap method not defined.')

        return lnprob


def __update_parameters__(parameters, **kwargs):

    """ This will update the parameters with the values from the MCMC
    chain parameters. This should be the first call during running of the
    MCMC chain when the probability is calculated
    """
    for parameter, key in zip(parameters, kwargs['mcmcmap']):
        kwargs['par'][key] = parameter


def __create_model__(**kwargs):

    """ This function will take the stored parameters (args) and
    will generate the requested model. NOTE: currently no checking is
    done that the parameters are actually defined for the model that is
    requested. This will likely result in a rather benign AttributeError.
    """

    modeln = kwargs['mstring']['modelname']
    model = eval(modeln)(**kwargs)

    return model


def __get_regulargrid__(shape, kernelarea, sampling, center):

    """ Creates a regular grid of values for the given array shape. Such that
    there are approximately n samplings per kernelarea, where n is given by
    the sampling parameter.
    """

    # approximate scalelength for the points in pixels
    r = np.sqrt(kernelarea / sampling * np.pi)

    # xarray
    tx = np.zeros(shape[2])
    xidx = np.arange(center[0] - r * shape[2], center[0] + r * shape[2], r)
    xidx = xidx[(xidx > 0) & (xidx < shape[2])].astype(int)
    tx[xidx] = 1
    x = np.tile(tx[np.newaxis, np.newaxis, :], (shape[0], shape[1], 1))

    # yarray
    ty = np.zeros(shape[1])
    yidx = np.arange(center[1] - r * shape[1], center[1] + r * shape[1], r)
    yidx = yidx[(yidx > 0) & (yidx < shape[1])].astype(int)
    ty[yidx] = 1
    y = np.tile(ty[np.newaxis, :, np.newaxis], (shape[0], 1, shape[2]))

    mask = x * y

    return mask


def __get_sigmamask__(data, variance, kernel, sigma, nblobs, fmaskgrow):

    """ Creates a mask based on the variance of the cube
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
