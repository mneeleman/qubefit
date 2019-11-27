# modules
import numpy as np
from scipy.special import erf
import emcee
from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from scipy.stats import norm, uniform, kstest
from qubefit.qube import Qube
from qubefit.qfmodels import *


class QubeFit(Qube):

    def __init__(self, modelname='ThinDisk',
                 intensityprofile=['Exponential', None, 'Exponential'],
                 velocityprofile=['Constant', None, None],
                 dispersionprofile=['Constant', None, None],
                 nbootstrapsamples=500):

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
        self.nbootstrapsamples = nbootstrapsamples

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
        if not hasattr(self, 'bootstraparray'):
            self.__create_bootstraparray__()

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

    def calculate_probability(self, KStest=False, fullarray=False, Mask=None):

        """ this will calculate a goodness-of-fit either a KS test or
        reduced chi-squared
        """

        # calculate the residual
        residual = self.data - self.model
        if Mask is not None:
            residual = residual * Mask

        # generate a bootstraparray (if needed)
        if not hasattr(self, 'bootstraparray'):
            self.__create_bootstraparray__()

        # calculate the probability through bootstrap sampling
        pval = []
        for bootstrap in np.arange(self.bootstraparray.shape[1]):
            indices = self.bootstraparray[:, bootstrap]
            residualsample = residual.flatten()[indices]
            if Mask is not None:
                maskidx = np.isfinite(residualsample)
                residualsample = residualsample[maskidx]

            # KS test
            if KStest:
                sigmadist_theo = norm(scale=np.median(np.sqrt(self.variance)))
                residualsample = residualsample - np.mean(residualsample)
                prob = kstest(residualsample, sigmadist_theo.cdf)[1]
                if prob <= 0.:
                    pval.append(-np.inf)
                else:
                    pval.append(np.log(prob))
            else:
                variancesample = self.variance.flatten()[indices]
                if Mask is not None:
                    variancesample = variancesample[maskidx]
                chisq = np.sum(np.square(residualsample) / variancesample)
                dof = len(residualsample) - len(self.mcmcpar)
                redchisq = chisq / dof
                pval.append(redchisq)

        # return the median log-probability (unless full=True)
        if fullarray:
            return pval
        else:
            return np.median(pval)

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
                 'variance', 'bootstraparray', 'initpar', 'mask']

        kwargs = {'mstring': MString}
        for akey in akeys:
            kwargs[akey] = getattr(self, akey, None)
        kwargs['convolve'] = True

        return kwargs

    def __create_bootstraparray__(self, sampling=1.):

        """ This will generate the bootstrap array. It should only be done
        once so that the bootstrap calls will give the same probability each
        time. This is necessary for the MCMC code to give reasonable results.
        """

        size = (int(sampling * self.data.size / self.kernelarea),
                self.nbootstrapsamples)
        self.bootstraparray = np.random.randint(self.data.size, size=size)

#####################################################################
# THIS IS THE END OF THE CLASS SPECIFIC DEFINITIONS. BELOW ARE SOME #
# "UNDER-THE-HOOD" functions and definitions                        #


def __lnprob__(parameters, fullarray=False, reduced_chisquared=False,
               **kwargs):

    """ This will calculate the log-probability of the model compared
    to the data. It will either use the ks-test approach to calculate the
    probability that the residual is consistent with a Gaussian (if a
    single variance is given in self.variance) or a chi-squared probability
    is calculated (if self.variance is an array the size of the data).
    """

    # log of the priors
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

    if not np.isfinite(np.sum(lnprior)):
        return np.sum(lnprior)
    else:
        # update the parameters according to the predefined map
        __update_parameters__(parameters, **kwargs)

        # create a model for the cube
        model = __create_model__(**kwargs)

        # calculate the residual
        residual = kwargs['data'] - model
        if kwargs['mask'] is not None:
            residual = residual * kwargs['mask']

        # calculate the probability through bootstrap sampling
        pval = []
        for bootstrap in np.arange(kwargs['bootstraparray'].shape[1]):
            indices = kwargs['bootstraparray'][:, bootstrap]
            residualsample = residual.flatten()[indices]
            if kwargs['mask'] is not None:
                maskidx = np.isfinite(residualsample)
                residualsample = residualsample[maskidx]

            if kwargs['variance'].shape != kwargs['data'].shape:
                scale = np.median(np.sqrt(kwargs['variance']))
                sigmadist_theo = norm(scale=scale)
                residualsample = residualsample - np.mean(residualsample)
                prob = kstest(residualsample, sigmadist_theo.cdf)[1]
                if prob <= 0.:
                    pval.append(-np.inf)
                else:
                    pval.append(np.log(prob))
            else:
                variancesample = kwargs['variance'].flatten()[indices]
                if kwargs['mask'] is not None:
                    variancesample = variancesample[maskidx]
                chisq = np.sum(np.square(residualsample) / variancesample)
                if reduced_chisquared:
                    dof = len(residualsample) - len(kwargs['mcmcpar'])
                    redchisq = chisq / dof
                    pval.append(-0.5 * redchisq)
                else:
                    pval.append(-0.5 * chisq)

        # return the median log-probability (unless fullarray=True)
        if fullarray:
            return pval
        else:
            return np.median(pval)


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
