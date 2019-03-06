# modules
import numpy as np
import emcee
from progressbar import Bar, AdaptiveETA, Percentage, ProgressBar
from astropy.convolution import convolve, Gaussian1DKernel, Gaussian2DKernel
from scipy.stats import norm, uniform, kstest
from qube.qube import Qube
from astropy.coordinates.matrix_utilities import rotation_matrix as rm
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import CylindricalRepresentation
import astropy.units as u


class QubeFit(Qube):

    def __init__(self, modelname='ThinDisk',
                 intensityprofile=['Exponential', None, 'Delta'],
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
        Bpa  = self.beam['BPA']
        theta = np.pi / 2. + np.radians(Bpa)

        # Here decide which channel(s) to use for generating the kernel
        if channels is None:
            channels = np.arange(len(Bmaj))
            
        # create an array of LSFSigma (if needed)
        if LSFSigma is not None and type(LSFSigma) is float:
            LSFSigma = np.full(len(Bmaj),LSFSigma)

        Kernel=()
        for ii in channels:    
            TwoDKernel = Gaussian2DKernel(Bmaj[ii], Bmin[ii], theta=theta[ii],
                         x_size = 2 * np.ceil(kernelsize*Bmaj[ii]) + 1,
                         y_size = 2 * np.ceil(kernelsize*Bmaj[ii]) + 1).array

        if LSFSigma is None:
            Kernel = Kernel + (TwoDKernel,)
        else:
            LSFKernel = Gaussian1DKernel(LSFSigma[ii]).array
            TKernel = np.zeros(LSFKernel.shape +  TwoDKernel.shape)
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
        kernelarea = Bmaj * Bmin * np.pi
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
        
        
    def create_model(self, **kwargs):
        
        """ This function will take the stored parameters (in self.par) and 
        will generate the requested model. NOTE: currently no checking is
        done that the parameters are actually defined for the model that is 
        requested. This will likely result in a rather benign AttributeError.
        """
  
      # define the args keyword:
        kwargs = self.__define_kwargs__()
        
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
            raise ValueError('Initial parameters yield zero probability.'+
                                 ' Please choose other initial parameters.')
        
        print('Intial probability is: {}'.format(initprob))

        # define the sampler
        sampler = emcee.EnsembleSampler(nwalkers, self.mcmcdim, 
                                        __lnprob__, kwargs=kwargs, 
                                        threads=threads)
        
        # initiate the walkers
        p0 = [(1 + 0.02 * np.random.rand(self.mcmcdim)) * self.mcmcpar
              for walker in range(nwalkers)]
    
        ############################
        #### run the mcmc chain ####
        
        # progress bar definition
        if fancy:
            widgets = [Percentage(),' ', Bar(),' ', AdaptiveETA()]
            pbar = ProgressBar(widgets=widgets, maxval=nruns)
            pbar.start()
        else:
            print('Starting the MCMC chain')
            width = 30
        
        # writing to file
        if tofile:
            f = open(filename, "w")
            f.close()
            storechain=False
        else:
            storechain=True
            
            
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
                    
        ## Now store the results into an array, and calculate some useful
        ## quantities
        
        if not tofile:
            self.mcmcarray = sampler.chain
        
        
        return sampler
    
    
    def get_chainresults(self, chain, burnin=0.0):
    
        """ calling this function will generate a dictionary with the median 
        values, and 1, 2, 3  sigma ranges of the parameters. These have been 
        converted into their original units as given in the initial parameters.
        """
        
        ## if file is given for string convert to numpy array
        if type(chain) == str:
            chain = np.load(chain)
        
        ## get the burnin value below which to ditch the data
        burninvalue = int(np.ceil(chain.shape[1]*burnin))
        
        ## sigma ranges
        perc = (1/2 + 1/2 * np.erf(np.arange(-3,4,1)/np.sqrt(2))) * 100.
    
        ## find the index with the highest probability
        BestValIdx = np.unravel_index(chain[:,:,-1].argmax(), chain[:,:,-1].shape)
        
        ## create the output dictionary from the initial input dictionary
        par, MedArr, BestArr = {}, {}, {}
        for key in self.initpar.keys():
            
            # get unit of the cube values
            if self.initpar[key]['Conversion'] is not None:
                Unit = (self.initpar[key]['Unit']/ 
                        self.initpar[key]['Conversion'].unit)
            else:
                Unit = self.initpar[key]['Unit']
                
            ## now calculate median, etc. only if it was not held fixed
            if not self.initpar[key]['Fixed']:
                idx = self.mcmcmap.index(key)
                Data = chain[:,burninvalue:,idx]
                Values = np.percentile(Data, perc) * Unit
                MedArr.update({key:Values[3].value})
                BestValue = chain[BestValIdx[0],BestValIdx[1],idx] * Unit
                BestArr.update({key:BestValue.value})
                if self.initpar[key]['Conversion'] is not None:
                    Values = Values * self.initpar[key]['Conversion']
                    BestValue = BestValue * self.initpar[key]['Conversion']
            else:
                Values = (np.array([0,0,0,self.initpar[key]['Value'],0,0,0]) * 
                          self.initpar[key]['Unit'])
                BestValue = (self.initpar[key]['Value'] * 
                             self.initpar[key]['Unit'])
    
            par.update({key:{'Median': Values[3].value, 
                             '1Sigma': [Values[2].value, Values[4].value], 
                             '2Sigma': [Values[1].value, Values[5].value],
                             '3Sigma': [Values[0].value, Values[6].value],
                             'Best': BestValue.value, 'Unit':Values[0].unit,
                             'Conversion': self.initpar[key]['Conversion']}})
        par.update({'MedianArray':MedArr, 'BestArray':BestArr})
        
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
        
        ## generate a bootstraparray (if needed)
        if not hasattr(self, 'bootstraparray'):
            self.__create_bootstraparray__()
            
        # calculate the probability through bootstrap sampling
        pval = []
        for bootstrap in np.arange(self.bootstraparray.shape[1]):
            indices = self.bootstraparray[:,bootstrap]
            residualsample = residual.flatten()[indices]
            if Mask is not None:
                maskidx = np.isfinite(residualsample)
                residualsample = residualsample[maskidx]
        
            ## KS test
            if KStest:
                sigmadist_theo = norm(scale=np.median(np.sqrt(self.variance)))
                residualsample =  residualsample - np.mean(residualsample)
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
        
        kwargs = {'mstring':MString}
        for akey in akeys:
            kwargs[akey] = getattr(self, akey, None)
        
        return kwargs
    
    
    def __create_bootstraparray__(self, sampling=1.):
        
        """ This will generate the bootstrap array. It should only be done
        once so that the bootstrap calls will give the same probability each
        time. This is necessary for the MCMC code to give reasonable results.
        """
    
        self.bootstraparray = \
            np.random.randint(self.data.size, size=(int(sampling * 
                                                        self.data.size / 
                                                        self.kernelarea),
                                                    self.nbootstrapsamples))
        
        
##### THIS IS THE END OF THE CLASS SPECIFIC DEFINITIONS. BELOW ARE SOME 
##### "UNDER-THE-HOOD" functions and definitions 

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
            indices = kwargs['bootstraparray'][:,bootstrap]
            residualsample = residual.flatten()[indices]
            if kwargs['mask'] is not None:
                maskidx = np.isfinite(residualsample)
                residualsample = residualsample[maskidx]
        
            if kwargs['variance'].shape != kwargs['data'].shape:
                sigmadist_theo = norm(scale=np.median(np.sqrt(kwargs['variance'])))
                residualsample =  residualsample - np.mean(residualsample)
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
            
        # return the median log-probability (unless full=True)
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
    
    modeln = '_' + kwargs['mstring']['modelname'] + '_'
    model = eval(modeln)(**kwargs)
    
    return model
    
def __get_coordinates__(twoD=False, **kwargs):
    
    """ This will generate a set of cylindrical coordinates, rotated by
    the paralactic angle and the inclination. It returns both the sky 
    coordinates (R, phi, z) and the rotated coordinates (Rprime, phiprime,
    zprime).
    
    twoD: if set to True the radial and azimuthal angle are returned 
          for a plane rotated by PA and Inc.
          
    NOTE: THIS FUNCTION CAN BE/SHOULD BE REPLACED WITH THE MORE GENERIC
          FUNCTION THAT HOLDS FOR BOTH 2D and 3D (SEE RHOPHIZ_ARRAY BELOW)
    """
    
    if twoD:
        
        # make a grid of x and y values values in the plane of the sky,
        # where Y points in the direction of the PA of the galaxy
        X = (np.mgrid[0:kwargs['shape'][-2], 0:kwargs['shape'][-1]][-1] - 
             kwargs['par']['Xcen']).astype(float)
        Y = (np.mgrid[0:kwargs['shape'][-2], 0:kwargs['shape'][-1]][-2] -
             kwargs['par']['Ycen']).astype(float)

        R, Phi = __cartesian2polar__(X, Y, **kwargs)
    
        # now do the transformation into the frame of the galaxy
        XPrime = (-1 * R * np.sin(Phi - kwargs['par']['PA']) / 
                  np.cos(kwargs['par']['Incl']))
        YPrime = R * np.cos(Phi - kwargs['par']['PA'])
        
        # Convert these to polar coordinates
        RPrime, PhiPrime = __cartesian2polar__(XPrime, YPrime, **kwargs)
    
        return R, Phi, RPrime, PhiPrime
    
    else:
        
        # not ready for this yet
        raise NotImplementedError('Not ready for this yet...')
        
        
def __cartesian2polar__(X, Y, **kwargs):
    
    """ simple function that converts cartesian coordinates, X and Y,
    to polar coordinates, R and Phi
    """
 
    # R array       
    R = np.sqrt(X**2 + Y**2)
        
    # phi array
    Y[np.where(Y == 0.)] = 1E-99 # fix for divide by 0.
    Phi = np.arctan(-1 * X / Y)
    Phi[Y < 0] = Phi[Y < 0] + np.pi
    Phi[(X > 0) & (Y >= 0)] = Phi[(X > 0) & (Y >= 0)] + np.pi * 2           
    # convert central pix to PA along minor axis (i.e., v=0)
    Phi[(X == 0.) & (Y == 1E-99)]=kwargs['par']['PA'] + np.pi / 2.

    return R, Phi
    

def __get_centralvelocity__(Phi, VDep, **kwargs):
    
    """ This will calculate the line-of-sight velocity of an infinitely 
    thin rotating disk at an inclination (Inc) and position angle (PA).
    """
    
    V_sqrt = (np.sqrt(1. + np.sin(Phi - kwargs['par']['PA'])**2 * 
                      np.tan(kwargs['par']['Incl'])**2))
    V = ((np.cos(Phi - kwargs['par']['PA']) * 
          np.sin(kwargs['par']['Incl']) * VDep / 
          V_sqrt) + kwargs['par']['Vcen'])
    
    return V


def __rhophiz_array__(rotate=True, Cart=False, **kwargs):
    
    # create x, y ,and z arrays
    tshape = np.array(kwargs['shape'])
    shape = (tshape[-1], tshape[-2], np.max(tshape))
    Indices = np.indices(shape, dtype=float)
    
    # translate the array
    Indices[0] = Indices[0] - kwargs['par']['Xcen']
    Indices[1] = Indices[1] - kwargs['par']['Ycen']
    Indices[2] = Indices[2] - int(kwargs['shape'][2]/2.)
    
    # make a cartesian representation for matrix rotation
    CarRep = CartesianRepresentation(Indices, unit=u.pix)
    
    if rotate:
        # rotation due to PA (NOTE different axis definition)
        CarRep = CarRep.transform(rm(kwargs['par']['PA']+np.pi/2, axis='z', 
                                     unit=u.rad))
    
        # rotation due to inclination (NOTE different axis definition)
        CarRep = CarRep.transform(rm(kwargs['par']['Incl'], axis='x', 
                                     unit=u.rad))
    else:
        # rotate by 90 degrees because of definition of PA
        CarRep = CarRep.transform(rm(np.pi/2, axis='z', unit=u.rad))
    
    if Cart:
        return CarRep
    else:
        # now transform to cylindrical coordinates
        CylRep = CylindricalRepresentation.from_cartesian(CarRep)   
        return CylRep.rho.value, CylRep.phi.value, CylRep.z.value
    
    
def __sudophi_array__(**kwargs):
    
    """ This will calculate the angle in the sky frame from the center of the 
    emission. Note that this is not the same as the angle in the rotated frame,
    as this is defined in the rotated frame not the sky frame. It is also not
    the phi in the sky frame, as this is defined from the origin and for this
    sudophi the origin changes as a function of height (z).
    """
    # create x, y ,and z arrays
    tshape = np.array(kwargs['shape'])
    shape = (tshape[-1], tshape[-2], np.max(tshape))
    Indices = np.indices(shape, dtype=float)
    Indices[2] = Indices[2] - int(kwargs['shape'][2]/2.)
    
    # find x0 and y0 for each slice in z and subtract this from the indices
    Indices[0] = (Indices[0] - kwargs['par']['Xcen'] + Indices[2] * 
           np.sin(kwargs['par']['Incl']) * np.cos(kwargs['par']['PA']))
    Indices[1] = (Indices[1] - kwargs['par']['Ycen'] + Indices[2] * 
           np.sin(kwargs['par']['Incl']) * np.sin(kwargs['par']['PA']))
    
    # make a cartesian representation for matrix rotation
    CarRep = CartesianRepresentation(Indices, unit=u.pix)
    
    # rotate by 90 degrees because of definition of PA
    CarRep = CarRep.transform(rm(np.pi/2., axis='z', unit=u.rad))
    
    # now transform to cylindrical coordinates
    CylRep = CylindricalRepresentation.from_cartesian(CarRep)
    
    # change zero point of phi
    phi = CylRep.phi.value
    phi[phi < 0] = phi[phi < 0] + 2*np.pi
    
    return phi


#### Below this line are the definitions of all of the models and profiles

def _ThinDisk_(Convolve=True, **kwargs):
    
    """ This will create a 'thin disk' model from the stored parameters 
    specified in kwargs. 
    """
    
    R, Phi, RPrime, PhiPrime = __get_coordinates__(twoD=True, **kwargs)
        
    # get the radial, velocity, and dispersion maps (these are 2D in 
    # the plane of the sky)
    IMap = eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')\
           (RPrime, kwargs['par']['Rd']) * kwargs['par']['I0']
    VDep = eval('_' + kwargs['mstring']['velocityprofile'][0] + '_')\
           (RPrime, kwargs['par']['Rv']) * kwargs['par']['Vmax']
    VMap = __get_centralvelocity__(Phi, VDep, **kwargs)
    DMap = eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')\
           (RPrime, kwargs['par']['Rv']) * kwargs['par']['Disp']
    # note that VCenter is based on the "sky PA" (Phi)
        
    # convert these maps into 3d matrices
    # also generate a velocity array (Z-array) which contains the
    # z pixel number (i.e., velocity) per slice
    ICube = np.tile(IMap, (kwargs['shape'][-3], 1, 1))
    VCube = np.tile(VMap, (kwargs['shape'][-3], 1, 1))
    DCube = np.tile(DMap, (kwargs['shape'][-3], 1, 1))
    ZCube = np.tile(np.arange(kwargs['shape'][-3])[:,np.newaxis,np.newaxis],
                    (1, kwargs['shape'][-2], kwargs['shape'][-1]))
    
    # create the model
    Model = (ICube * np.exp(-1 * (ZCube - VCube)**2 / (2 * DCube**2)))
    
    if Convolve:
        Model = convolve(Model, kwargs['kernel'])
        
    return Model


def _ThickDisk_(Convolve=True, **kwargs):
    
    """ This will create a 'thick disk' model from the stored parameters 
    specified in kwargs. 
    """
    
    rhoArray, phiArray, zArray = __rhophiz_array__(**kwargs)
    
    # disk profile (in rho and z)
    IMap = (eval('_' + 
                 kwargs['mstring']['intensityprofile'][0] + 
                 '_')(rhoArray, kwargs['par']['Rd']) * 
            eval('_' + 
                 kwargs['mstring']['intensityprofile'][2] + 
                 '_')(np.abs(zArray), kwargs['par']['Zf'] * 
                     kwargs['par']['Rd']) *
            kwargs['par']['I0'])
    # velocity map (3D)
    VDep = eval('_' + kwargs['mstring']['velocityprofile'][0] + '_')\
           (rhoArray, kwargs['par']['Rd']) * kwargs['par']['Vmax']
    Phi = __sudophi_array__(**kwargs)
    VMap = __get_centralvelocity__(Phi, VDep, **kwargs)
    # dispersion map
    DMap = eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')\
           (rhoArray, kwargs['par']['Rd']) * kwargs['par']['Disp']
    
    # now make 4D arrays from these
    I4DArr = np.tile(IMap, (kwargs['shape'][-3],1,1,1))
    V4DArr = np.tile(VMap, (kwargs['shape'][-3],1,1,1))
    D4DArr = np.tile(DMap, (kwargs['shape'][-3],1,1,1))
    
    # the 4D velocity array
    VArr = np.indices(V4DArr.shape)[0]

    # create big array
    FullArr = (I4DArr * np.exp(np.square(VArr - V4DArr) / 
                               (-2 * np.square(D4DArr))))

    # collapse big array along z-axis and swap columns (z,x,y -> z,y,x)
    Model = np.nansum(FullArr, axis=3)
    Model = np.swapaxes(Model, 1, 2)
    
    # Convolve
    if Convolve:
        Model = convolve(Model, kwargs['kernel'])

    return Model


def _Spherical_(Convolve=True, **kwargs):
    
    """ Spherical model (supports only R-dependence)
    """
    
    rhoArray, phiArray, zArray = __rhophiz_array__(**kwargs)
    
    # Intensity Profile
    R3d = np.sqrt(np.square(rhoArray) + np.square(zArray))
    IMap = (eval('_' + 
                 kwargs['mstring']['intensityprofile'][0] + 
                 '_')(R3d, kwargs['par']['Rd']) * kwargs['par']['I0'])
    
    # now set those pixels that are within a cone of p[CANGLE0] equal to zero
    # first rotate the array by two more angles
    CarRep = __rhophiz_array__(Cart=True, **kwargs)
    CarRep = CarRep.transform(rm(kwargs['par']['CANGLE2'], axis='z', unit=u.rad))
    CarRep = CarRep.transform(rm(kwargs['par']['CANGLE1'], axis='x', unit=u.rad))
    #CarRep = CarRep.transform(rm(kwargs['par']['CANGLE2'], axis='z', unit=u.rad))
    CylRep = CylindricalRepresentation.from_cartesian(CarRep)
    
    
    IntArr = np.ones_like(rhoArray)
    mask = np.where(CylRep.rho <= np.abs(CylRep.z * 
                                       np.tan(kwargs['par']['CANGLE0'])))
    IntArr[mask] = 0.0
    
    IntArr[np.where(R3d > 0.6 * kwargs['par']['Rd'])] = 1
    
    # apply truncated intensity cone
    IMap = IMap * IntArr
    
    # Velocity Profile
    VRDep = eval('_' + kwargs['mstring']['velocityprofile'][0] + '_')\
            (R3d, kwargs['par']['Rv']) * kwargs['par']['Vmax']
    Phi = __sudophi_array__(**kwargs)
    VMap = __get_centralvelocity__(Phi, VRDep, **kwargs)    

    # Dispersion Profile
    DMap = eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')\
           (R3d, kwargs['par']['Rd']) * kwargs['par']['Disp']
    
    # now make 4D arrays from these
    I4DArr = np.tile(IMap, (kwargs['shape'][-3],1,1,1))
    V4DArr = np.tile(VMap, (kwargs['shape'][-3],1,1,1))
    D4DArr = np.tile(DMap, (kwargs['shape'][-3],1,1,1))
    
    # the 4D velocity array
    VArr = np.indices(V4DArr.shape)[0]

    # create big array
    FullArr = (I4DArr * np.exp(np.square(VArr - V4DArr) / 
                               (-2 * np.square(D4DArr))))

    # collapse big array along z-axis
    Model = np.nansum(FullArr, axis=3)

    # for consistency swap columns 1 and 2 (z, x, y -> z, y, x)
    Model = np.swapaxes(Model, 1, 2)
    
    # Convolve
    if Convolve:
        Model = convolve(Model, kwargs['kernel'])
        
    return Model


def _Exponential_(X, X0):
    
    return np.exp(-1 * X / X0)
    
            
def _Constant_(X, X0):    
        
    return np.ones_like(X)


def _Step_(X, X0):
    
    Arr = np.ones_like(X)
    Arr[np.where(X > X0)] = 0.
    
    return Arr

def _Power_(X, X0, power=-0.5):
    
    return np.power((X / X0), power)

def _Atan_(X, X0):

    return (2. / np.pi) * np.arctan(X / X0)

def _Custom_(X, X0):
    
    Arr = np.ones_like(X)
    Arr[np.where(X > X0)] = 0.
  
    return  Arr

def _Custom2_(X, X0):
    
    Arr = np.ones_like(X)
    Arr[np.where(X < X0)] = 0.7
  
    return  Arr

def _TestFunc_(X, X0, power=1.7325):
    
    return np.sqrt(np.power(np.tanh(X / 9.738), power) / X)
