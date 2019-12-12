# Testing the qubefit program: does it run, and does it do OK with
# running in parallel

import numpy as np
from astropy.io import fits
import astropy.units as u
from qubefit.qubefit import QubeFit
from astropy.convolution import convolve


def create_testcube():

    # load the QubeFit structure  and some parameters
    Cube = QubeFit()
    Cube.data = np.zeros((20, 50, 50))
    Cube.beam = {'BMAJ': np.full(20, 1 / 3600),
                 'BMIN': np.full(20, 1 / 3600),
                 'BPA': np.zeros(20),
                 'CHAN': np.arange(20),
                 'POL': np.zeros(20)}

    # the header
    header = fits.Header()
    header['NAXIS3'] = 20.
    header['CRVAL1'] = 180.
    header['CRPIX1'] = 25.0
    header['CDELT1'] = 0.1 / 3600
    header['CRVAL2'] = 0.
    header['CRPIX2'] = 25.0
    header['CDELT2'] = 0.1 / 3600
    header['CRVAL3'] = 4.75134225E11
    header['CRPIX3'] = 10.
    header['CDELT3'] = 60E6
    header['CTYPE3'] = 'FREQ'
    Cube.shape = (20, 50, 50)
    header['RESTFRQ'] = 4.75134225E11
    Cube.header = header

    return Cube


def create_thindiskmodel():

    # crere the test cube
    Cube = create_testcube()

    # create the gaussian kernel
    Cube.create_gaussiankernel(channels=[10], LSFSigma=0.1)

    # load the parameters
    Cube.load_initialparameters(__kwargs_thindiskmodel__())

    # create the model
    Cube.modelname = 'ThinDisk'
    Cube.create_model()

    # now load the model with noise into the cube
    maxSN = 15
    Noise = np.random.randn(Cube.shape[0], Cube.shape[1], Cube.shape[2])
    ConvNoise = convolve(Noise, Cube.kernel)
    Sigma = np.max(Cube.model) / maxSN
    ScaledNoise = ConvNoise / np.max(ConvNoise) * Sigma
    Cube.data = Cube.model + ScaledNoise
    Cube.variance = np.full_like(Cube.data, np.square(Sigma))

    return Cube


def __kwargs_thindiskmodel__():
    kwargs = {'Xcen': {'Value': 25.0, 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 20, 'Dscale': 20},
              'Ycen': {'Value': 25.0, 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 10, 'Dscale': 20},
              'Incl': {'Value': 30.0, 'Unit': u.deg, 'Fixed': False,
                       'Conversion': (180 * u.deg) / (np.pi * u.rad),
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 90},
              'PA': {'Value': 45.0, 'Unit': u.deg, 'Fixed': False,
                     'Conversion': (180 * u.deg) / (np.pi * u.rad),
                     'Dist': 'uniform', 'Dloc': 0, 'Dscale': 360},
              'I0': {'Value': 1, 'Unit': u.mJy / u.beam, 'Fixed': False,
                     'Conversion': None,
                     'Dist': 'uniform', 'Dloc': 0, 'Dscale': 10},
              'Rd': {'Value': 5., 'Unit': u.pix, 'Fixed': False,
                     'Conversion': None,
                     'Dist': 'uniform', 'Dloc': 0, 'Dscale': 20},
              'Rv': {'Value': 5., 'Unit': u.pix, 'Fixed': True,
                     'Conversion': None,
                     'Dist': 'uniform', 'Dloc': 0, 'Dscale': 20},
              'Vmax': {'Value': 5., 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 20},
              'Vcen': {'Value': 10., 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 20},
              'Disp': {'Value': 2., 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 20}}

    return kwargs


def test_qubefit_single():

    """ This test will load a thin disk data set which includes gaussian noise
    and it will try to fit this data set using the qubefit procedure.
    """

    # load the thin disk model
    Cube = create_thindiskmodel()

    # define the mask to use for the fitting
    Cube.create_maskarray(sigma=3, fmaskgrow=0.01, method='ChiSq')

    # run the emcee chain in serial
    sampler = Cube.run_mcmc(nwalkers=50, nruns=50, Nprocesses=1)

    # run the emcee chain in parallel with 'optimum' processes
    sampler = Cube.run_mcmc(nwalkers=50, nruns=50, Nprocesses=None)

    # run the emcee chain in paralel with defined number of processes
    sampler = Cube.run_mcmc(nwalkers=50, nruns=50, Nprocesses=4)

    return sampler
