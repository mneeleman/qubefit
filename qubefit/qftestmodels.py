# test the models defined in qfmodels

import numpy as np
from astropy.io import fits
import astropy.units as u

from qfmodels import *
from qubefit import QubeFit


def load_testcube():

    # load the QubeFit structure  and some parameters
    TCube = QubeFit()
    TCube.data = np.zeros((20, 50, 50))
    TCube.beam = {'BMAJ': np.full(20, 1.0),
                  'BMIN': np.full(20, 1.0),
                  'BPA': np.zeros(20),
                  'CHAN': np.arange(20),
                  'POL': np.zeros(20)}

    # the header
    header = fits.Header()
    header['NAXIS3'] = 20.
    header['CRVAL1'] = 180.
    header['CRPIX1'] = 25.0
    header['CDELT1'] = 0.1
    header['CRVAL2'] = 0.
    header['CRPIX2'] = 25.0
    header['CDELT2'] = 0.1
    header['CRVAL3'] = 4.75134225E11
    header['CRPIX3'] = 10.
    header['CDELT3'] = 60E6
    header['CTYPE3'] = 'FREQ'
    TCube.shape = (20, 50, 50)
    header['RESTFRQ'] = 4.75134225E11
    TCube.header = header

    return TCube


def kwargs_dispersionsphere():

    kwargs = {'Xcen': {'Value': 30.0, 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 20, 'Dscale': 20},
              'Ycen': {'Value': 20.0, 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 10, 'Dscale': 20},
              'I0': {'Value': 1.0E-3, 'Unit': u.Jy / u.beam, 'Fixed': False,
                     'Conversion': None,
                     'Dist': 'uniform', 'Dloc': 0, 'Dscale': 1E-2},
              'Rd': {'Value': 0.2, 'Unit': u.kpc, 'Fixed': False,
                     'Conversion': (0.2 * u.kpc) / (1 * u.pix),
                     'Dist': 'uniform', 'Dloc': 0, 'Dscale': 5},
              'Vcen': {'Value': 10.0, 'Unit': u.pix, 'Fixed': False,
                       'Conversion': None,
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 10},
              'Disp': {'Value': 100.0, 'Unit': u.km/u.s, 'Fixed': False,
                       'Conversion': (37.86 * u.km / u.s) / (1 * u.pix),
                       'Dist': 'uniform', 'Dloc': 0, 'Dscale': 200}
              }

    return kwargs


def kwargs_twospheres():

    kwargs = {'Xcen1': {'Value': 30.0, 'Unit': u.pix, 'Fixed': False,
                        'Conversion': None,
                        'Dist': 'uniform', 'Dloc': 20, 'Dscale': 20},
              'Ycen1': {'Value': 20.0, 'Unit': u.pix, 'Fixed': False,
                        'Conversion': None,
                        'Dist': 'uniform', 'Dloc': 10, 'Dscale': 20},
              'I01': {'Value': 1.0E-3, 'Unit': u.Jy / u.beam, 'Fixed': False,
                      'Conversion': None,
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 1E-2},
              'Rd1': {'Value': 1.0, 'Unit': u.kpc, 'Fixed': False,
                      'Conversion': (0.2 * u.kpc) / (1 * u.pix),
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 5},
              'Vcen1': {'Value': 10.0, 'Unit': u.pix, 'Fixed': False,
                        'Conversion': None,
                        'Dist': 'uniform', 'Dloc': 0, 'Dscale': 10},
              'Disp1': {'Value': 100.0, 'Unit': u.km/u.s, 'Fixed': False,
                        'Conversion': (37.86 * u.km / u.s) / (1 * u.pix),
                        'Dist': 'uniform', 'Dloc': 0, 'Dscale': 200},
              'Xcen2': {'Value': 20.0, 'Unit': u.pix, 'Fixed': False,
                        'Conversion': None,
                        'Dist': 'uniform', 'Dloc': 20, 'Dscale': 20},
              'Ycen2': {'Value': 30.0, 'Unit': u.pix, 'Fixed': False,
                        'Conversion': None,
                        'Dist': 'uniform', 'Dloc': 10, 'Dscale': 20},
              'I02': {'Value': 2.0E-3, 'Unit': u.Jy / u.beam, 'Fixed': False,
                      'Conversion': None,
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 1E-2},
              'Rd2': {'Value': 0.8, 'Unit': u.kpc, 'Fixed': False,
                      'Conversion': (0.2 * u.kpc) / (1 * u.pix),
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 5},
              'Vcen2': {'Value': 7.0, 'Unit': u.pix, 'Fixed': False,
                        'Conversion': None,
                        'Dist': 'uniform', 'Dloc': 0, 'Dscale': 10},
              'Disp2': {'Value': 100.0, 'Unit': u.km/u.s, 'Fixed': False,
                        'Conversion': (37.86 * u.km / u.s) / (1 * u.pix),
                        'Dist': 'uniform', 'Dloc': 0, 'Dscale': 200}
              }

    return kwargs


def test_dispersionsphere():

    Tcube = load_testcube()

    # create the gaussian kernel
    Tcube.create_gaussiankernel(channels=[10], LSFSigma=0.1)

    # load the parameters
    Tcube.load_initialparameters(kwargs_dispersionsphere())

    # create the model
    Tcube.modelname = 'DispersionSphere'
    Tcube.create_model()

    return Tcube


def test_twospheres():

    Tcube = load_testcube()

    # create the gaussian kernel
    Tcube.create_gaussiankernel(channels=[10], LSFSigma=0.1)

    # load the parameters
    Tcube.load_initialparameters(kwargs_twospheres())

    # create the model
    Tcube.modelname = 'TwoSpheres'
    Tcube.intensityprofile = [['Exponential', None, 'Delta'],
                              ['Exponential', None, 'Delta']]
    Tcube.dispersionprofile = [['Constant', None, None],
                               ['Constant', None, None]]
    Tcube.create_model()

    return Tcube
