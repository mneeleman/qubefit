import numpy as np
import astropy.units as u
from qubefit.qubefit import QubeFit


def set_model():

    # Initialize the QubeFit Instance
    DataFile = './WolfeDiskCube.fits'
    Qube = QubeFit.from_fits(DataFile)
    Qube.file = DataFile

    # Trimming the Data Cube
    center, sz, chan = [128, 128], [45, 45], [6, 19]
    xindex = (center[0] - sz[0], center[0] + sz[0] + 1)
    yindex = (center[1] - sz[1], center[1] + sz[1] + 1)
    zindex = (chan[0], chan[1])
    QubeS = Qube.get_slice(xindex=xindex, yindex=yindex, zindex=zindex)

    # Calculating the Variance
    QSig = Qube.calculate_sigma()
    QSig = np.tile(QSig[chan[0]: chan[1], np.newaxis, np.newaxis],
                   (1, 2 * sz[1] + 1, 2 * sz[0] + 1))
    QubeS.variance = np.square(QSig)

    # Defining the Kernel
    QubeS.create_gaussiankernel(channels=[0], lsf_sigma=0.1)

    # Setting the Mask
    QubeS.create_maskarray(sigma=3, fmaskgrow=0.01)

    # Define the Model
    QubeS.modelname = 'ThinDisk'
    QubeS.intensityprofile[0] = 'Exponential'
    QubeS.velocityprofile[0] = 'Constant'
    QubeS.dispersionprofile[0] = 'Constant'
    
    # Parameters and Priors
    PDict = {'Xcen': {'Value': 45.0, 'Unit': u.pix, 'Fixed': False,
                      'Conversion': None,
                      'Dist': 'uniform', 'Dloc': 35, 'Dscale': 20},
             'Ycen': {'Value': 45.0, 'Unit': u.pix, 'Fixed': False,
                      'Conversion': None,
                      'Dist': 'uniform', 'Dloc': 35, 'Dscale': 20},
             'Incl': {'Value': 45.0, 'Unit': u.deg, 'Fixed': False,
                      'Conversion': (180 * u.deg) / (np.pi * u.rad),
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 90},
             'PA': {'Value': 100.0, 'Unit': u.deg, 'Fixed': False,
                    'Conversion': (180 * u.deg) / (np.pi * u.rad),
                    'Dist': 'uniform', 'Dloc': 0, 'Dscale': 360},
             'I0': {'Value': 8.0E-3, 'Unit': u.Jy / u.beam, 'Fixed': False,
                    'Conversion': None,
                    'Dist': 'uniform', 'Dloc': 0, 'Dscale': 1E-1},
             'Rd': {'Value': 1.5, 'Unit': u.kpc, 'Fixed': False,
                    'Conversion': (0.1354 * u.kpc) / (1 * u.pix),
                    'Dist': 'uniform', 'Dloc': 0, 'Dscale': 5},
             'Rv': {'Value': 1.0, 'Unit': u.kpc, 'Fixed': True,  # not used
                    'Conversion': (0.1354 * u.kpc) / (1 * u.pix),
                    'Dist': 'uniform', 'Dloc': 0, 'Dscale': 5},
             'Vmax': {'Value': 250.0, 'Unit': u.km / u.s, 'Fixed': False,
                      'Conversion': (50 * u.km / u.s) / (1 * u.pix),
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 1000},
             'Vcen': {'Value': 6.0, 'Unit': u.pix, 'Fixed': False,
                      'Conversion': None,
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 20},
             'Disp': {'Value': 80.0, 'Unit': u.km/u.s, 'Fixed': False,
                      'Conversion': (50 * u.km / u.s) / (1 * u.pix),
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 300}
             }
    QubeS.load_initialparameters(PDict)

    # Making the Model
    QubeS.create_model()

    return QubeS
