import numpy as np
import astropy.units as u
from qubefit.qubefit import QubeFit


def set_model():

    # Initialize the QubeFit Instance
    DataFile = './WolfeDiskCube.fits'
    Qube = QubeFit.from_fits(DataFile)
    Qube.file = DataFile

    # Trimming the Data Cube
    center, sz, chan = [507, 513], [45, 45], [15, 40]
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
    QubeS.create_gaussiankernel(channels=[0], LSFSigma=0.1)

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
             'Rd': {'Value': 1.0, 'Unit': u.kpc, 'Fixed': False,
                    'Conversion': (0.1354 * u.kpc) / (1 * u.pix),
                    'Dist': 'uniform', 'Dloc': 0, 'Dscale': 5},
             'Rv': {'Value': 1.0, 'Unit': u.kpc, 'Fixed': True,  # not used
                    'Conversion': (0.1354 * u.kpc) / (1 * u.pix),
                    'Dist': 'uniform', 'Dloc': 0, 'Dscale': 5},
             'Vmax': {'Value': 250.0, 'Unit': u.km / u.s, 'Fixed': False,
                      'Conversion': (25 * u.km / u.s) / (1 * u.pix),
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 1000},
             'Vcen': {'Value': 12.0, 'Unit': u.pix, 'Fixed': False,
                      'Conversion': None,
                      'Dist': 'uniform', 'Dloc': 4, 'Dscale': 20},
             'Disp': {'Value': 100.0, 'Unit': u.km/u.s, 'Fixed': False,
                      'Conversion': (25 * u.km / u.s) / (1 * u.pix),
                      'Dist': 'uniform', 'Dloc': 0, 'Dscale': 300}
             }
    QubeS.load_initialparameters(PDict)

    # Making the Model
    QubeS.create_model()

    return QubeS


def run_mcmcchain(nwalkers=20, nruns=50, chainfile='ThinDisk.npy'):

    # the model cube
    QubeS = set_model()

    # run the mcmc
    sampler = QubeS.run_mcmc(nwalkers=nwalkers, nruns=nruns, nproc=6)

    # save the chain (until we know what to do with it)
    Chain = np.zeros((nwalkers, nruns, 10))
    Chain[:, :, :9] = sampler.chain
    Chain[:, :, 9] = sampler.lnprobability
    np.save(chainfile, Chain)

    return sampler
