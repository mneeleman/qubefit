"""
Models that can be used for Qubefit.

Any additional models should be put here and can then be referenced by the
code. If you have a model that could be useful to others, please feel free
to contact me, and I can add the model permanently to the list.
"""
import numpy as np
from astropy.coordinates.matrix_utilities import rotation_matrix as rm
from astropy.coordinates import CartesianRepresentation
from astropy.coordinates import CylindricalRepresentation
from astropy.coordinates import SphericalRepresentation
import astropy.units as u
from astropy.convolution import convolve
import copy


def ThinDisk(**kwargs):
    """
    Create a model of a thin disk.

    This will create an infintely thin disk model from the stored parameters
    specified in kwargs. The thin disk model is described in detail in the
    online documentation and uses and limitations to this model are discussed
    there and in qubefit's reference paper.

    Parameters
    ----------
    **kwargs : Dictionary
        The kwargs dictionary contains all of the information to run the
        fitting procedure. It consists out of several nested dictionaries.
        The first level consists out the keys: {'mstring', 'mcmcmap', 'par',
        'shape', 'data', 'kernel', 'variance', 'maskarray', 'initpar',
        'probmethod', 'kernelarea', 'convolve'}. The kwargs dictionary should
        be generated directly from a fully populated qubefit instance using
        the function kwargs = self.__define_kwargs__(). This extra step is
        necessary for the input structure of emcee. To create the model, the
        important keys are:

        'mstring': Dictionary
            Contains the model name and profiles used for the model. That is
            the 'modelname', 'intensityprofile', 'velocityprofile', and
            'dispersionprofile'. The first is a string with the model name,
            the latter three are list of strings with the name of the profiles
            in each of the three dimensions, e.g., ['Exponential', None,
            'Exponential'].

        'par': Dictionary
            Contains the parameters needed to successfully create the model.
            These are given in the online documentation, but in short they are
            'Xcen', 'Ycen', 'PA', 'Incl', 'Rd', 'I0', 'Rv', 'Vmax', 'Vcen',
            and 'Disp'. Each should be given in intrinsic units. In addition,
            some profiles (e.g., Sersic and Power) require an additional
            parameter, which is given by the optional parameters:
            'IIdx', 'VIdx', and 'DIdx'.

        'shape': tuple
            The shape of the array to be created.

        'kernel': np.ndarray
            Array representation of the PSF. Will be used by astropy.convolve
            to convolve with the model.

        'convolve': Boolean
            If set to true the model cube will be convolved with the kernel.

    Returns
    -------
    Model : np.ndarray
        Array of size kwargs['shape'] with the model that was generated from
        the parameters in kwargs['par'].

    """
    # get coordinates in the plane of the sky (prime) and disk (non-prime).
    RPrime, PhiPrime, R, Phi = __get_coordinates__(twoD=True, **kwargs)

    # get radial, velocity, and dispersion maps (2D in plane of the sky)
    if True:
        if 'IIdx' in kwargs['par'].keys():
            IMap = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
                    (R, kwargs['par']['Rd'], kwargs['par']['IIdx']) *
                    kwargs['par']['I0'])
        else:
            IMap = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
                    (R, kwargs['par']['Rd']) * kwargs['par']['I0'])
            if 'VIdx' in kwargs['par'].keys():
                VDep = (eval('_' + kwargs['mstring']['velocityprofile'][0] +
                             '_')(R, kwargs['par']['Rv'], kwargs['par']
                                  ['VIdx']) * kwargs['par']['Vmax'])
            else:
                VDep = (eval('_' + kwargs['mstring']['velocityprofile'][0] +
                             '_')(R, kwargs['par']['Rv']) *
                        kwargs['par']['Vmax'])
        # note that VMap is based on the "sky angle" (Phi)
        VMap = __get_centralvelocity__(PhiPrime, VDep, **kwargs)
        DMap = (eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')
                (R, kwargs['par']['Rv']) * kwargs['par']['Disp'])

        # convert these maps into 3d arrays
        ICube = np.tile(IMap, (kwargs['shape'][-3], 1, 1))
        VCube = np.tile(VMap, (kwargs['shape'][-3], 1, 1))
        DCube = np.tile(DMap, (kwargs['shape'][-3], 1, 1))

        # create velocity array (in pixel units)
        ZCube = np.indices(kwargs['shape'])[0]
    else:
        profile = kwargs['mstring']['intensityprofile']
        IArray = __getarray__(profile=profile, CP=[RPrime, None, None],
                              C1P=None, C2P=None, C3P=None,
                              scale=[None, None, None],
                              sidx=[None, None, None],
                              separable=True, **kwargs)
        print(IArray)

    # create the model
    Model = (ICube * np.exp(-1 * (ZCube - VCube)**2 / (2 * DCube**2)))

    # Convolve
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model


def DispersionBulge(**kwargs):
    """
    Create a model of a dispersion-dominated bulge.

    This will create a dispersion-dominated bulge model from the stored
    parameters specified in kwargs. The bulge model is described in detail
    in the online documentation and uses and limitations to this model are
    discussed there and in qubefit's reference paper.

    Parameters
    ----------
    **kwargs : Dictionary
        The kwargs dictionary contains all of the information to run the
        fitting procedure. It consists out of several nested dictionaries.
        The first level consists out the keys: {'mstring', 'mcmcmap', 'par',
        'shape', 'data', 'kernel', 'variance', 'maskarray', 'initpar',
        'probmethod', 'kernelarea', 'convolve'}. The kwargs dictionary should
        be generated directly from a fully populated qubefit instance using
        the function kwargs = self.__define_kwargs__(). This extra step is
        necessary for the input structure of emcee. To create the model, the
        important keys are:

        'mstring': Dictionary
            Contains the model name and profiles used for the model. That is
            the 'modelname', 'intensityprofile', 'velocityprofile', and
            'dispersionprofile'. The first is a string with the model name,
            the latter three are list of strings with the name of the profiles
            in each of the three dimensions, e.g., ['Exponential', None,
            'Exponential'].

        'par': Dictionary
            Contains the parameters needed to successfully create the model.
            These are given in the online documentation, but in short they are
            'Xcen', 'Ycen', 'Rd', 'I0', 'Rv', 'Vcen' and 'Disp'. Each should
            be given in intrinsic units. In addition, some profiles
            (e.g., Sersic and Power) require an additional
            parameter, which is given by the optional parameters:
            'IIdx' and 'DIdx'.

        'shape': tuple
            The shape of the array to be created.

        'kernel': np.ndarray
            Array representation of the PSF. Will be used by astropy.convolve
            to convolve with the model.

        'convolve': Boolean
            If set to true the model cube will be convolved with the kernel.

    Returns
    -------
    Model : np.ndarray
        Array of size kwargs['shape'] with the model that was generated from
        the parameters in kwargs['par'].

    """
    # get the polar coordinates in the plane of the sky
    RPrime, PhiPrime = __get_coordinates__(twoD=True, rotate=False, **kwargs)

    # the intensity and dispersion profile
    if 'IIdx' in kwargs['par'].keys():
        IMap = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
                (RPrime, kwargs['par']['Rd'], kwargs['par']['IIdx']) *
                kwargs['par']['I0'])
    else:
        IMap = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
                (RPrime, kwargs['par']['Rd']) * kwargs['par']['I0'])
        DMap = (eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')
                (RPrime, kwargs['par']['Rv']) * kwargs['par']['Disp'])

    # convert these maps into 3d matrices
    ICube = np.tile(IMap, (kwargs['shape'][-3], 1, 1))
    DCube = np.tile(DMap, (kwargs['shape'][-3], 1, 1))
    VCube = np.full(kwargs['shape'], kwargs['par']['Vcen'])

    # create velocity array (in pixel units)
    ZCube = np.indices(kwargs['shape'])[0]

    # create the model
    Model = (ICube * np.exp(-1 * (ZCube - VCube)**2 / (2 * DCube**2)))

    # Convolve
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model


def TwoClumps(**kwargs):
    """
    Create a model of two bulges.

    This model combines two instances of bulges into a single model. This very
    basic model is useful to see if the velocity gradient seen in low
    spatial resolution data can be reproduced using a pair of merging clumps
    that are not rotating themselves.

    Parameters
    ----------
    **kwargs : Dictionary
        The kwargs dictionary contains all of the information to run the
        fitting procedure. It consists out of several nested dictionaries.
        The first level consists out the keys: {'mstring', 'mcmcmap', 'par',
        'shape', 'data', 'kernel', 'variance', 'maskarray', 'initpar',
        'probmethod', 'kernelarea', 'convolve'}. The kwargs dictionary should
        be generated directly from a fully populated qubefit instance using
        the function kwargs = self.__define_kwargs__(). This extra step is
        necessary for the input structure of emcee. To create the model, the
        important keys are:

        'mstring': Dictionary
            Contains the model name and profiles used for the model. That is
            the 'modelname', 'intensityprofile', 'velocityprofile', and
            'dispersionprofile'. The first is a string with the model name,
            the latter three are list of strings with the name of the profiles
            in each of the three dimensions for **both** clumps, e.g.,
            [['Exponential', None, 'None'], ['Exponential', None, 'None']]

        'par': Dictionary
            Contains the parameters needed to successfully create the model.
            These are given in the online documentation, but in short they are
            'Xcen1', 'Ycen1', 'Rd1', 'I01', 'Rv1', 'Vcen1' and 'Disp1' for the
            first clump and similarly for the second clump, 'Xcen2', 'Ycen2',
            'Rd2', 'I02', 'Rv2', 'Vcen2' and 'Disp2'. Each should be given in
            intrinsic units. In addition, some profiles
            (e.g., Sersic and Power) require an additional parameter,
            which is given by the optional parameters, 'IIdx1', 'DIdx1',
            'IIdx2' and 'DIdx2'.

        'shape': tuple
            The shape of the array to be created.

        'kernel': np.ndarray
            Array representation of the PSF. Will be used by astropy.convolve
            to convolve with the model.

        'convolve': Boolean
            If set to true the model cube will be convolved with the kernel.

    Returns
    -------
    Model : np.ndarray
        Array of size kwargs['shape'] with the model that was generated from
        the parameters in kwargs['par'].

    """
    # split up the kwargs into two different keywords for the two clumps
    kwargs1 = copy.deepcopy(kwargs)
    kwargs1['mstring']['intensityprofile'] = \
        kwargs['mstring']['intensityprofile'][0]
    kwargs1['mstring']['dispersionprofile'] = \
        kwargs['mstring']['dispersionprofile'][0]
    kwargs1['par']['Xcen'] = kwargs['par']['Xcen1']
    kwargs1['par']['Ycen'] = kwargs['par']['Ycen1']
    kwargs1['par']['I0'] = kwargs['par']['I01']
    kwargs1['par']['Rd'] = kwargs['par']['Rd1']
    kwargs1['par']['Vcen'] = kwargs['par']['Vcen1']
    kwargs1['par']['Disp'] = kwargs['par']['Disp1']
    kwargs1['par']['Rv'] = kwargs['par']['Rv1']
    if 'IIdx1' in kwargs['par'].keys():
        kwargs1['par']['IIdx'] = kwargs['par']['IIdx1']
    if 'DIdx1' in kwargs['par'].keys():
        kwargs1['par']['DIdx'] = kwargs['par']['DIdx1']
    kwargs1['convolve'] = False

    kwargs2 = copy.deepcopy(kwargs)
    kwargs2['mstring']['intensityprofile'] = \
        kwargs['mstring']['intensityprofile'][1]
    kwargs2['mstring']['dispersionprofile'] = \
        kwargs['mstring']['dispersionprofile'][1]
    kwargs2['par']['Xcen'] = kwargs['par']['Xcen2']
    kwargs2['par']['Ycen'] = kwargs['par']['Ycen2']
    kwargs2['par']['I0'] = kwargs['par']['I02']
    kwargs2['par']['Rd'] = kwargs['par']['Rd2']
    kwargs2['par']['Vcen'] = kwargs['par']['Vcen2']
    kwargs2['par']['Disp'] = kwargs['par']['Disp2']
    kwargs2['par']['Rv'] = kwargs['par']['Rv2']
    if 'IIdx2' in kwargs['par'].keys():
        kwargs2['par']['IIdx'] = kwargs['par']['IIdx2']
    if 'DIdx2' in kwargs['par'].keys():
        kwargs2['par']['DIdx'] = kwargs['par']['DIdx2']
    kwargs2['convolve'] = False

    # create the models individually  (not convolved with kernel)
    Model1 = DispersionBulge(**kwargs1)
    Model2 = DispersionBulge(**kwargs2)

    # combine the two models
    Model = Model1 + Model2

    # convolve
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model


def ThinSpiral(**kwargs):
    """
    Create a model of a thin spiral disk.

    This will create an infintely thin disk model with a spiral pattern
    from the stored parameters specified in kwargs. The thin spiral model
    is described in detail in the online documentation and uses and
    limitations to this model are discussed there. It has been used in
    Chittidi et al. 2020 arXiv200513158

    Parameters
    ----------
    **kwargs : Dictionary
        The kwargs dictionary contains all of the information to run the
        fitting procedure. It consists out of several nested dictionaries.
        The first level consists out the keys: {'mstring', 'mcmcmap', 'par',
        'shape', 'data', 'kernel', 'variance', 'maskarray', 'initpar',
        'probmethod', 'kernelarea', 'convolve'}. The kwargs dictionary should
        be generated directly from a fully populated qubefit instance using
        the function kwargs = self.__define_kwargs__(). This extra step is
        necessary for the input structure of emcee. To create the model, the
        important keys are:

        'mstring': Dictionary
            Contains the model name and profiles used for the model. That is
            the 'modelname', 'intensityprofile', 'velocityprofile', and
            'dispersionprofile'. The first is a string with the model name,
            the latter three are list of strings with the name of the profiles
            in each of the three dimensions, e.g., ['Exponential', None,
            'Exponential'].

        'par': Dictionary
            Contains the parameters needed to successfully create the model.
            These are given in the online documentation, but in short they are
            'Xcen', 'Ycen', 'PA', 'Incl', 'Rd', 'I0', 'Rv', 'Vmax', 'Vcen',
            and 'Disp'. Each should be given in intrinsic units. In addition,
            some profiles (e.g., Sersic and Power) require an additional
            parameter, which is given by the optional parameters:
            'IIdx', 'VIdx', and 'DIdx'.
            The spiral pattern is described by an additonal six parameters,
            'NSpiral', 'Phi0', 'Spcoef', 'Dphi', 'Ispf', and 'Rs'

        'shape': tuple
            The shape of the array to be created.

        'kernel': np.ndarray
            Array representation of the PSF. Will be used by astropy.convolve
            to convolve with the model.

        'convolve': Boolean
            If set to true the model cube will be convolved with the kernel.

    Returns
    -------
    Model : np.ndarray
        Array of size kwargs['shape'] with the model that was generated from
        the parameters in kwargs['par'].

    """
    # get the polar coordinates in the plane of the sky (prime) and
    # in the plane of the disk (non-prime).
    RPrime, PhiPrime, R, Phi = __get_coordinates__(twoD=True, **kwargs)

    # get the radial, velocity, and dispersion maps (these are 2D in
    # the plane of the sky)
    # note that VMap is based on the "sky angle" (Phi)
    if 'IIdx' in kwargs['par'].keys():
        IM1 = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
               (R, kwargs['par']['Rd'], kwargs['par']['IIdx']))
    else:
        IM1 = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
               (R, kwargs['par']['Rd']))

    #  spiral arm density profile
    IM2 = 0
    for idx in np.arange(kwargs['par']['Nspiral']):
        # find the starting phi at R=0
        CPhi0 = (kwargs['par']['Phi0'] +
                 (2 * np.pi) / kwargs['par']['Nspiral'] * idx)
        CPhi = CPhi0 + kwargs['par']['Spcoef'] * R
        CPhi = np.mod(CPhi, 2 * np.pi)
        # CPhi[np.where(CPhi > np.pi)] -= 2 * np.pi
        IM2 += np.exp(-0.5 * (Phi - CPhi)**2 / kwargs['par']['Dphi']**2)

    IM2 *= kwargs['par']['Ispf']
    IM2 *= eval('_Step_')(R, kwargs['par']['Rs'])

    # add the disk and the spiral arm structure
    IMap = IM1 + IM2
    IMap *= kwargs['par']['I0']

    # velocity and dispersion maps
    VDep = (eval('_' + kwargs['mstring']['velocityprofile'][0] + '_')
            (R, kwargs['par']['Rv']) * kwargs['par']['Vmax'])
    VMap = __get_centralvelocity__(Phi, VDep, **kwargs)
    DMap = (eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')
            (R, kwargs['par']['Rv']) * kwargs['par']['Disp'])

    # convert these maps into 3d matrices
    ICube = np.tile(IMap, (kwargs['shape'][-3], 1, 1))
    VCube = np.tile(VMap, (kwargs['shape'][-3], 1, 1))
    DCube = np.tile(DMap, (kwargs['shape'][-3], 1, 1))

    # create velocity array (in pixel units)
    ZCube = np.indices(kwargs['shape'])[0]

    # create the model
    Model = (ICube * np.exp(-1 * (ZCube - VCube)**2 / (2 * DCube**2)))

    # Convolve
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model


# THE FOLLOWING PROFILES HAVE NOT BEEN TESTED AND HAVE SOME BUGS IN THEM
def _ThickDisk(**kwargs):
    """
    Create a model of a thick disk.

    This will create a 'thick disk' model from the stored parameters
    specified in kwargs. Variables that should be defined are: 'Xcen', 'Ycen',
    'PA', 'Incl', 'Rd', 'Zf', 'I0', 'Rv', 'Vmax', 'Disp'.
    """
    # get the cylindrical coordinates in the plane of the disk.
    rhoArray, phiArray, zArray = __getarray__(**kwargs)

    # disk profile (in rho and z)
    IMap = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
            (rhoArray, kwargs['par']['Rd']) *
            eval('_' + kwargs['mstring']['intensityprofile'][2] + '_')
            (np.abs(zArray), kwargs['par']['Zf'] * kwargs['par']['Rd']) *
            kwargs['par']['I0'])
    # velocity map (3D)
    # note that VMap is based on a modified "sky PA" (Phi)
    # if is the sky angle for each slice at high z
    VDep = (eval('_' + kwargs['mstring']['velocityprofile'][0] + '_')
            (rhoArray, kwargs['par']['Rd']) * kwargs['par']['Vmax'])
    Phi = __sudophi_array__(**kwargs)
    VMap = __get_centralvelocity__(Phi, VDep, **kwargs)
    # dispersion map
    DMap = (eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')
            (rhoArray, kwargs['par']['Rd']) * kwargs['par']['Disp'])

    # now make 4D arrays from these
    I4DArr = np.tile(IMap, (kwargs['shape'][-3], 1, 1, 1))
    V4DArr = np.tile(VMap, (kwargs['shape'][-3], 1, 1, 1))
    D4DArr = np.tile(DMap, (kwargs['shape'][-3], 1, 1, 1))

    # the 4D velocity array
    VArr = np.indices(V4DArr.shape)[0]

    # create big array
    FullArr = (I4DArr * np.exp(np.square(VArr - V4DArr) /
                               (-2 * np.square(D4DArr))))

    # collapse big array along z-axis and swap columns (z,x,y -> z,y,x)
    Model = np.nansum(FullArr, axis=3)
    Model = np.swapaxes(Model, 1, 2)

    # Convolve
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model


def _DispersionSphere(**kwargs):

    """ Spherical model with zero mean velocity. The only shift in frequencies
    is due to the dispersion of the model.
    """

    rhoArray, phiArray, zArray = __getarray__(representation='Spherical',
                                              rotate=False, **kwargs)

    # Intensity Profile
    R3d = np.sqrt(np.square(rhoArray) + np.square(zArray))
    IMap = (eval('_' +
                 kwargs['mstring']['intensityprofile'][0] +
                 '_')(R3d, kwargs['par']['Rd']) * kwargs['par']['I0'])

    # Velocity Profile
    VMap = np.full_like(IMap, kwargs['par']['Vcen'])

    # Dispersion Profile
    DMap = (eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')
            (R3d, kwargs['par']['Rd']) * kwargs['par']['Disp'])

    # now make 4D arrays from these
    I4DArr = np.tile(IMap, (kwargs['shape'][-3], 1, 1, 1))
    V4DArr = np.tile(VMap, (kwargs['shape'][-3], 1, 1, 1))
    D4DArr = np.tile(DMap, (kwargs['shape'][-3], 1, 1, 1))

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
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model


def _TwoSpheres(**kwargs):

    # split up the kwargs into two different keyword cases for the two
    # different model spheres.
    kwargs1 = copy.deepcopy(kwargs)
    kwargs1['mstring']['intensityprofile'] = \
        kwargs['mstring']['intensityprofile'][0]
    kwargs1['mstring']['dispersionprofile'] = \
        kwargs['mstring']['dispersionprofile'][0]
    kwargs1['par']['Xcen'] = kwargs['par']['Xcen1']
    kwargs1['par']['Ycen'] = kwargs['par']['Ycen1']
    kwargs1['par']['I0'] = kwargs['par']['I01']
    kwargs1['par']['Rd'] = kwargs['par']['Rd1']
    kwargs1['par']['Vcen'] = kwargs['par']['Vcen1']
    kwargs1['par']['Disp'] = kwargs['par']['Disp1']
    kwargs1['convolve'] = False

    kwargs2 = copy.deepcopy(kwargs)
    kwargs2['mstring']['intensityprofile'] = \
        kwargs['mstring']['intensityprofile'][1]
    kwargs2['mstring']['dispersionprofile'] = \
        kwargs['mstring']['dispersionprofile'][1]
    kwargs2['par']['Xcen'] = kwargs['par']['Xcen2']
    kwargs2['par']['Ycen'] = kwargs['par']['Ycen2']
    kwargs2['par']['I0'] = kwargs['par']['I02']
    kwargs2['par']['Rd'] = kwargs['par']['Rd2']
    kwargs2['par']['Vcen'] = kwargs['par']['Vcen2']
    kwargs2['par']['Disp'] = kwargs['par']['Disp2']
    kwargs2['convolve'] = False

    Model1 = _DispersionSphere(**kwargs1)
    Model2 = _DispersionSphere(**kwargs2)

    # simply add the two models
    Model = Model1 + Model2

    # Convolve
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model


def _JetSphere(Convolve=True, **kwargs):

    """ Spherical model with 'jet' (supports only R-dependence)
    """

    rhoArray, phiArray, zArray = __getarray__(**kwargs)

    # Intensity Profile
    R3d = np.sqrt(np.square(rhoArray) + np.square(zArray))
    IMap = (eval('_' +
                 kwargs['mstring']['intensityprofile'][0] +
                 '_')(R3d, kwargs['par']['Rd']) * kwargs['par']['I0'])

    # now set those pixels that are within a cone of p[CANGLE0] equal to zero
    # first rotate the array by two more angles
    CarRep = __getarray__(representation='Cartesian', **kwargs)
    CarRep = CarRep.transform(rm(kwargs['par']['CANGLE2'], axis='z',
                                 unit=u.rad))
    CarRep = CarRep.transform(rm(kwargs['par']['CANGLE1'], axis='x',
                                 unit=u.rad))
    # CarRep = CarRep.transform(rm(kwargs['par']['CANGLE2'], axis='z',
    #                              unit=u.rad))

    CylRep = CylindricalRepresentation.from_cartesian(CarRep)

    IntArr = np.ones_like(rhoArray)
    mask = np.where(CylRep.rho <=
                    np.abs(CylRep.z * np.tan(kwargs['par']['CANGLE0'])))
    IntArr[mask] = 0.0

    IntArr[np.where(R3d > 0.6 * kwargs['par']['Rd'])] = 1

    # apply truncated intensity cone
    IMap = IMap * IntArr

    # Velocity Profile
    VRDep = (eval('_' + kwargs['mstring']['velocityprofile'][0] + '_')
             (R3d, kwargs['par']['Rv']) * kwargs['par']['Vmax'])
    Phi = __sudophi_array__(**kwargs)
    VMap = __get_centralvelocity__(Phi, VRDep, **kwargs)

    # Dispersion Profile
    DMap = (eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')
            (R3d, kwargs['par']['Rd']) * kwargs['par']['Disp'])

    # now make 4D arrays from these
    I4DArr = np.tile(IMap, (kwargs['shape'][-3], 1, 1, 1))
    V4DArr = np.tile(VMap, (kwargs['shape'][-3], 1, 1, 1))
    D4DArr = np.tile(DMap, (kwargs['shape'][-3], 1, 1, 1))

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


def _SpiralGalaxy(Convolve=True, **kwargs):

    """ This is a model of a thick disk galaxy that contains spiral arms.
    """

    # get the cylindrical coordinates in the plane of the disk.
    rhoArray, phiArray, zArray = __getarray__(**kwargs)

    # disk profile (in rho and z)
    IM1 = (eval('_' + kwargs['mstring']['intensityprofile'][0] + '_')
           (rhoArray, kwargs['par']['Rd']))

    #  spiral arm density profile
    IM2 = 0
    for idx in np.arange(kwargs['par']['Nspiral']):
        # find the starting phi at R=0
        CPhi0 = (kwargs['par']['Phi0'] +
                 (2 * np.pi) / kwargs['par']['Nspiral'] * idx)
        CPhi = CPhi0 + kwargs['par']['Spcoef'] * rhoArray
        CPhi = np.mod(CPhi, 2 * np.pi)
        CPhi[np.where(CPhi > np.pi)] -= 2 * np.pi
        IM2 += np.exp(-0.5 * (phiArray - CPhi)**2 / kwargs['par']['Dphi']**2)

    IM2 *= kwargs['par']['Ispf']
    IM2 *= eval('_Step_')(rhoArray, kwargs['par']['Rs'])

    # add the disk and the spiral arm structure and mux by the z-profile
    IMap = IM1 + IM2
    IMap = (IMap * eval('_' + kwargs['mstring']['intensityprofile'][2] + '_')
            (np.abs(zArray), kwargs['par']['Zf'] * kwargs['par']['Rd']) *
            kwargs['par']['I0'])

    # velocity map (3D)
    # note that VMap is based on a modified "sky PA" (Phi)
    # if is the sky angle for each slice at high z
    VDep = (eval('_' + kwargs['mstring']['velocityprofile'][0] + '_')
            (rhoArray, kwargs['par']['Rd']) * kwargs['par']['Vmax'])
    Phi = __sudophi_array__(**kwargs)
    VMap = __get_centralvelocity__(Phi, VDep, **kwargs)

    # dispersion map
    DMap = (eval('_' + kwargs['mstring']['dispersionprofile'][0] + '_')
            (rhoArray, kwargs['par']['Rd']) * kwargs['par']['Disp'])

    # now make 4D arrays from these
    I4DArr = np.tile(IMap, (kwargs['shape'][-3], 1, 1, 1))
    V4DArr = np.tile(VMap, (kwargs['shape'][-3], 1, 1, 1))
    D4DArr = np.tile(DMap, (kwargs['shape'][-3], 1, 1, 1))

    # the 4D velocity array
    VArr = np.indices(V4DArr.shape)[0]

    # create big array
    FullArr = (I4DArr * np.exp(np.square(VArr - V4DArr) /
                               (-2 * np.square(D4DArr))))

    # collapse big array along z-axis and swap columns (z,x,y -> z,y,x)
    Model = np.nansum(FullArr, axis=3)
    Model = np.swapaxes(Model, 1, 2)

    # Convolve
    if kwargs['convolve']:
        Model = convolve(Model, kwargs['kernel'])

    return Model

##############################################################
# Available profiles for Intensity, Velocity, and Dispersion #


def _Exponential_(X, X0, *args):

    return np.exp(-1 * X / X0)


def _Constant_(X, X0, *args):

    return np.ones_like(X)


def _Step_(X, X0, *args):

    Arr = np.ones_like(X)
    Arr[np.where(X > X0)] = 0.

    return Arr


def _Power_(X, X0, N, *args):

    return np.power((X / X0), N)


def _Atan_(X, X0, *args):

    return (2. / np.pi) * np.arctan(X / X0)


def _Delta_(X, X0, *args):  # not recommended to use this profile

    Arr = np.zeros_like(X)
    Arr[np.where(X == X0)] = 1

    return Arr


def _Sersic_(X, X0, N, *args):

    return np.exp(-(2.*N-(1./3.))*(((X/X0)**N)-1))


def _Sech2_(X, X0, *args):

    return (1. / np.cos(-1. * X / X0))**2


def _ExpConst_(X, X0, *args):

    return (1 + 1.5 * np.exp(-3 * X / X0))


def _Custom_(X, X0, *args):

    return (np.power((X / X0), -0.5) * 0.3888 + 1.)
##############################################################


###########################################
# Under the hood functions for the models #


def __get_coordinates__(twoD=False, rotate=True, **kwargs):

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
        X = (np.mgrid[0:kwargs['shape'][-2], 0: kwargs['shape'][-1]][-1] -
             kwargs['par']['Xcen']).astype(float)
        Y = (np.mgrid[0:kwargs['shape'][-2], 0: kwargs['shape'][-1]][-2] -
             kwargs['par']['Ycen']).astype(float)

        R, Phi = __cartesian2polar__(X, Y)

        if rotate:
            # now do the transformation into the frame of the galaxy
            XPrime = (-1 * R * np.sin(Phi - kwargs['par']['PA']) /
                      np.cos(kwargs['par']['Incl']))
            YPrime = R * np.cos(Phi - kwargs['par']['PA'])

            # Convert these to polar coordinates
            RPrime, PhiPrime = __cartesian2polar__(XPrime, YPrime)

            return R, Phi, RPrime, PhiPrime
        else:
            return R, Phi
    else:

        # not ready for this yet
        raise NotImplementedError('Not ready for this yet...')


def __cartesian2polar__(X, Y):

    """ simple function that converts cartesian coordinates, X and Y,
    to polar coordinates, R and Phi
    """

    # R array
    R = np.sqrt(X**2 + Y**2)

    # phi array
    Y[np.where(Y == 0.)] = 1E-99  # fix for divide by 0.
    Phi = np.arctan(-1 * X / Y)
    Phi[Y < 0] = Phi[Y < 0] + np.pi
    Phi[(X > 0) & (Y >= 0)] = Phi[(X > 0) & (Y >= 0)] + np.pi * 2

    return R, Phi


def __get_centralvelocity__(Phi, VDep, **kwargs):

    """ This will calculate the line-of-sight velocity of an infinitely
    thin rotating disk at an inclination ('Incl') and position angle ('PA').
    This formula is used in Neeleman et al. (2016) and also in
    H.-W.Chen et al. (2005).
    """

    V_sqrt = (np.sqrt(1. + np.sin(Phi - kwargs['par']['PA'])**2 *
                      np.tan(kwargs['par']['Incl'])**2))
    V = ((np.cos(Phi - kwargs['par']['PA']) *
          np.sin(kwargs['par']['Incl']) * VDep /
          V_sqrt) + kwargs['par']['Vcen'])

    return V


def __getcoordinatearrays__(rotate=True, representation='Cylindrical',
                            **kwargs):

    """ This function will create a 2D/3D array from the requested shape in
    kwargs['shape']. It can either return this array into cartesian,
    polar/cylindrical or spherical coordinates. Using the optional rotate
    keyword the array can also be rotated into the plane of the sky. This
    requires the position angle, 'PA', and the inclinatio, 'Incl'.

    Parameters
    ----------
    rotate : 'True' | 'False'
        This will either return the rotated or non-rotated array.
    representation : 'Cylindrical' | 'Cartesian' | 'Spherical'
        Representation to use for the returned array.
    Returns
    -------
    2 or 3 numpy arrays corresponding to the rho, phi, (z) array, the x, y, (z)
    array or the rho, lon, (lat array).
    """

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

    # the representation to use and the return values
    if representation == 'Cartesian':
        return CarRep.x.value, CarRep.y.value, CarRep.z.value
    elif representation == 'Cylindrical':
        Rep = CylindricalRepresentation.from_cartesian(CarRep)
        return Rep.rho.value, Rep.phi.value, Rep.z.value
    elif representation == 'Spherical':
        Rep = SphericalRepresentation.from_cartesian(CarRep)
        return Rep.distance.value, Rep.lon.value, Rep.lat.value


def __sudophi_array__(**kwargs):
    """
    Create array of sudophi, angle in sky frame w.r.t. to center of rotation.

    This will calculate the angle in the sky frame from the center of the
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


def __getarray__(profile=None, CPrime=[None, None, None],
                 scale=[None, None, None], sidx=[None, None, None],
                 separable=True, **kwargs):

    if separable:
        first = True
        for idx, CP in enumerate(CPrime):
            if CP is not None:
                if scale[idx] is None:
                    raise ValueError('__getintensityarray__: Set the scale ' +
                                     'for coordinate index-{}'.format(idx))
                tArray = (eval('_' + profile[idx] + '_')
                          (CP, scale[idx], sidx[idx]))
                if first:
                    Array = tArray
                    first = False
                else:
                    Array *= tArray
    else:
        raise NotImplementedError('__getintensityarray__: Non separable ' +
                                  'profiles have not been implemented')

    return Array
