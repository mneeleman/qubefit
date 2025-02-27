import numpy as np
from scipy.integrate import quad
from scipy import optimize
import astropy.units as u
import astropy.constants as const
from astropy.cosmology import FlatLambdaCDM
from astropy.modeling.physical_models import BlackBody


def calculate_ltir(flux, nu, z, cosmo=None, do_cmbcorrection=True, t_dust=47*u.K, **kwargs):
    """ This function will calculate the TIR luminosity
    needed to match the measured flux density (flux) at
    the given frequency (nu) for an object at redshift (z).
    It assumes the spectrum satisfies a modified black body
    spectrum with parameters, T(dust), alpha, beta which has
    been normalized on the interval given by (normalize).
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) if cosmo is None else cosmo
    nu = __get_nu__(nu) if type(nu) is str else nu
    scale = flux / __mbb__(nu, t_dust=t_dust, **kwargs)
    fcmb = 1.0 if not do_cmbcorrection else __cmbcor__(z, nu, t_dust)
    tir = 4 * np.pi * cosmo.luminosity_distance(z)**2 * scale / (1+z) /fcmb
    return tir.cgs


def calculate_lline(sdv, nu, z, cosmo=None):
    """ This function will calculate the total line luminosity 
    given the velocity-integrated flux density (sdv) the rest
    frequency of the line (nu) and the redshift of the line (z). 
    It uses the equation in i.e., Solomon et al. 1997 ApJ, 478, 144
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) if cosmo is None else cosmo
    nu = __get_nu__(nu) if type(nu) is str else nu
    lline = (4 * np.pi / const.c) * sdv * nu * cosmo.luminosity_distance(z)**2 / (1 + z)
    return lline.cgs


def calculate_sdvline(lline, nu, z, cosmo=None, prime=False):
    """ This function will calculate the velocity-integrated 
    flux density (sdv) from the line luminosity using the 
    equation given Solomon et al. 1997, ApJ, 478, 144. This is 
    simply the inverse of the calculate_lline function
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) if cosmo is None else cosmo
    nu = __get_nu__(nu) if type(nu) is str else nu
    # line luminosity in units of erg/s or equivalent
    if prime:
        lline = lline * u.K*u.km*u.s**-1*u.pc**2 if type(lline) is float else lline
        sdv = lline * ((2 * const.k_B) / const.c**2) * (1 + z) * nu**2 / cosmo.luminosity_distance(z)**2
    else:
        sdv = lline * (1 + z) / ((4 * np.pi / const.c) * nu * cosmo.luminosity_distance(z)**2)
    return sdv.to(u.Jy*u.km/u.s)
    

def calculate_lprimeline(sdv, nu, z, cosmo=None):
    """ This function will calculate the total line luminosity 
    given the velocity-integrated flux density (sdv) the rest
    wavelength of the line (nu) and the redshift of the line (z). 
    It uses the equation in i.e., Solomon et al. 1997 ApJ, 478, 144
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) if cosmo is None else cosmo
    nu = __get_nu__(nu) if type(nu) is str else nu
    lprimeline = (const.c**2 / (2 * const.k_B)) * sdv * cosmo.luminosity_distance(z)**2 / ((1 + z) * nu**2)
    return lprimeline.to(u.K*u.km*u.s**-1*u.pc**2)


def calculate_lspecline(snu, nu, z, cosmo=None):
    """ This function will calculate the spectral line luminosity 
    given the flux measurement (s) the rest wavelength of the 
    line (nu) and the redshift of the line (z).
    """
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) if cosmo is None else cosmo
    nu = __get_nu__(nu) if type(nu) is str else nu
    lspec = 4 * np.pi * cosmo.luminosity_distance(z)**2 * snu * nu / (1 + z)
    return lspec.to(u.erg / u.s)


def calculate_moleculargasmass(lcoprime, trans=1, lratio=None,
                               alpha_co=4.3 * u.M_sun * (u.K * u.km * u.s ** -1 * u.pc ** 2) ** -1,
                               **kwargs):
    """ This function takes in a luminosity (in K km s-1 pc2) and 
    converts it to a molecular mass. If the line ratio is not given,
    it will convert the luminosity of the transition given to a 
    CO(1-0) luminosity assuming the conversion factors in
    Carilli & Walter 2013 ARAA, 51, 105. Optionally, if the ltir keyword
    is set it will use the FIR-to-LCO relationships defined in 
    Greve et al. 2014, ApJ, 794, 142 or Kamenetzky et al.
    2016, ApJ, 829, 93 (the latter being experimental). Default alpha_CO
    is 4.3 Msun/(K km s-1 pc2)-1 (units are optional on the conversion 
    factor for ease of use)
    """
    lratio = __line_ratio__(trans=trans, **kwargs) if lratio is None else lratio
    alpha_co = alpha_co * u.M_sun * (u.K * u.km * u.s ** -1 * u.pc ** 2) ** -1 if type(alpha_co) is float else alpha_co
    return lcoprime / lratio * alpha_co


def calculate_dustmass(snu, nu, z, t_dust=47*u.K, beta=1.6, cosmo=None, kappa0=2.64*u.m**2/u.kg):
    if cosmo is None:
        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dl = cosmo.luminosity_distance(z)
    bb = BlackBody(t_dust)(nu) * u.rad**2
    kappa = kappa0 * (nu * 125*u.um / const.c)**beta
    return (snu * dl**2 / ((1 + z) * kappa * bb)).to(u.Msun)


def __mbb__(nu, t_dust=47 * u.K, alpha=2.0, beta=1.6, normalize=(8 * u.um, 1000 * u.um),
            rootguess=2E12 * u.s ** -1):
    """ This function will return a modified black body
    spectrum with dust temperature (T), Raleigh-Jeans slope
    (beta), and mid-IR excess slope (alpha) at the requested
    frequency/frequencies, normalized so that the area under the
    curve is 1 for the range specified with normalize.
    """
    nulim = ((const.c / normalize[0]).cgs, (const.c / normalize[1]).cgs)
    if alpha is None:
        norm = quad(__unscaled_graybody__, nulim[1].value, nulim[0].value, (t_dust.value, beta))
        mbb = __unscaled_graybody__(nu.cgs.value, t_dust.value, beta) / norm[0]
    else:
        sol = optimize.root(__mbb_root__, rootguess.value, args=(t_dust.value, alpha, beta))
        if not sol.success:
            raise ValueError('No roots found!')
        nu_c = sol.x
        scale = __unscaled_graybody__(nu_c, t_dust.value, beta) / nu_c ** (-1 * alpha)
        norm = quad(__unscaled_mbb__, nulim[1].value, nulim[0].value, (t_dust.value, alpha, beta, nu_c, scale))
        mbb = __unscaled_mbb__(nu.cgs.value, t_dust.value, alpha, beta, nu_c, scale) / norm[0]
    return mbb * u.s


def __line_ratio__(trans=1, ltir=None, k16=False, ulirg=False):
    """ This function finds the line ratio for a given transition. It 
    defaults to the Papadopoulos et al. 2012 MNRAS 426, 2601 for ULIRGs
    with additional data from Carilli & Walter 2013 ARAA, 51, 105 data and
    Daddi et al 2015 for 'normal' galaxies (bzK). If TIR
    is given, then it will default to the Greve et al, 2014 ApJ, 794, 142
    results unless K16 is set to True (Kamenetzsky et al. 2016, ApJ, 829, 93/
    The latter is somewhat experimental.
    """
    if ltir is None:
        lr_cw = [1.0, 0.90, 0.62, 0.46, 0.39] if ulirg else [1.0, 0.76, 0.42, 0.31, 0.08]
        line_ratio = lr_cw[trans - 1]
    else:
        if k16:  # ULIRG K16
            if ltir < 1E11*u.L_sun:  # NON-ULIRGs K16
                a = [1.05, 1.12, 1.05, 1.09, 1.01, 1.01, 1.00, 1.07, 1.06, 1.12, 1.10, 1.03, 1.23]
                b = [0.8, 0.4, 0.9, 1.2, 2.1, 2.3, 2.8, 2.5, 2.7, 2.5, 2.7, 3.4, 2.2]
            else:
                a = [1.15, 0.66, 0.94, 0.68, 0.96, 1.02, 0.92, 0.91, 0.92, 0.83, 0.86, 0.79, 0.85]
                b = [0.2, 4.9, 2.3, 4.9, 2.7, 2.4, 3.5, 3.7, 3.8, 4.7, 4.7, 5.3, 5.0]
        else:
            a = [1.00, 1.05, 1.00, 1.08, 0.97, 0.95, 0.87, 0.66, 0.85, 0.69, 0.61, 0.55, 0.51]
            b = [2.0, 1.7, 2.2, 1.5, 2.8, 3.2, 4.1, 6.1, 4.6, 6.1, 6.8, 7.5, 7.9]
        line_ratio = ltir.to(u.L_sun).value**(a[trans-1]**-1 - a[0]**-1) * 10**(b[0]/a[0] - b[trans-1]/a[trans-1])
    return line_ratio


def __unscaled_graybody__(nu, t_dust, beta):
    x = (const.h.cgs.value * nu) / (const.k_B.cgs.value * t_dust)
    return nu**(beta + 3) / (np.exp(x) - 1)


def __mbb_root__(nu, t_dust, alpha, beta):
    x = (const.h.cgs.value * nu) / (const.k_B.cgs.value * t_dust)
    return x * np.exp(x) / (np.exp(x) - 1) - 3 - alpha - beta


def __unscaled_mbb__(nu, t_dust, alpha, beta, nu_c, scale):
    if type(nu) is float or type(nu) is np.float64:
        mbb = __unscaled_graybody__(nu, t_dust, beta) if nu < nu_c else scale*nu**(-1*alpha)
    else:
        mbb = np.zeros_like(nu)
        mbb[nu < nu_c] = __unscaled_graybody__(nu[nu < nu_c], t_dust, beta)
        mbb[nu >= nu_c] = scale*nu[nu >= nu_c]**(-1*alpha)
    return mbb


def __get_nu__(nu):
    getnu = {'CII': 1900.5369 * u.GHz, 'CO10': 115.2712 * u.GHz, 'CO21': 230.5380 * u.GHz,
             'CO32': 345.7960 * u.GHz, 'CO43': 461.0408 * u.GHz, 'CO54': 576.2679 * u.GHz}
    return getnu[nu]


def __cmbcor__(z, nu, t_dust):
    return 1 - (BlackBody(2.72548 * u.K * (1 + z))(nu) / BlackBody(t_dust)(nu)).cgs.value

    
