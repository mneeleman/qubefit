# several methods are defined here to create a rotation curve
import numpy as np
from qubefit.qfmodels import __get_coordinates__ as gc
import matplotlib.pyplot as plt
from scipy.special import erf
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def velocity_curve(Mom1, PA, Incl, Center, vshift=0, scale=1.,
                   minoraxismask=0.1, bins=None, degrees=True):

    """ this will take a moment-1 image of a line and takes these mean
    velocities and generate the maxmimum rotational velocities for it. This
    is returned as an array (both R and Vmax). It will also take the Vmax
    values and calculate median and 1 sigma values for bins determined by
    bins. It will return these in the form x, y, xerr, yerr (so it can be
    easily displayed using matplotlib.pyplot)
    """

    if bins is None:
        ValueError('Please set the radial bins fo which to calculate' +
                   'the averages')

    if degrees:
        # convert angles
        PA = PA * np.pi / 180.
        Incl = Incl * np.pi / 180.

    # get the deprojected-radius
    par = {'PA': PA, 'Incl': Incl, 'Xcen': Center[0], 'Ycen': Center[1]}
    _, Phi, Rpix, _ = gc(twoD=True, shape=Mom1.data.shape, par=par)
    RPrime = scale * Rpix

    # get the velocities
    Vsqrt = np.sqrt(1 + np.sin(Phi - PA)**2 * np.tan(Incl)**2)
    Vmax = (Mom1.data - vshift) / (np.cos(Phi - PA) * np.sin(Incl)) * Vsqrt

    # mask region near minor axis
    MinorAxisDist = np.abs(np.sin(Phi - PA + np.pi / 2))
    Vmask = np.where(MinorAxisDist < minoraxismask, np.NaN, 1)
    RPrime = RPrime * Vmask
    Vmax = Vmax * Vmask

    # find the median values and uncertainty per radial bin
    binlo = bins[:-1]
    binhi = bins[1:]
    Vmaxlow = Vmaxmed = Vmaxhigh = np.array([])
    for blo, bhi in zip(binlo, binhi):
        Mask = (np.where(RPrime > blo, 1, np.NaN) *
                np.where(RPrime < bhi, 1, np.NaN) *
                np.where(np.isfinite(Vmax), 1, np.NaN))
        perc = (1/2 + 1/2 * erf(np.arange(-1, 2, 1)/np.sqrt(2))) * 100.
        VmaxMasked = (Vmax * Mask).flatten()
        VmaxVals = VmaxMasked[np.isfinite(VmaxMasked)]
        try:
            Values = np.percentile(VmaxVals, perc)
            Vmaxlow = np.append(Vmaxlow, Values[0])
            Vmaxmed = np.append(Vmaxmed, Values[1])
            Vmaxhigh = np.append(Vmaxhigh, Values[2])
        except IndexError:
            Vmaxlow = np.append(Vmaxlow, np.NaN)
            Vmaxmed = np.append(Vmaxmed, np.NaN)
            Vmaxhigh = np.append(Vmaxhigh, np.NaN)

    # return the values used for plt.errorbar
    x = (binlo + binhi) / 2.
    xerr = [x - binlo, binhi - x]
    y = Vmaxmed
    yerr = [Vmaxmed - Vmaxlow, Vmaxhigh - Vmaxmed]

    return x, y, xerr, yerr


def velocity_profile(qube, PA, Incl, Center, vshift=0, scale=1.,
                     minoraxismask=0.1, bins=None, degrees=True,
                     mom1=False, peak=False, plot=False, unc=False, **kwargs):

    """ This function will take in an inclination and angle and convert the
    velocities of the data cube into rotational velocities assuming
    that all of the velocity is constrained wihtin the plane of the galaxy.
    These rotational velocities are then either:
    1) weighted by the intensity to get an intensity-weighted
    rotational velocity as a function of radius.
    2) fitted with a spline function to get the rotational velocities by
    measurign the peak of the intensity.
    optionally one can then get uncertainties from bootstrapping the results.
    Note that this will likely underestimate the uncertainty (especially for
    the peak method) as the uncertainty in the inclination is likely the
    dominant uncertainty.
    """

    if bins is None:
        ValueError('Please set the radial bins fo which to calculate' +
                   'the averages')

    # convert angles
    if degrees:
        PA = PA * np.pi / 180.
        Incl = Incl * np.pi / 180.

    # get the deprojected-radius
    par = {'PA': PA, 'Incl': Incl, 'Xcen': Center[0], 'Ycen': Center[1]}
    _, Phi, Rpix, _ = gc(twoD=True, shape=qube.data.shape, par=par)
    RPrime = scale * Rpix

    # create the velocity cube to calculate the rotation value
    Vcube = np.tile(qube.get_velocity()[:, np.newaxis, np.newaxis] - vshift,
                    (1, qube.shape[-2], qube.shape[-1]))
    Vsqrt = np.sqrt(1 + np.sin(Phi - PA)**2 * np.tan(Incl)**2)
    Vmax = Vcube / (np.cos(Phi - PA) * np.sin(Incl)) * Vsqrt

    # mask region near minor axis
    MinorAxisDist = np.abs(np.sin(Phi - PA + np.pi / 2))
    Vmask = np.where(MinorAxisDist < minoraxismask, np.NaN, 1)
    RPrime = RPrime * Vmask
    Vmax = Vmax * Vmask

    # declare the bin sizes, bin edges, etc.
    nbins = len(bins) - 1
    binlo = bins[:-1]
    binhi = bins[1:]
    V, Vul, Vuh = np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)

    if plot:
        fig, ax = plt.subplots(nbins, 1, figsize=(6, 4 * nbins))

    for blo, bhi, i in zip(binlo, binhi, np.arange(nbins)):
        # create the 3D mask of the radial bins
        Mask = (np.where(RPrime > blo, 1, np.NaN) *
                np.where(RPrime < bhi, 1, np.NaN))
        Mask3D = np.tile(Mask, (qube.shape[0], 1, 1))

        # create flat arrays for the velocity and data
        Dflat = (qube.data * Mask3D).flatten()
        Vflat = (Vmax * Mask3D).flatten()
        Vidx = np.isfinite(Vflat)
        Data = Dflat[Vidx]
        Vel = Vflat[Vidx]

        # calculate the intensity-weighted mean rotational velocity
        if mom1:
            V[i], Vul[i], Vuh[i] = _get_mom1vel_(Data, Vel, unc=unc, **kwargs)

        # calculate the peak rotational velocity using spline fitting
        if peak:
            V[i], Vul[i], Vuh[i] = _get_peakvel_(Data, Vel, unc=unc, **kwargs)

        if plot:
            ax[i].scatter(Vel, Data, marker='.', alpha=0.3, color='steelblue')
            ax[i].set_xlim(-1000, 1000)
            ax[i].axvline(V[i], color='black', ls='-', lw=2)
            ax[i].axvline(Vul[i], color='black', ls=':', lw=2)
            ax[i].axvline(Vuh[i], color='black', ls=':', lw=2)

    if plot:
        plt.show()
        plt.close('all')

    # return the values used for plt.errorbar
    x = (binlo + binhi) / 2.
    y = V
    xerr = [x - binlo, binhi - x]
    yerr = [V - Vul, Vuh - V]

    return x, y, xerr, yerr


def get_dispersioncurve(Mom2, PA, Incl, Center, vshift=0, scale=1.,
                        minoraxismask=0.4, bins=None, degrees=True):

    # get the deprojected-radius
    par = {'PA': PA, 'Incl': Incl, 'Xcen': Center[0],
           'Ycen': Center[1]}
    _, Phi, Rpix, _ = gc(twoD=True, shape=Mom2.data.shape, par=par)
    RPrime = scale * Rpix

    # mask region near minor axis
    MinorAxisDist = np.abs(np.sin(Phi - PA + np.pi / 2))
    Vmask = np.where(MinorAxisDist < minoraxismask, np.NaN, 1)
    RPrime = RPrime * Vmask
    Data = Mom2.data * Vmask

    # find the median values and uncertainty per radial bin
    binlo = bins[:-1]
    binhi = bins[1:]
    Vmaxlow = Vmaxmed = Vmaxhigh = np.array([])
    for blo, bhi in zip(binlo, binhi):
        Mask = (np.where(RPrime > blo, 1, np.NaN) *
                np.where(RPrime < bhi, 1, np.NaN) *
                np.where(np.isfinite(Data), 1, np.NaN))
        perc = (1/2 + 1/2 * erf(np.arange(-1, 2, 1)/np.sqrt(2))) * 100.
        VmaxMasked = (Data * Mask).flatten()
        VmaxVals = VmaxMasked[np.isfinite(VmaxMasked)]
        try:
            Values = np.percentile(VmaxVals, perc)
            Vmaxlow = np.append(Vmaxlow, Values[0])
            Vmaxmed = np.append(Vmaxmed, Values[1])
            Vmaxhigh = np.append(Vmaxhigh, Values[2])
        except IndexError:
            Vmaxlow = np.append(Vmaxlow, np.NaN)
            Vmaxmed = np.append(Vmaxmed, np.NaN)
            Vmaxhigh = np.append(Vmaxhigh, np.NaN)

    # return the values used for plt.errorbar
    x = (binlo + binhi) / 2.
    xerr = np.array([x - binlo, binhi - x])
    y = Vmaxmed
    yerr = np.array([Vmaxmed - Vmaxlow, Vmaxhigh - Vmaxmed])

    return x, y, xerr, yerr


def _get_mom1vel_(Data, Vel, unc=True, **kwargs):

    V = np.nansum(Vel * Data) / np.nansum(Data)

    if unc:
        Vul, Vuh = _velbootstrap_(Data, Vel, mom1=True, **kwargs)
    else:
        Vul, Vuh = 0., 0.

    return V, Vul, Vuh


def _get_peakvel_(Data, Vel, unc=True, **kwargs):

    cs = __spline_fit__(Vel, Data, minval=-1000, maxval=1000, n_knots=10)
    xt = np.arange(-500, 500, 1)
    V = xt[np.argmax(cs.predict(xt))]
    if unc:
        Vul, Vuh = _velbootstrap_(Data, Vel, peak=True, **kwargs)
    else:
        Vul, Vuh = 0., 0.

    return V, Vul, Vuh


def _velbootstrap_(Data, Vel, mom1=False, peak=False, nboot=500):

    VuArr = np.zeros(nboot)
    for i in np.arange(nboot):
        idx = np.random.randint(len(Data), size=len(Data))
        tData = Data[idx]
        tVel = Vel[idx]

        if mom1:
            VuArr[i], _Vul, _Vuh = _get_mom1vel_(tData, tVel, unc=False)
        if peak:
            VuArr[i], _Vunc, _Vuh = _get_peakvel_(tData, tVel, unc=False)

    perc = (1/2 + 1/2 * erf([-1, 1]/np.sqrt(2))) * 100.
    Values = np.percentile(VuArr, perc)

    return Values[0], Values[1]


def __spline_fit__(x, y, minval=None, maxval=None, n_knots=None, knots=None):

    """ helper function for fitting a spline (copied from internet)
    https://stackoverflow.com/questions/51321100/
    python-natural-smoothing-splines

    Get a natural cubic spline model for the data.

    For the knots, give (a) `knots` (as an array) or (b) minval, maxval
    and n_knots.

    If the knots are not directly specified, the resulting knots are equally
    space within the *interior* of (max, min).  That is, the endpoints are
    *not* included as knots.

    Parameters
    ----------
    x: np.array of float
        The input data
    y: np.array of float
        The outpur data
    minval: float
        Minimum of interval containing the knots.
    maxval: float
        Maximum of the interval containing the knots.
    n_knots: positive integer
        The number of knots to create.
    knots: array or list of floats
        The knots.

    Returns
    --------
    model: a model object
        The returned model will have following method:
        - predict(x):
            x is a numpy array. This will return the predicted y-values.
    """

    if knots:
        spline = NaturalCubicSpline(knots=knots)
    else:
        spline = NaturalCubicSpline(max=maxval, min=minval, n_knots=n_knots)

    p = Pipeline([
        ('nat_cubic', spline),
        ('regression', LinearRegression(fit_intercept=True))
    ])

    p.fit(x, y)

    return p


class AbstractSpline(BaseEstimator, TransformerMixin):
    """Base class for all spline basis expansions."""

    def __init__(self, max=None, min=None, n_knots=None, n_params=None,
                 knots=None):
        if knots is None:
            if not n_knots:
                n_knots = self._compute_n_knots(n_params)
            knots = np.linspace(min, max, num=(n_knots + 2))[1:-1]
            max, min = np.max(knots), np.min(knots)
        self.knots = np.asarray(knots)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, *args, **kwargs):
        return self


class NaturalCubicSpline(AbstractSpline):
    """ direct port of spline fitting code from internet.
    https://github.com/madrury/basis-expansions/blob/master/
    basis_expansions/basis_expansions.py
    """

    def _compute_n_knots(self, n_params):
        return n_params

    @property
    def n_params(self):
        return self.n_knots - 1

    def transform(self, X, **transform_params):
        X_spl = self._transform_array(X)
        if isinstance(X, pd.Series):
            col_names = self._make_names(X)
            X_spl = pd.DataFrame(X_spl, columns=col_names, index=X.index)
        return X_spl

    def _make_names(self, X):
        first_name = "{}_spline_linear".format(X.name)
        rest_names = ["{}_spline_{}".format(X.name, idx)
                      for idx in range(self.n_knots - 2)]
        return [first_name] + rest_names

    def _transform_array(self, X, **transform_params):
        X = X.squeeze()
        try:
            X_spl = np.zeros((X.shape[0], self.n_knots - 1))
        except IndexError:  # For arrays with only one element
            X_spl = np.zeros((1, self.n_knots - 1))
        X_spl[:, 0] = X.squeeze()

        def d(knot_idx, x):
            def ppart(t): return np.maximum(0, t)

            def cube(t): return t*t*t
            numerator = (cube(ppart(x - self.knots[knot_idx]))
                         - cube(ppart(x - self.knots[self.n_knots - 1])))
            denominator = self.knots[self.n_knots - 1] - self.knots[knot_idx]
            return numerator / denominator

        for i in range(0, self.n_knots - 2):
            X_spl[:, i+1] = (d(i, X) - d(self.n_knots - 2, X)).squeeze()

        return X_spl
