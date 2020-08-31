# modules
import numpy as np
import copy
from astropy.io import fits
from astropy.modeling import models, fitting
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import astropy.units as u
from astropy import constants as const
from astropy.table import Table
from scipy.ndimage.interpolation import rotate
from astropy.stats import sigma_clip


class Qube(object):

    """ Class for the (hopefully) easy handeling of a variety of data cubes
    primarily tested for sub-mm. In particular, CASA and AIPS fits files can
    be treated in a similar manner.
    """

    @classmethod
    def from_fits(cls, fitsfile, extension=0, **kwargs):

        """ Read in a fits file. This is the primary way to load in a data
        cube.
        """

        # initiate (currently blank)
        self = cls(**kwargs)

        # Open the file
        hdu = fits.open(fitsfile)
        self.data = np.squeeze(hdu[extension].data)
        self.header = hdu[extension].header
        self.shape = self.data.shape
        self.__fix_header__()

        # adapt the header depending on the instrument and reduction software
        self.__instr_redux__(hdu)

        return self

    def __init__(self):

        # Attributes
        self.data = None
        self.header = None

    def get_slice(self, xindex=None, yindex=None, zindex=None):

        """ slice the data cube into smaller data cubes, this will update the
        header correctly.
        """

        # init
        Slice = copy.deepcopy(self)

        # deal with unassigned indices and tuple values
        if xindex is None:
            xindex = np.arange(Slice.data.shape[-1])
        elif type(xindex) is tuple:
            xindex = np.arange(xindex[0], xindex[1])
        if yindex is None:
            yindex = np.arange(Slice.data.shape[-2])
        elif type(yindex) is tuple:
            yindex = np.arange(yindex[0], yindex[1])
        # third axis only if it exists
        if Slice.data.ndim >= 3 and zindex is None:
            zindex = np.arange(Slice.data.shape[-3])
        elif type(zindex) is tuple:
            zindex = np.arange(zindex[0], zindex[1])

        # crop the data, model and sigma/variance cubes
        # NOTE: THE DATA COLUMN HAS TO BE PRESENT
        for attr in ['data', 'model', 'sig', 'variance']:
            if hasattr(Slice, attr):
                temp = getattr(Slice, attr)
                if temp.ndim == 2:
                    temp = temp[yindex[:, np.newaxis], xindex[np.newaxis, :]]
                    temp = np.squeeze(temp)
                    setattr(Slice, attr, temp)
                elif temp.ndim == 3:
                    temp = temp[zindex[:, np.newaxis, np.newaxis],
                                yindex[np.newaxis, :, np.newaxis],
                                xindex[np.newaxis, np.newaxis, :]]
                    temp = np.squeeze(temp)
                    setattr(Slice, attr, temp)
                else:
                    raise ValueError('Unable to crop data with this dimension')

        # now fix the coordinates in the header
        Slice.header['CRPIX1'] = Slice.header['CRPIX1'] - np.min(xindex)
        Slice.header['CRPIX2'] = Slice.header['CRPIX2'] - np.min(yindex)
        if Slice.data.ndim >= 3:
            Slice.header['CRPIX3'] = Slice.header['CRPIX3'] - np.min(zindex)

        # adjust the header and beam
        Slice.shape = Slice.data.shape
        Slice.__fix_header__()
        Slice.__fix_beam__(channels=zindex)

        return Slice

    def calculate_sigma(self, ignorezero=True, fullarray=False, channels=None,
                        plot=False, plotfile='./sigma_estimate.pdf', **kwargs):

        """ This function will fit a Gaussian to the data and return the
        sigma value. For multiple channels it will do the calculation for each
        channel and return an array of sigma values.

        keywords:
        ---------
        ignorezero (Bool| True):
            If set, this will ignore any zero values in the array for the
            gaussian fitting. This is useful if masked values are set to
            zero.

        fullarray (Bool| False):
            If set, the returned array will be a numpy array with the same
            shape as the input data, where each point corresponds to the
            sigma value of that channel

        channels (list| None):
            The list of channels to calculate the sigma value for. This can
            be either a list or a numpy array. When not specfied or set to
            None, the full range is taken.

        plot (Bool| False):
            If set a QA file is made of the Gaussian fits, to determine the
            fit of the Gaussian.

        plotfile (str| './sigma_estimate.pdf'):
            The file in which to save the sigma QA plot.

        doguess (Bool| True):
            If set, the code will calculate the bin size and the intial
            estimates for the gaussian fit

        gausspar (list| None):
            If a list is given it is taken as the initial guesses for the
            Gaussian fit. This has to be set with the number of bins for
            the histogram, otherwise doguess has to be set to True (which
            will overwrite the values given here)

        bins (int| None):
            The numbe of bins to use in the histogram. If not set, doguess
            needs to be set, which calculates this value.

        Returns:
        --------
        float or nd.array of the sigma for each requested channel.
        if full array is set, it will generate a new instance of Qube and
        will return the array in the 'data' attribute for this Qube.
        """

        # either 2D or 3D case
        if self.data.ndim == 2:

            # parse the data to be fitted
            Data = self.data
            if ignorezero:
                Data = Data[np.where(Data != 0)]

            # if qa plot is wanted
            if plot:
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                kwargs['ax'] = ax

            # fit the data
            g = __fit_gaussian__(Data, **kwargs)

            # save and close the plot
            if plot:
                plt.savefig(plotfile, format='pdf', dpi=300)
                plt.close('all')

            # get sigma
            Sigma = g.stddev.value

        elif self.data.ndim == 3:

            # channels to loop over
            if channels is None:
                channels = np.arange(0, self.data.shape[-3])

            # if qa plot is wanted
            if plot:
                pdf = matplotlib.backends.backend_pdf.PdfPages(plotfile)

            Sigma = []
            for channel in channels:
                Data = self.data[channel, :, :]

                if ignorezero:
                    Data = Data[np.where(Data != 0)]

                if plot:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    kwargs['ax'] = ax
                    kwargs['channel'] = channel

                # fit the data and get sigma
                goodidx = np.where((Data != 0) * (np.isfinite(Data)))
                if goodidx != np.array([]):
                    g = __fit_gaussian__(Data, **kwargs)
                    Sigma.append(g.stddev.value)
                    if plot:
                        pdf.savefig(fig)
                        plt.close('all')
                else:
                    Sigma.append(0)

            # close the plot
            if plot:
                pdf.close()

            # convert to numpy array
            Sigma = np.array(Sigma)

        # convert sigma to a full array the size of the original data
        # and upgrade it to a full Qube
        if fullarray:

            TSigma = copy.deepcopy(self)
            if Sigma.shape != self.data.shape[-3]:
                TSigma = self.get_slice(zindex=channels)
            else:
                TSigma = copy.deepcopy(self)
            TSigma.data = np.tile(Sigma[:, np.newaxis, np.newaxis],
                                  (1, self.data.shape[-2],
                                   self.data.shape[-1]))
            Sigma = TSigma

        return Sigma

    def mask_region(self, ellipse=None, rectangle=None, value=None,
                    moment=None, channels=None, mask=None, applymask=True):

        """ This function will mask the data cube.
        """

        # init
        MaskRegion = copy.deepcopy(self)
        Mask = np.ones_like(MaskRegion.data)

        if ellipse is not None:  # Ellipse Reg.: [xcntr,ycntr,rmaj,rmin,angle]
            # create indices array
            tidx = np.indices(MaskRegion.data.shape)

            # translate
            tidx[-1, :] = tidx[-1, :] - ellipse[0]
            tidx[-2, :] = tidx[-2, :] - ellipse[1]

            # rotate
            angle = (90 + ellipse[4]) * np.pi / 180.
            rmaj = tidx[-1, :] * np.cos(angle) + tidx[-2, :] * np.sin(angle)
            rmin = tidx[-2, :] * np.cos(angle) - tidx[-1, :] * np.sin(angle)

            # find pixels within ellipse and mask
            size = np.array([ellipse[2], ellipse[3]])
            tmask = np.where(((rmaj / size[0])**2 + (rmin / size[1])**2) <= 1,
                             1, np.nan)
            Mask = Mask * tmask

        if rectangle is not None:  # rectangular region
            # create indices array
            tidx = np.indices(MaskRegion.data.shape)

            tmask = np.where((tidx[-1, :] >= rectangle[0] - rectangle[2]) &
                             (tidx[-1, :] <= rectangle[0] + rectangle[2]) &
                             (tidx[-2, :] >= rectangle[1] - rectangle[3]) &
                             (tidx[-2, :] <= rectangle[1] + rectangle[3]))

            Mask = Mask * tmask

        if value is not None:  # reject values below this value
            if (type(value) is float or type(value) is int or
                    type(value) is np.float64):
                tval = np.full(MaskRegion.data.shape, value)
            else:  # assume list
                tval = np.zeros(MaskRegion.data.shape)
                for chan in np.arange(MaskRegion.data.shape[0]):
                    tval[chan, :, :] = value[chan]

            tmask = np.where(MaskRegion.data >= tval, 1, np.nan)

            Mask = Mask * tmask

        if moment is not None:
            raise NotImplementedError('Need to define the moments')
            # create a temporary moment-zero image
            if channels is None:
                raise ValueError('Channels need to be set for moment-zero' +
                                 ' masking!')
            Moment0 = MaskRegion.calc_moment(moment=0, channels=channels)
            MomentSig = Moment0.calc_sigma(bins=np.linspace(-2E-1, 2E-1, 100),
                                           gausssd=0.05)
            MomentMaskValue = np.ones_like(Moment0.data) * MomentSig * moment
            MomentMask = np.where(Moment0.data >= MomentMaskValue, 1, np.nan)

            tmask = np.tile(MomentMask, (MaskRegion.data.shape[0], 1, 1))
            Mask = Mask * tmask

        if mask is not None:
            Mask = Mask * mask

        # apply the mask
        if applymask:
            MaskRegion.data = MaskRegion.data * Mask
            return MaskRegion
        else:
            return Mask

    def calculate_moment(self, moment=0, channels=None, restfreq=None):
        """
        What are the units that come out??

        Args:
            moment:
            channels:
            restfreq:

        Returns:
            Qube:

        """

        # init
        mom = copy.deepcopy(self)

        # slice if needed
        if channels is not None:
            mom = mom.get_slice(zindex=channels)

        # update the rest frequency in the moment header
        if restfreq is not None:
            mom.header['RESTFRQ'] = restfreq

        # calculate the moment
        if moment == 0:     # 0th moment
            dv = mom.__get_velocitywidth__()
            mom.data = np.nansum(mom.data, axis=0) * dv
        elif moment == 1:   # 1st moment
            tVel = mom._getvelocity_()
            VelArr = np.tile(tVel[:, np.newaxis, np.newaxis],
                             (1, mom.shape[1], mom.shape[2]))
            tmom0 = np.nansum(mom.data, axis=0)
            mom.data = np.nansum(VelArr * mom.data, axis=0) / tmom0
        elif moment == 2:   # 2nd moment
            tVel = mom._getvelocity_()
            VelArr = np.tile(tVel[:, np.newaxis, np.newaxis],
                             (1, mom.shape[1], mom.shape[2]))
            tmom0 = np.nansum(mom.data, axis=0)
            tmom1 = np.nansum(VelArr * mom.data, axis=0) / tmom0
            tmom2 = mom.data * np.square(VelArr - tmom1)
            mom.data = np.sqrt(np.nansum(tmom2, axis=0) / tmom0)
        else:
            raise NotImplementedError("Moment not supported - yet.")

        # fix the header and update beam
        mom.__fix_header__()
        mom.__fix_beam__()

        return mom

    def gaussian_moment(self, mom1=None, mom2=None, channels=None):

        # init
        data = copy.deepcopy(self)

        # slice if needed
        if channels is not None:
            data = data.get_slice(zindex=channels)

        # the guesses:
        if mom1 is None:
            mom1 = data.calculate_moment(moment=1)
        if mom2 is None:
            mom2 = data.calculate_moment(moment=2)

        # get velocity array
        VelArr = data._getvelocity_()

        # now go over each spatial pixel and compute moments
        for ii in np.arange(mom1.shape[-1]):
            for jj in np.arange(mom1.shape[-2]):
                RowData = data.data[:, jj, ii]
                isfin = np.isfinite(RowData)
                if np.sum(isfin) > 3:
                    gausspar = [np.nanmax(RowData), mom1.data[jj, ii],
                                mom2.data[jj, ii]]
                    g_init = models.Gaussian1D(amplitude=gausspar[0],
                                               mean=gausspar[1],
                                               stddev=gausspar[2])
                    fit_g = fitting.LevMarLSQFitter()
                    g = fit_g(g_init, VelArr[isfin], RowData[isfin])
                    mom1.data[jj, ii] = g.mean.value
                    mom2.data[jj, ii] = g.stddev.value
                else:
                    mom1.data[jj, ii] = np.NaN
                    mom2.data[jj, ii] = np.NaN

        return mom1, mom2

    def generative_moment(self, variance=None, cwidths=np.arange(1, 21, 1)):

        # create a slew of moment-0 images
        VelArr = self._getvelocity_()
        DV = self.__get_velocitywidth__()

        if variance is None:
            variance = np.square(self.calculate_sigma)

        SNR = np.zeros((1, self.shape[1], self.shape[2]))
        fMom0 = np.zeros((1, self.shape[1], self.shape[2]))
        ChanLE = [0]
        ChanRE = [0]

        for CW in cwidths:
            for LE in np.arange(0, self.shape[0] - CW):
                ChanLE.append(LE)
                ChanRE.append(LE + CW)
                Sig = np.sqrt(np.sum(variance[LE: LE + CW]))
                tMom0 = np.nansum(self.data[LE: LE + CW, :, :], axis=0) * DV
                fMom0 = np.append(fMom0, tMom0[np.newaxis, :], axis=0)
                SNR = np.append(SNR, tMom0[np.newaxis, :] / (Sig * DV), axis=0)

        # now find the moment image with the highest SNR
        SNRargmax = np.nanargmax(SNR, axis=0)
        SNRmax = np.nanmax(SNR, axis=0)
        Mom0 = np.zeros((fMom0.shape[1], fMom0.shape[2]))
        VelStart = np.zeros((fMom0.shape[1], fMom0.shape[2]))
        VelEnd = np.zeros((fMom0.shape[1], fMom0.shape[2]))
        for i in np.arange(fMom0.shape[1]):
            for j in np.arange(fMom0.shape[2]):
                Mom0[i, j] = fMom0[SNRargmax[i, j], i, j]
                VelStart[i, j] = VelArr[ChanLE[SNRargmax[i, j]]]
                VelEnd[i, j] = VelArr[ChanRE[SNRargmax[i, j]]]

        return Mom0, SNRmax, VelStart, VelEnd

    def get_spec1d(self, convention='radio', continuumcorrect=False,
                   limits=None, beamcorrect=True, **kwargs):

        """
        Generate a 1D spectrum from the Cube

        keywords:
        ---------

        convention: (String | 'Radio'):
            Which Doppler convention to use to calculate the velocities.

        continuumcorrect: (Bool | False):
            Will correct the spectrum for any potential continuum flux by
            fitting a second order polynomial to the spectrum outside the
            channels given by limits.

        limits: (tuple | None):
            Two-element tuple which contains the limits outside which the
            continuum can be fit.

        beamcorrect: (Bool | False):
            Correct the flux by dividing by the area of the beam. This
            should be the default when dealing with interferometric data
            in Jy/beam.

        Returns:
        --------
            np.ndarray, np.ndarray: Summed flux and 'Velocity'
        """

        # sum up the pixels in each channel
        if self.data.ndim == 2:
            spec = np.nansum(self.data)
        if self.data.ndim == 3:
            spec = np.nansum(np.nansum(self.data, axis=1), axis=1)

        # divide by the number of pixels per beam to get flux density
        if beamcorrect:
            # find the beam size (in pixels)
            beam_area = ((self.beam['BMAJ'] * self.beam['BMIN'] * np.pi) /
                         (4 * np.log(2)))
            beam_pix = beam_area / (np.abs(self.header['CDELT1'] *
                                           self.header['CDELT2']))
            spec = spec / beam_pix

        # deal with the 'velocity' array (or freq, etc).
        if self.data.ndim == 2:
            vel = self.header['RESTFRQ']
        if self.data.ndim == 3:
            vel = self._getvelocity_(convention=convention)

        # apply correction for potential continuum emission
        if continuumcorrect:
            if limits is None:
                raise ValueError('Please set the limits keyword!')
            spec = __correct_flux__(spec, vel, limits)

        return spec, vel

    def bootstrap_sigma(self, mask, nboot=500, asymmetric=False,
                        gaussian=True, **kwargs):

        """calculate the typical sigma for the cube by moving a mask around
        randomly
        """

        # recast a 2d array into 3d
        if self.data.ndim == 2:
            shape = (1,) + self.data.shape
        else:
            shape = self.data.shape

        # generate the random shift of the masks
        newx = (np.random.randint(shape[-1]/2, size=nboot) -
                shape[-1]/4).astype(int)
        newy = (np.random.randint(shape[-2]/2, size=nboot) -
                shape[-2]/4).astype(int)

        # calculate the flux for each masked region
        SpecArr = np.zeros((shape[-3], nboot))
        for nx, ny, ii in zip(newx, newy, np.arange(nboot)):
            print(ii/nboot)
            NewMask = np.roll(mask, (nx, ny), axis=(-2, -1))
            CubeNewMask = self.mask_region(mask=NewMask)
            NewSpec, _NewVel = CubeNewMask.get_spec1d()
            SpecArr[:, ii] = NewSpec

        # calculate the uncertainty (asymmetric, gaussian, other)
        if asymmetric:
            raise NotImplementedError('Would give you assymetric errors')

        elif gaussian:
            std = np.zeros(shape[-3])
            for ii in np.arange(shape[-3]):
                fitgauss = __fit_gaussian__(SpecArr[ii, :], **kwargs)
                std[ii] = fitgauss.stddev.value

            return std

        else:
            return np.std(SpecArr, axis=1)

    def pvdiagram(self, PA, center, width=1., vshift=0.0, convention='radio',
                  scale=1.):

        """This is a direct port of the code in mncube. It will take a qube
        and create a 2D image with position along the axis and velocity on the
        second axis. It uses scipy's 'rotate' to rotate the datacube along the
        axis defined by the PA and center and then extracts the total flux
        along the 'slit' with a width given by the width keyword. The final
        2D output has been flipped (if needed) to have the velocity increasing
        upward and the extent of the pvdata is returned.

        NOTE: Rotate introduces a problem in that it does not conserve the
        total flux of a cube when rotated.

        Inputs for the function are:

        qube:       the data qube to use.
        PA:         position angle of the axis used to generate the PV diagram
        center:     the center of rotation.
        width:      2*width+1 is the size of the box used for extraction
        vshift:     shift in velocity for all values (w.r.t. the 'Restfrq'
                    of the qube).
        convention: convention to used for frequency-velocity conversion the
                    default is 'radio'.
        scale:      scale used for the pixels (x and y). default is 1.

        The function returns
        position:   vector of values (scaled) of the position along the axis
        velocity:   vector of values (in km/s) of the velocity of the
                    observations
        extent:     extent of the pv data (to use with the extent keyword of
                    matplotlib.pyplot.imshow)
        pvdata:     2D array of the pv diagram
        """

        # init
        data = copy.deepcopy(self)

        # rotate using spline interpolation
        PA = PA + 90.       # change to x-axis def. (rotate clockwise)
        # need to change NaN into real numbers (zeroes) (Scipy v0.19.1)
        data.data = np.nan_to_num(data.data)
        data.data = rotate(data.data, PA, (1, 2))

        # position of the center of rotation w.r.t. the middle of the data cube
        Middle = (np.array([data.shape[2], data.shape[1]]) - 1) / 2.
        Center = center - Middle

        # position of middle of the rotated data cube
        RotMiddle = (np.array([data.data.shape[2],
                               data.data.shape[1]]) - 1) / 2.

        # position of the center in the rotated datacube
        PArad = PA * np.pi / 180.
        RotCenter = [(Center[0] * np.cos(PArad) + Center[1] *
                      np.sin(PArad) + RotMiddle[0]),
                     (-1 * Center[0] * np.sin(PArad) + Center[1] *
                      np.cos(PArad) + RotMiddle[1])]

        # extract a pv slice along the y-axis (along the slit)
        box = [int(round(RotCenter[1]) - width // 2),
               int(round(RotCenter[1]) + width // 2 + 1)]
        data.data = data.data[:, box[0]:box[1], :]
        pvarray = np.nanmean(data.data, axis=1)

        # axes vectors
        pixposition = np.arange(pvarray.shape[1]) - RotCenter[0]
        position = pixposition * scale
        velocity = data._getvelocity_(convention=convention) - vshift
        if velocity[0] > velocity[1]:
            pvarray = np.flipud(pvarray)
            velocity = np.flip(velocity, axis=0)

        # extent of the pvdata
        dv = np.mean(velocity[1:] - velocity[:-1])
        extent = ((pixposition[0] - 0.5) * scale,
                  (pixposition[np.size(pixposition) - 1] + 0.5) * scale,
                  velocity[0] - dv, velocity[np.size(velocity) - 1] + dv)

        return position, velocity, extent, pvarray

    def save(self, fitsfile='./cube.fits'):

        # save the cube as a fits file
        Fit1 = fits.PrimaryHDU(self.data, header=self.header)

        # now make numerous fixes to the beam array so it can be
        # read in using CASA
        # renumber channels from 0 to number of channels
        # fix for single beams
        if self.beam['BMAJ'].size == 1:
            self.beam['BMAJ'] = np.array([self.beam['BMAJ']])
            self.beam['BMIN'] = np.array([self.beam['BMIN']])
            self.beam['BPA'] = np.array([self.beam['BPA']])
            self.beam['POL'] = np.array([self.beam['POL']])

        Beam = Table([self.beam['BMAJ'] * 3600,
                      self.beam['BMIN'] * 3600,
                      self.beam['BPA'],
                      np.arange(self.beam['CHAN'].size).astype('i4'),
                      self.beam['POL']],
                     names=('BMAJ', 'BMIN', 'BPA', 'CHAN', 'POL'))
        Beam['BMAJ'].unit = 'arcsec'
        Beam['BMIN'].unit = 'arcsec'
        Beam['BPA'].unit = 'deg'
        BeamHeader = fits.header.Header()
        BeamHeader.extend((('NCHAN', self.beam['CHAN'].size), ('NPOL', 1)))

        Fit2 = fits.BinTableHDU(Beam, name='BEAMS', header=BeamHeader, ver=1)
        Fit = fits.HDUList([Fit1, Fit2])
        Fit.writeto(fitsfile, overwrite=True)

    # some auxilliary calls for potentially useful information
    def _getvelocity_(self, convention='radio', channels=None):

        """ This function will take the header information of the third
        dimension and the rest frequency defined also in the header, and
        convert the frequency values (using astropy's units and
        equivalencies package) to velocities. This function is used in
        generating moment images (1 and higher), but can also be called by
        the user to get the velocities of a certain channel or fraction of a
        channel.

        keywords:
        ---------
        convention (str|default: 'radio')
            The convention used for converting the frequencies to velocities.
            Default is 'radio' but can also be 'optical', 'relativistic' and
            'frequency', in the latter the frequency array/value is returned.
        channels (list|default: None):
            If None then get velocities for all of the channels in the data
            array. otherwise only the velocities of those channels that are
            specfied are returned.

        Returns:
        --------
            np.ndarray:  Velocity values (in km/s or Hz)
        """

        # get the channels/values to convert
        if channels is None:
            channels = np.arange(self.header["NAXIS3"])

        # convert the given header spectral units to a frequency array
        Arr = ((channels - self.header["CRPIX3"] + 1) *
               self.header["CDELT3"] + self.header["CRVAL3"])

        RestFreq = self.header['RESTFRQ'] * u.Hz
        if self.header['CTYPE3'] == 'FREQ':
            FreqArr = Arr * u.Hz
        elif self.header['CTYPE3'] == 'FELO-HEL':
            FreqArr = RestFreq / (1 - Arr / const.c.value)
        elif self.header['CTYPE3'] == 'AWAV':
            FreqArr = (const.c / (Arr * u.AA)).to(u.Hz)
        elif self.header['CTYPE3'] == 'VOPT':
            Velocity = u.Quantity(Arr,
                                  unit=self.header['CUNIT3']).to('km/s').value
            return Velocity
        elif self.header['CTYPE3'] == 'VRAD':
            Velocity = u.Quantity(Arr,
                                  unit=self.header['CUNIT3']).to('km/s').value
            return Velocity
        else:
            raise ValueError(self.header['CTYPE3'])

        # now convert the FreqArr into a 'velocity'
        if convention == 'radio':
            Freq2Vel = u.doppler_radio(RestFreq)
            Velocity = FreqArr.to(u.km / u.s, equivalencies=Freq2Vel).value
        elif convention == 'optical':
            Freq2Vel = u.doppler_optical(RestFreq)
            Velocity = FreqArr.to(u.km / u.s, equivalencies=Freq2Vel).value
        elif convention == 'relativistic':
            Freq2Vel = u.doppler_relativistic(RestFreq)
            Velocity = FreqArr.to(u.km / u.s, equivalencies=Freq2Vel).value
        elif convention == 'frequency':
            Velocity = FreqArr.value
        else:
            raise ValueError('Doppler convention is not defined')

        return Velocity

    # SOME INTERNAL CALLS FOR FIXING HEADERS, ETC.
    def __fix_header__(self):

        # update the NAXIS keywords
        axes = ['NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4']
        for cnt, axis in enumerate(axes):
            if cnt >= self.data.ndim:
                break
            if (self.header.get(axis, default=-1) !=
                    self.data.shape[self.data.ndim-cnt-1]):
                self.header[axis] = self.data.shape[self.data.ndim-cnt-1]
        self.header['NAXIS'] = self.data.ndim

        # remove the 4th dimension if not needed
        if self.data.ndim <= 3:
            keys = ['PC04_01', 'PC04_02', 'PC04_03', 'PC04_04',
                    'PC01_04', 'PC02_04', 'PC03_04', 'CTYPE4',
                    'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4',
                    'CROTA4', 'NAXIS4', 'CNAME4', 'PC4_1',
                    'PC4_2', 'PC4_3', 'PC1_4', 'PC2_4', 'PC3_4', 'PC4_4']
            for key in keys:
                self.header.remove(key, ignore_missing=True)

        # remove the third dimension if not needed
        if self.data.ndim <= 2:
            keys = ['PC03_01', 'PC03_02', 'PC03_03', 'PC01_03',
                    'PC02_03', 'CTYPE3', 'CRVAL3', 'CDELT3',
                    'CRPIX3', 'CUNIT3', 'CROTA3', 'NAXIS3', 'CNAME3',
                    'CD3_3', 'PC3_1', 'PC3_2', 'PC3_3', 'PC1_3',
                    'PC2_3']
            for key in keys:
                self.header.remove(key, ignore_missing=True)

    def __instr_redux__(self, hdu):

        # perform some 'tests' on the header to see what instrument
        # or package was used then change header
        # and add beam (if applicable)

        if self.header['TELESCOP'] == 'ALMA':
            if 'AIPS' in self.header.tostring():
                self.instr = 'ALMA_AIPS'
                self.__AIPSfix__()
            else:
                self.instr = 'ALMA_CASA'

        elif self.header['TELESCOP'] == 'EVLA':
            if 'AIPS' in self.header.tostring():
                self.instr = 'EVLA_AIPS'
                self.__AIPSfix__()
            else:
                self.instr = 'EVLA_CASA'

        elif self.header['TELESCOP'] == 'Hale5m':
            self.instr = 'PCWI_IDL'
            self.__CWIfix__()

        elif self.header['TELESCOP'] == 'Keck II':
            self.instr = 'KCWI'
            self.__KCWIfix__()
        elif self.header['TELESCOP'] == 'NOEMA':
            self.instr = 'NOEMA_GILDAS'
            self.__GILDASfix__()
        else:
            raise ValueError('Instrument not supported yet.')

        # add beam seperately
        self.__add_beam__(hdu)

    def __add_beam__(self, hdu):

        # load the beam info into a special attribute
        if 'CASAMBM' in self.header:
            self.beam = {'BMAJ': hdu[1].data['BMAJ'],
                         'BMIN': hdu[1].data['BMIN'],
                         'BPA': hdu[1].data['BPA'],
                         'CHAN': hdu[1].data['CHAN'],
                         'POL': hdu[1].data['POL']}
            # stupid fixes
            self.beam['BMAJ'] = self.beam['BMAJ'] / 3600
            self.beam['BMIN'] = self.beam['BMIN'] / 3600
        else:
            if 'NAXIS3' in self.header:
                nchan = self.header['NAXIS3']
            else:
                nchan = 1
            self.beam = {'BMAJ': np.full(nchan, self.header['BMAJ']),
                         'BMIN': np.full(nchan, self.header['BMIN']),
                         'BPA': np.full(nchan, self.header['BPA']),
                         'CHAN': np.arange(0, nchan),
                         'POL': np.zeros(nchan)}

    def __fix_beam__(self, channels=None):

        if channels is None:
            self.beam["BMAJ"] = np.mean(self.beam["BMAJ"])
            self.beam["BMIN"] = np.mean(self.beam["BMIN"])
            self.beam["BPA"] = np.mean(self.beam["BPA"])
            self.beam["CHAN"] = np.mean(self.beam["CHAN"], dtype=int)
            self.beam["POL"] = np.mean(self.beam["POL"], dtype=int)
        else:
            self.beam["BMAJ"] = self.beam["BMAJ"][channels]
            self.beam["BMIN"] = self.beam["BMIN"][channels]
            self.beam["BPA"] = self.beam["BPA"][channels]
            self.beam["CHAN"] = self.beam["CHAN"][channels]
            self.beam["POL"] = self.beam["POL"][channels]

    def __AIPSfix__(self):

        # rename the rest frequency (if needed)
        if 'RESTFREQ' in self.header and 'RESTFRQ' not in self.header:
            self.header.rename_keyword('RESTFREQ', 'RESTFRQ')

        # look for a line like this in the history to get beam information
        # HISTORY AIPS   CLEAN BMAJ=  1.7599E-03 BMIN=  1.5740E-03
        # BPA=   2.61

        for line in self.header['History']:
            if 'BMAJ' in line:
                if 'BMAJ' not in self.header:
                    self.header.insert('History',
                                       ('BMAJ', float(line.split()[3])))
                if 'BMIN' not in self.header:
                    self.header.insert('History',
                                       ('BMIN', float(line.split()[5])))
                if 'BPA' not in self.header:
                    self.header.insert('History',
                                       ('BPA', float(line.split()[7])))


    def __GILDASfix__(self):

        # rename the rest frequency (if needed)
        if 'RESTFREQ' in self.header and 'RESTFRQ' not in self.header:
            self.header.rename_keyword('RESTFREQ', 'RESTFRQ')
        # add cunit3 keyword
        if 'CUNIT3' not in self.header:
            self.header['CUNIT3'] = 'm/s'

    def __KCWIfix__(self):

        # add RESTFRQ keyword to the header
        restfreq = const.c .value / (self.header['RESTWAV'] * 1E-10)
        self.header.set('RESTFRQ', restfreq)

        #  add CDELT3 keyword and convert values to frequency
        cdelt3 = self.header['CD3_3']
        self.header.set('CDELT3', cdelt3)

        # add some 'fake' beam parameters these should be first
        # updated to the seeing values of the data.
        self.header.set('BMAJ', 1.0 / 3600.)
        self.header.set('BMIN', 1.0 / 3600.)
        self.header.set('BPA',  0.0)

    def __get_velocitywidth__(self, **kwargs):

        """ small function that will get the velocities and calculate the
        median distance between them (i.e., width of the velocity channel).
        """

        VelArr = self._getvelocity_(**kwargs)
        return np.median(np.abs(VelArr - np.roll(VelArr, 1)))


# some auxilliary functions not directly part of the Qube class
def __fit_gaussian__(data, doguess=True, gausspar=None, bins=None,
                     ax=None, channel=0):

    # parse only the finite data
    data = data[np.isfinite(data)]

    # decicde on the range and the bins
    if doguess:
        gausspar = [0., np.mean(data), np.std(data)]
        nbins = np.min([np.sqrt(len(data)).astype(int), 101])
        bins = np.linspace(-5*gausspar[2], 5*gausspar[2], nbins)
    else:
        if bins is None or gausspar is None:
            raise ValueError('Please set the bin range and/or approximate' +
                             'Gaussian values.')

    # create a histogram
    hist, txval = np.histogram(data, bins=bins)
    xval = (txval[:-1]+txval[1:]) / 2

    # update the gaussian guess for amplitude
    if doguess:
        gausspar[0] = np.max(hist)

    # define the Gaussian model
    g_init = models.Gaussian1D(amplitude=gausspar[0],
                               mean=gausspar[1],
                               stddev=gausspar[2])
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, xval, hist)

    if ax is not None:
        __make_sigplot__(g.stddev.value, xval, hist, g, bins, ax, channel)

    return g


def __make_sigplot__(sigma, xval, hist, g, bins, ax, channel):

    ax.plot(xval, hist, 'o')
    ax.plot(xval, g(xval), label='Gaussian')
    if np.isfinite(np.min(xval)+np.max(xval)):
        ax.set_xlim([np.min(xval), np.max(xval)])
    ax.text(0.02, 0.95, 'Gaussian sigma is: {:.3e}'.format(sigma),
            transform=ax.transAxes)
    ax.legend(loc=1)
    ax.text(0.02, 1.03, 'Channel: {}'.format(str(channel)),
            transform=ax.transAxes)


def __correct_flux__(flux, vel, limits):

    # fit the spectrum for potential continuum residual
    # (second order polynomial)

    finit = models.Polynomial1D(2, c0=0, c1=0, c2=0)
    fitter = fitting.LevMarLSQFitter()
    ofitter = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3,
                                                sigma=3.0)
    FitIdx = (vel < limits[0]) + (vel > limits[1])
    OFit, OFitData = ofitter(finit, vel[FitIdx], flux[FitIdx])

    return flux - OFit(vel)
