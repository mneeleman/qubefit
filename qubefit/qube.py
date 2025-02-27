"""Base class used in the QubeFit package."""
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
    """
    Initate the Qube class.

    Class for handeling of a variety of data cubes. This includes reading
    in the data cube, and preforming basic operations on the data cube. It
    also allows for a couple of higher level operations, such as moment
    generation and simple position-velocity diagrams. The cube has been
    primarily tested and written for (sub)-mm data cubes, but can work with
    a range of optical / NIR data cubes as well.
    """

    @classmethod
    def from_fits(cls, fitsfile, extension=0, **kwargs):
        """
        Instantiate a Qube class from a fits file.

        Read in a fits file. This is the primary way to load in a data
        cube. The fits file will be parsed for a header and data. These will
        be stored in the Qube class. Several other attributes will be
        defined (most notably the beam, shape and instrument).

        Parameters
        ----------
        cls : QUBE
            Qube Class instance.
        fitsfile : STRING
            The name of the fits file to be read in.
        extension : INT, optional
            The extension of the fots file to read in. The default is 0.
        **kwargs : DICT, optional
            keyword arguments that can be directly passed into the Qube class.

        Returns
        -------
        Qube
            Will return an instance of the qubefit.qube.Qube class. The
            instance should have the data and header attribute populated, as
            well as several other minor data attributes.

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
        self.__instr_redux__()
        # add beam
        self.__add_beam__(hdu)
        return self

    def get_slice(self, xindex=None, yindex=None, zindex=None):
        """
        Slice the data cube.

        This method will slice the data cube to extract smaller cubes or
        individual channels. For generality, the two spatial dimensions
        are the x and y indices. The frequency/velocity/wavelength dimension
        is denoted by the zindex. This method also updates the header files
        with the new information.

        Parameters
        ----------
        xindex : TUPLE or NUMPY.ARRAY, optional
            The indices to use to slice in the x-direction. This can either
            be a tuple, which will be intepreted as a range starting at the
            first up to but NOT including the second value, or a numpy array
            of indices. The default is None.
        yindex : TUPLE or NUMPY.ARRAY, optional
            The indices to use to slice in the y-direction. This can either
            be a tuple, which will be intepreted as a range starting at the
            first up to but NOT including the second value, or a numpy array
            of indices. The default is None.
        zindex : TUPLE or NUMPY.ARRAY, optional
            The indices to use to slice in the z-direction. This can either
            be a tuple, which will be intepreted as a range starting at the
            first up to but NOT including the second value, or a numpy array
            of indices. The default is None.

        Raises
        ------
        ValueError
            Can only crop two and three dimensional data. If another dimension
            is selected, this will result in a ValueError.

        Returns
        -------
        Qube
            Return a Qube instance with the updated data and header of the
            sliced data set (if present, the model, sig and variance are
            will also be siced).

        """
        # init
        data_slice = copy.deepcopy(self)
        # deal with unassigned indices and tuple values
        if xindex is None:
            xindex = np.arange(data_slice.data.shape[-1])
        elif type(xindex) is tuple:
            xindex = np.arange(xindex[0], xindex[1])
        if yindex is None:
            yindex = np.arange(data_slice.data.shape[-2])
        elif type(yindex) is tuple:
            yindex = np.arange(yindex[0], yindex[1])
        # third axis only if it exists
        if data_slice.data.ndim >= 3 and zindex is None:
            zindex = np.arange(data_slice.data.shape[-3])
        elif type(zindex) is tuple:
            zindex = np.arange(zindex[0], zindex[1])
        # crop the data, model and sigma/variance cubes
        # NOTE: THE DATA COLUMN HAS TO BE PRESENT
        for attr in ['data', 'model', 'sig', 'variance']:
            if hasattr(data_slice, attr):
                temp = getattr(data_slice, attr)
                if temp.ndim == 2:
                    temp = temp[yindex[:, np.newaxis], xindex[np.newaxis, :]]
                    temp = np.squeeze(temp)
                    setattr(data_slice, attr, temp)
                elif temp.ndim == 3:
                    temp = temp[zindex[:, np.newaxis, np.newaxis],
                                yindex[np.newaxis, :, np.newaxis],
                                xindex[np.newaxis, np.newaxis, :]]
                    temp = np.squeeze(temp)
                    setattr(data_slice, attr, temp)
        # now fix the coordinates in the header
        data_slice.header['CRPIX1'] = data_slice.header['CRPIX1'] - np.min(xindex)
        data_slice.header['CRPIX2'] = data_slice.header['CRPIX2'] - np.min(yindex)
        if data_slice.data.ndim >= 3:
            data_slice.header['CRPIX3'] = data_slice.header['CRPIX3'] - np.min(zindex)
        # adjust the header and beam
        data_slice.shape = data_slice.data.shape
        data_slice.__fix_header__()
        data_slice.__fix_beam__(channels=zindex)
        # return the data_slice
        return data_slice

    def calculate_sigma(self, ignorezero=True, fullarray=False, channels=None, use_residual=False,
                        plot=False, plotfile='./sigma_estimate.pdf', **kwargs):
        """
        Calculate the Gaussian noise of the data.

        This method will take the measurements of each channel, create a
        histogram and fits a Gaussian function to the histogram. It will
        return the Gaussian sigma of this fit as a noise estimate. This
        assumes that 1) the nosie is Gaussian and 2) the signal does not
        signficantly affect the noise property. If it does, then the data
        should be masked first.

        Parameters
        ----------
        ignorezero : BOOLEAN, optional
            If set, this will ignore any zero values in the array for the
            gaussian fitting. This is useful if masked values are set to
            zero. The default is True.
        fullarray : BOOLEAN, optional
            If set, the returned array will be a numpy array with the same
            shape as the input data, where each point corresponds to the
            sigma value of that channel. The default is False.
        channels : NUMPY.ARRAY, optional
            The list of channels to calculate the sigma value for. This can
            be either a list or a numpy array. When not specfied or set to
            None, the full range is taken. The default is None.
        plot : BOOLEAN, optional
            If set a QA file is made of the Gaussian fits, to determine the
            fit of the Gaussian for each channel. The default is False.
        plotfile : TYPE, optional
            The file in which to save the sigma QA plot.
            The default is './sigma_estimate.pdf'.
        use_residual : BOOLEAN, optional
            If set, the calculation will use the resdiual data array, which
            is often the data-model array for improved statistics.

        Inhereted Parameters
        --------------------
        The project also inherits the keywords from __fit_gaussian__. In
        particular the keywords doguess (Boolean that if set, will guess
        the initial estimates for the fit), guasspar (list, the inital
        estimates for the fit if doguess is not set), and bins (int, the
        number of bins in the histogram). Normally doguess does a good
        job and you do not need to worry about these.

        Returns
        -------
        FLOAT, NUMPY.ARRAY or Qube instance
            The Gaussian noise (sigma) for each requested channel. If full
            array is set, it will generate a new instance of Qube and
            will return the array in the 'data' attribute for this Qube.

        """
        # either 2D or 3D case
        if self.data.ndim == 2:
            # parse the data to be fitted
            if use_residual and hasattr(self, 'residual'):
                data = self.residual
            else:
                data = self.data
            if ignorezero:
                data = data[np.where(data != 0)]
            if plot:  # if qa plot is wanted
                fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                g = __fit_gaussian__(data, ax=ax, **kwargs)
                plt.savefig(plotfile, format='pdf', dpi=300)
                plt.close('all')
            else:
                g = __fit_gaussian__(data, **kwargs)
            # get sigma
            sigma = g.stddev.value
        elif self.data.ndim == 3:
            # channels to loop over
            if channels is None:
                channels = np.arange(0, self.data.shape[-3])
            # if qa plot is wanted
            if plot:
                pdf = matplotlib.backends.backend_pdf.PdfPages(plotfile)
            sigma = []
            for channel in channels:
                if use_residual and hasattr(self, 'residual'):
                    data = self.residual[channel, :, :]
                else:
                    data = self.data[channel, :, :]
                if ignorezero:
                    data = data[np.where(data != 0)]
                if plot:
                    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
                    kwargs['ax'] = ax
                    kwargs['channel'] = channel
                # fit the data and get sigma
                goodidx = np.where((data != 0) * (np.isfinite(data)))
                if goodidx != np.array([]):
                    g = __fit_gaussian__(data, **kwargs)
                    sigma.append(g.stddev.value)
                    if plot:
                        pdf.savefig(fig)
                        plt.close('all')
                else:
                    sigma.append(0)
            # close the plot
            if plot:
                pdf.close()
            # convert to numpy array
            sigma = np.array(sigma)
        else:
            raise ValueError('data dimensions need to be either 2D or 3D.')

        # convert sigma to a full array the size of the original data
        # and upgrade it to a full Qube
        if fullarray:
            temp_sigma = copy.deepcopy(self)
            if sigma.shape != self.data.shape[-3]:
                temp_sigma = self.get_slice(zindex=channels)
            else:
                temp_sigma = copy.deepcopy(self)
            temp_sigma.data = np.tile(sigma[:, np.newaxis, np.newaxis], (1, self.data.shape[-2], self.data.shape[-1]))
            sigma = temp_sigma
        return sigma

    def mask_region(self, ellipse=None, rectangle=None, value=None,
                    moment=None, channels=None, mask=None, applymask=True):
        """
        Mask a region of the data cube.

        This method will mask the data cube. Several options for masking
        are available. Either a rectangle or ellipse for regions can be
        chosen or a specfiic value in the data cube. Finally, an option is
        to mask the region from a summed cube (moment-zero). A mask can also
        be read in and finally either the mask is returned as a numpy array,
        or the mask is applied and another Qube instance is returned.
        It should be noted that the masks are multiplicative.

        Parameters
        ----------
        ellipse : TUPLE, optional
            The 5-element tuple describing an ellipse (xc, yc, maj, min, ang).
            Each of the values in the tuple correspond to: the center of the
            ellipse in the x- and y-direction, (xc and yc), the length of the
            major and minor axis (maj and min) and the angle of the major
            axis (ang). The default is None.
        rectangle : TUPLE, optional
            The 4-element tuple describing the rectangles bottom left
            (xb, yb) and uppper right (xt, yt) corners (xb, yb, xt, yt).
            The default is None.
        value : FLOAT or LIST, optional
            Value used to mask everything below this value. This value can
            also be a list with the same size as the number of channels, in
            which case the comparison per channel is done piece-wise.
            The default is None.
        moment : FLOAT, optional
            First a temporary moment image is created, using the channels
            keyword. Then the cube will be masked for all pixels below
            the value set by moment, which is given in terms of the Gaussian
            noise sigma. The default is None.
        channels : TUPLE or NUMPY.ARRAY, optional
            Values used to slice the data in the spectral direction. This
            only needs to be set if the moment mask is requested, and if not
            set, it will use the full spetral range.
            The default is None.
        mask : NUMPY.ARRAY, optional
            Predefined numpy array to use as a mask. The default is None.
        applymask : BOOLEAN, optional
            This will apply the mask and return a Qube instance. If this is
            not set the mask will be returned as a numpy.array.
            The default is True.

        Returns
        -------
        NUMPY.ARRAY or Qube instance
            If applymask is true the mask data set will be returned as a
            Qubefit instance, if false, a numpy array of the mask will be
            returned.

        """
        # init
        mask_region = copy.deepcopy(self)
        msk = np.ones_like(mask_region.data)

        # Ellipse: [xcntr,ycntr,rmaj,rmin,angle]
        if ellipse is not None:
            # create indices array
            tidx = np.indices(mask_region.data.shape)
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
            msk = msk * tmask

        # Rectangle: [xb, yb, xt, yt]
        if rectangle is not None:
            # create indices array
            tidx = np.indices(mask_region.data.shape)
            tmask = np.where((tidx[-1, :] >= rectangle[0] - rectangle[2]) &
                             (tidx[-1, :] <= rectangle[0] + rectangle[2]) &
                             (tidx[-2, :] >= rectangle[1] - rectangle[3]) &
                             (tidx[-2, :] <= rectangle[1] + rectangle[3]))
            msk = msk * tmask

        # Value: data > value
        if value is not None:  # reject values below this value
            if (type(value) is float or type(value) is int or
                    type(value) is np.float64):
                tval = np.full(mask_region.data.shape, value)
            else:  # assume list
                tval = np.zeros(mask_region.data.shape)
                for chan in np.arange(mask_region.data.shape[0]):
                    tval[chan, :, :] = value[chan]
            tmask = np.where(mask_region.data >= tval, 1, np.nan)
            msk = msk * tmask

        # Moment: mom0 > moment
        if moment is not None:
            # create a temporary moment-zero image
            mom0 = mask_region.calculate_moment(moment=0, channels=channels)
            mom_sig = mom0.calculate_sigma()
            moment_mask_value = np.ones_like(mom0.data) * mom_sig * moment
            moment_mask = np.where(mom0.data >= moment_mask_value, 1, np.nan)
            tmask = np.tile(moment_mask, (mask_region.data.shape[0], 1, 1))
            msk = msk * tmask

        # Set mask manually
        if mask is not None:
            msk = msk * mask

        # apply the mask
        if applymask:
            mask_region.data = mask_region.data * msk
            return mask_region
        else:
            return msk

    def calculate_moment(self, moment=0, channels=None, restfreq=None,
                         use_model=False, **kwargs):
        """
        Calculate the moments for the data.

        This method will determine the moments for the data. These are
        the moments w.r.t. the spectral axis, as is typical. A detailed
        description of the different moments can be found elsewhere.
        In summary, the moment 0 will yield an integrated measurement of
        the flux density, the moment 1 will give an estimate of the velocity
        field and the moment 2 is an estimate of the velocity dispersion.

        Parameters
        ----------
        moment : INT, optional
            Determines which moment to return. Currently only moments up to
            and including 2 are defined. The default is 0.
        channels : TUPLE or NUMPY.ARRAY, optional
            The channels to use in creating the moment If not set, the full
            data cube is used. The channel ranges can be given as a numpy
            array or 2-element tuple with the first and last channel, where
            the last channel is NOT included. The default is None.
        restfreq : FLOAT, optional
            The rest frequency to use (in Hz). It is recommended that this
            is defined directly in the Qube instance, i.e., when it it read
            in or right after. However, for convenience this can be
            (re-)defined here. The default is None.
        use_model : BOOLEAN, optional
            If set to true, the moment will be calculated from the model
            data attribute instead of the data attribute. The default is False.
        **kwargs : VARIED, optional
            This method will take in the keywords defined in the method
            get_velocity. In particular the convention keyword which can
            change the output unit of the moments (see below)

        Raises
        ------
        NotImplementedError
            Will raise a NotImplementedError if the moment keyword is set
            to something else than 0, 1, 2.

        Returns
        -------
        Qube
            To output is a full instance of Qube where the data attribute
            containts the moment calculation. The units of the output varies
            and depend on the input and the convention used to get the
            velocities. In general the first moment will be an integrated
            flux density. For example if the data of the cube is in Jy/beam
            and the velocities are in km/s (the default), then the moment-0
            will have the units of Jy * km/s / beam. The moment-1 and
            moment-2 will have the same unit as the velocity convention
            which for the default is km/s.

        """
        # init
        mom = copy.deepcopy(self)

        # slice if needed
        if channels is not None:
            mom = mom.get_slice(zindex=channels)

        # update the rest frequency in the moment header
        if restfreq is not None:
            mom.header['RESTFRQ'] = restfreq

        # select the array to perform the calculation on
        if use_model:
            array = mom.model
        else:
            array = mom.data

        # calculate the moment
        if moment == 0:     # 0th moment
            dv = mom.get_velocitywidth(**kwargs, as_quantity=True)
            mom.data = np.nansum(array, axis=0) * dv.value
            mom.header['BUNIT'] = mom.header['BUNIT'] + dv.unit.to_string()
        elif moment == 1:   # 1st moment
            t0Vel = mom.get_velocity(**kwargs, as_quantity=True)
            tVel = t0Vel.value
            VelArr = np.tile(tVel[:, np.newaxis, np.newaxis],
                             (1, mom.shape[1], mom.shape[2]))
            tmom0 = np.nansum(array, axis=0)
            mom.data = np.nansum(VelArr * array, axis=0) / tmom0
            mom.header['BUNIT'] = t0Vel.unit.to_string()
        elif moment == 2:   # 2nd moment
            t0Vel = mom.get_velocity(**kwargs, as_quantity=True)
            tVel = t0Vel.value
            VelArr = np.tile(tVel[:, np.newaxis, np.newaxis],
                             (1, mom.shape[1], mom.shape[2]))
            tmom0 = np.nansum(array, axis=0)
            tmom1 = np.nansum(VelArr * array, axis=0) / tmom0
            tmom2 = array * np.square(VelArr - tmom1)
            mom.data = np.sqrt(np.nansum(tmom2, axis=0) / tmom0)
            mom.header['BUNIT'] = t0Vel.unit.to_string()
        else:
            raise NotImplementedError("Moment not supported - yet.")

        # fix the header and update beam
        mom.__fix_header__()
        mom.__fix_beam__()

        return mom

    def gaussian_moment(self, mom1=None, mom2=None, channels=None,
                        use_model=False, return_amp=False, **kwargs):
        """
        Calculate the Gaussian 'moments' of the cube.

        This method will fit a Gaussian to the spectrum of each spatial
        pixel. The velocity field and velocity dispersion field are then
        estimated from the velocity shift of the Gaussian and the width of
        the Gaussian, respectively. This method is often more robust in the
        case of low S/N and lower resolution (see the reference paper). To
        help provide better convergence of the fitting routine, it is
        recommended to supply initial guesses to the velocity field and
        dispersion field, which probably come from the method
        'calculate_moment'.

        This method can be very slow for a large data cube in the spatial
        direction.

        Parameters
        ----------
        mom1 : Qube, optional
            This is a qube instance that contains the initial guess for the
            velocity field. If not given this estimate will be calculated
            from the data cube (not recommended). The default is None.
        mom2 : Qube, optional
            This is a qube instance that contains the initial guess for the
            velocity dispersion field. If not given this estimate will be
            calculated from the data cube (not recommended). The default is
            None.
        channels : NUMPY.ARRAY or TUPLE, optional
            The channels to use in creating the gaussian 'moments' If not set,
            the full data cube is used. The channel ranges can be given as a
            numpy array or 2-element tuple with the first and last channel,
            where the last channel is NOT included. It is recommended to use
            a large range in channels, so the zero level can be accurately
            estimated, which is one of the strengths of this appraoch.
            The default is None.
        use_model : BOOLEAN, optional
            If set to true, the moment will be calculated from the model
            data attribute instead of the data attribute. The default is False.
        **kwargs : VARIED , optional
            This method will take in the keywords defined in the method
            get_velocity. In particular the convention keyword which can
            change the output unit of the moments.

        Returns
        -------
        mom1 : Qube
            A qube instance where the data attribute contains the velocity
            field estimate. Typically in km/s.
        mom2 : Qube
            A qube instance where the data attribute contains the velocity
            dispersion field estimate. Typically in km/s.

        """
        # init
        data = copy.deepcopy(self)

        # slice if wanted
        if channels is not None:
            data = data.get_slice(zindex=channels)

        # the guesses:
        if return_amp:
            amp = data.calculate_moment(moment=0, use_model=use_model)
        if mom1 is None:
            mom1 = data.calculate_moment(moment=1, use_model=use_model)
        if mom2 is None:
            mom2 = data.calculate_moment(moment=2, use_model=use_model)

        # get velocity array
        VelArr = data.get_velocity(**kwargs)

        # now go over each spatial pixel and compute moments
        for ii in np.arange(mom1.shape[-1]):
            for jj in np.arange(mom1.shape[-2]):
                if use_model:
                    RowData = data.model[:, jj, ii]
                else:
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
                    if return_amp:
                        amp.data[jj, ii] = g.amplitude.value
                else:
                    if return_amp:
                        amp.data[jj, ii] = np.NaN
                    mom1.data[jj, ii] = np.NaN
                    mom2.data[jj, ii] = np.NaN

        if return_amp:
            return amp, mom1, mom2
        else:
            return mom1, mom2

    def get_spec1d(self, continuum_correct=False, limits=None,
                   beam_correct=True, use_model=False, **kwargs):
        """
        Generate a 1D spectrum from the data cube.

        This method will extract a spectrum from the data cube integrated
        over an area This is a simple sum over all of the pixels in the
        data cube and therefore it probably is most useful after some
        masking has been applied using the method 'mask_region'. Standard
        is to correct for the beam to get a flux density from the region,
        i.e., if the data cube is in Jy/beam then the result will be a
        flux density in Jy. It is also possible to correct for any
        residual continuum.

        Parameters
        ----------
        continuum_correct: BOOLEAN, optional
            If set, this will correct the spectrum for any potential
            continuum flux by fitting a second order polynomial to the
            spectrum outside the channels given by limits.
            The default is False.
        limits: LIST, optional
            Two-element list which contains the limits outside which the
            continuum will be fit, if set. The default is None.
        beam_correct: BOOLEAN, optional
            Correct the flux by dividing by the area of the beam. This
            should be the default when dealing with interferometric data
            in Jy/beam. The default is True.
        use_model : BOOLEAN, optional
            If set to true, the moment will be calculated from the model
            data attribute instead of the data attribute. The default is False.
        **kwargs : VARIED , optional
            This method will take in the keywords defined in the method
            get_velocity. In particular the convention keyword which can
            change the output unit of the moments.

        Returns
        -------
            NUMPY.ARRAY, NUMPY.ARRAY
                Summed flux and 'Velocity' are returned. The first is in the
                same units as the data cube, whereas the latter quantity has
                the units as defined by the convention. This is either a
                velocity, frequency or wavelength array. The default is to
                return a velocity array in km/s.

        """
        # select the data
        if use_model:
            data = self.model
        else:
            data = self.data

        # sum up the pixels in each channel
        if data.ndim == 2:
            spec = np.nansum(data)
        elif data.ndim == 3:
            spec = np.nansum(np.nansum(data, axis=1), axis=1)
        else:
            raise NotImplementedError('Can only deal with 2D or 3D data.')

        # divide by the number of pixels per beam to get flux density
        if beam_correct:
            # find the beam size (in pixels)
            beam_area = ((self.beam['BMAJ'] * self.beam['BMIN'] * np.pi) /
                         (4 * np.log(2)))
            beam_pix = beam_area / (np.abs(self.header['CDELT1'] *
                                           self.header['CDELT2']))
            spec = spec / beam_pix

        # deal with the 'velocity' array (or freq, etc).
        if data.ndim == 2:
            vel = self.header['RESTFRQ']
        if data.ndim == 3:
            vel = self.get_velocity(**kwargs)

        # apply correction for potential continuum emission
        if continuum_correct:
            if limits is None:
                raise ValueError('If you want to continuum corerct, you need' +
                                 ' to set the limits keyword. If you want to' +
                                 ' fit the while spectrum set limits=[0, 0]')
            spec = __correct_flux__(spec, vel, limits)

        return spec, vel

    def pvdiagram(self, PA, center, width=3., vshift=0.0, scale=1.,
                  use_model=False, **kwargs):
        """
        Create the PV diagram along the given line.

        This method will take a data or model attribute from the qube instance,
        and create a 2D image with position along the axis and velocity on the
        second axis. It uses scipy's 'rotate' to rotate the datacube along the
        axis defined by the PA and center and then extracts the total flux
        along the 'slit' with a width given by the width keyword. The final
        2D output has been flipped (if needed) to have the velocity increasing
        upward and the extent of the pvdata is returned.

        NOTE: Rotate introduces a problem in that it does not conserve the
        total flux of a cube when rotated. This effect is often small (<10%),
        but should be kept in mind when looking at these pv diagrams.

        Parameters
        ----------
        PA : FLOAT
            The position angle of the axis used to generate the PV diagram in
            degrees east of north.
        center : Tuple
            Two-element tuple describing the center of the PV line, where
            the first value is the x position and the second value is the
            y position.
        width : INT, optional
            The 'width' of the 'slit'. The slit width is actually defined
            as two times this width plus 1 (2 x width + 1). Because of the
            issues with rotation, it is not advisable to set this value to
            0 (i.e., a width of exactly 1 pixel). The default is 3.
        vshift: FLOAT, optional
            This keyword allows the velocity to be shifted w.r.t. to the
            zero velocity as defined by the rest frequency of the cube. This
            is useful to make small velocity corrections.
        scale: FLOAT, optional
            The scale to use for the pixels (x and y). This can provide a
            convenient way to convert the distance along the PV line from
            the unit pixels to more useful quantities such as kpc or arcsec.
            The default is 1.
        **kwargs : VARIED , optional
            This method will take in the keywords defined in the method
            get_velocity. In particular the convention keyword which can
            change the output unit of the moments.

        Returns
        -------
        DICT
            The method returns a dictionary with four items defined.
            'pvdata', which is a numpy.array that contains the PV array of
            values'. 'extent', which is a four-element tuple with the extrema
            of the extent of the pv array, i.e., (xmin, xmax, vmin, vmax).
            This can be used with the extent keyword of
            matplotlib.pyplot.imshow. 'position', which is a numpy.array of
            positions defining each pixel along the PV line. Finally,
            'velocity' is a numpy.array of velocities defining each channel
            of the spectrum.

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
        velocity = data.get_velocity(**kwargs) - vshift
        if velocity[0] > velocity[1]:
            pvarray = np.flipud(pvarray)
            velocity = np.flip(velocity, axis=0)

        # extent of the pvdata
        dv = np.mean(velocity[1:] - velocity[:-1])
        extent = ((pixposition[0] - 0.5) * scale,
                  (pixposition[np.size(pixposition) - 1] + 0.5) * scale,
                  velocity[0] - dv / 2, velocity[-1] + dv / 2)

        return dict({'pvdata': pvarray, 'extent': extent, 'position': position,
                     'velocity': velocity})

    def to_fits(self, fitsfile='./cube.fits', overwrite_beam=True):
        """
        Write the qube instance to fits file.

        Parameters
        ----------
        fitsfile : STRING, optional
            The name of the fits file to save the qube instance to.
            The default is './cube.fits'.

        Returns
        -------
        None.

        """
        # if the image hasa a single beam (i.e., one channel such as a
        # continuum image or moment image), then store the beam in the
        # primary header and remove the CASAMBM
        if self.beam['BMAJ'].size == 1:
            if overwrite_beam:
                try:
                    self.header['BMAJ'] = self.beam['BMAJ'][0]
                    self.header['BMIN'] = self.beam['BMIN'][0]
                    self.header['BPA'] = self.beam['BPA'][0]
                except IndexError:
                    self.header['BMAJ'] = self.beam['BMAJ']
                    self.header['BMIN'] = self.beam['BMIN']
                    self.header['BPA'] = self.beam['BPA']
            self.header.remove('CASAMBM', ignore_missing=True)
            Fit = fits.PrimaryHDU(self.data, header=self.header)
            Fit.writeto(fitsfile, overwrite=True)
        else:
            Fit1 = fits.PrimaryHDU(self.data, header=self.header)
            # Some fixes to the multibeam structure
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

            Fit2 = fits.BinTableHDU(Beam, name='BEAMS', header=BeamHeader,
                                    ver=1)
            Fit = fits.HDUList([Fit1, Fit2])
            Fit.writeto(fitsfile, overwrite=True)

    # some auxilliary calls for potentially useful information
    def get_velocity(self, convention='radio', channels=None,
                     as_quantity=False):
        """
        Get 'velocities' of the channels in a data cube.

        This function will take the header information of the third
        dimension and the rest frequency defined also in the header, and
        convert the frequency values (using astropy's units and
        equivalencies package) to velocities.

        Parameters
        ----------
        convention : STRING, optional
            The convention used for converting the frequencies to velocities.
            For possible choices see the astropy.equivalencies documentation.
            They are 'radio', 'optical', 'relativistic'. In addition, the
            'frequency' and 'wavelength' can be give, which will return the
            array in frequency (Hz) or wavelength (m). The default is 'radio'.
        channels : NUMPY.ARRAY, optional
            If None then get velocities for all of the channels in the data
            array. otherwise only the velocities of those channels that are
            specfied are returned. The default is None.
        as_quantity : BOOLEAN, optional
            If set, it will return the array as a astropy.quantity instead of
            a unitless numpy.array. The default is False.

        Raises
        ------
        ValueError
            Method will raise a ValueError if the 'CTYPE3' in the header has
            a value that is unknown.

        Returns
        -------
        NUMPY.ARRAY or ASTROPY.QUANTITY
            'Velocity' array (in km/s or Hz, m)

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
            Velocity = u.Quantity(Arr, unit=self.header['CUNIT3']).to('km/s')
            Freq2Vel = u.doppler_radio(RestFreq)
            FreqArr = Velocity.to(u.Hz, equivalencies=Freq2Vel)
        elif self.header['CTYPE3'] == 'VRAD':
            Velocity = u.Quantity(Arr, unit=self.header['CUNIT3']).to('km/s')
            Freq2Vel = u.doppler_radio(RestFreq)
            FreqArr = Velocity.to(u.Hz, equivalencies=Freq2Vel)
        else:
            raise ValueError(self.header['CTYPE3'])

        # now convert the FreqArr into a 'velocity'
        if convention == 'radio':
            Freq2Vel = u.doppler_radio(RestFreq)
            Velocity = FreqArr.to(u.km / u.s, equivalencies=Freq2Vel)
        elif convention == 'optical':
            Freq2Vel = u.doppler_optical(RestFreq)
            Velocity = FreqArr.to(u.km / u.s, equivalencies=Freq2Vel)
        elif convention == 'relativistic':
            Freq2Vel = u.doppler_relativistic(RestFreq)
            Velocity = FreqArr.to(u.km / u.s, equivalencies=Freq2Vel)
        elif convention == 'frequency':
            Velocity = FreqArr
        elif convention == 'wavelength':
            Velocity = FreqArr.to(u.m, equivalencies=u.spectral())
        else:
            raise ValueError('Doppler convention is not defined')

        if as_quantity:
            return Velocity
        else:
            return Velocity.value

    def get_velocitywidth(self, **kwargs):
        """
        Calculate the channel width.

        This is a small function that will get the velocities and calculate
        the median distance between them (i.e., width of the velocity channel).
        It inherits the same keywords as the get_velocity method.
        """
        VelArr = self.get_velocity(**kwargs)
        return np.median(np.abs(VelArr - np.roll(VelArr, 1)))

    def _generative_moment_(self, variance=None, cwidths=np.arange(1, 21, 1)):
        """
        Generate a slew of moments to get the 'best' width.

        This method will calculate a bunch of different moment zero images
        and finds the width that results in the highest signal to nosie
        ratio. This can be used as a quick seach algorithm or as a way to
        show both narrow and wide emission features at the same time.
        """
        mom0 = copy.deepcopy(self)
        
        # create a slew of moment-0 images
        VelArr = self.get_velocity()
        DV = self.get_velocitywidth()

        if variance is None:
            variance = np.square(self.calculate_sigma())

        SNR = np.zeros((1, self.shape[1], self.shape[2]))
        fMom0 = np.zeros((1, self.shape[1], self.shape[2]))
        ChanLE = [0]
        ChanRE = [0]

        for CW in cwidths:
            print('qube.generative_moment: finding max S/N for ' +
                  'channel width of {} pixel'.format(CW))
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
        mom0.data = Mom0
                
        # fix the header and update beam
        mom0.__fix_header__()
        mom0.__fix_beam__()

        return mom0, SNRmax, VelStart, VelEnd

    def _bootstrap_sigma_(self, mask, nboot=500, asymmetric=False,
                          gaussian=True, **kwargs):
        """
        Calculate the typical sigma by moving a mask around randomly.

        This method will estimate the uncertainty by moving a mask around
        at randowm places and redoing the flux measurement. This works OK
        only for those masks that are bigger than the beam.
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

    ################################################
    # SOME INTERNAL CALLS FOR FIXING HEADERS, ETC. #

    def __init__(self):
        """Initiate the Qube instance (currently empty)."""
        self.data = None
        self.header = None

    def __fix_header__(self):
        """Fix the header for empty axes."""
        # update the NAXIS keywords
        axes = ['NAXIS1', 'NAXIS2', 'NAXIS3', 'NAXIS4']
        for cnt, axis in enumerate(axes):
            if cnt >= self.data.ndim:
                break
            if (self.header.get(axis, default=-1) !=
                    self.data.shape[self.data.ndim-cnt-1]):
                self.header[axis] = self.data.shape[self.data.ndim-cnt-1]
        self.header['NAXIS'] = self.data.ndim

        # remove the fourth dimension if not needed
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

    def __instr_redux__(self):
        """
        Assign the instrument/data reduction data attribute.

        This function will assign the instrument and data reduction
        software attribute. This will be used also to define some telescope
        specific 'fixes' to the data cube to read in the file correctly.
        """
        inst_red = {'ALMA': self.__ALMA__, 'EVLA': self.__EVLA__,
                    'Hale5m': self.__PCWI__, 'Keck II': self.__KCWI__,
                    'NOEMA': self.__NOEMA__, 'ESO-VLT-U4': self.__MUSE__,
                    'NGVLA': self.__NGVLA__, 'VLA': self.__VLA__,
                    'VLBA': self.__VLBA__}

        if 'INSTRUME' in self.header and 'TELESCOP' not in self.header:
            self.header['TELESCOP'] = self.header['INSTRUME']
        inst_red[self.header['TELESCOP']]()

    def __ALMA__(self):
        """Fix for ALMA, assuming reduction with AIPS or CASA."""
        if 'AIPS' in self.header.tostring():
            self.instr = 'ALMA_AIPS'
            self.__AIPS__()
        else:
            self.instr = 'ALMA_CASA'

    def __EVLA__(self):
        """Fix for EVLA, assuming reduction with AIPS or CASA."""
        if 'AIPS' in self.header.tostring():
            self.instr = 'EVLA_AIPS'
            self.__AIPS__()
        else:
            self.instr = 'EVLA_CASA'

    def __VLA__(self):
        """Fix for VLA, assuming reduction with AIPS or CASA."""
        if 'AIPS' in self.header.tostring():
            self.instr = 'EVLA_AIPS'
            self.__AIPS__()
        else:
            self.instr = 'VLA_CASA'

    def __NOEMA__(self):
        """Fix for NOEMA and GILDAS."""
        self.instr = 'NOEMA_GILDAS'
        # rename the rest frequency (if needed)
        if 'RESTFREQ' in self.header and 'RESTFRQ' not in self.header:
            self.header.rename_keyword('RESTFREQ', 'RESTFRQ')
        # add cunit3 keyword
        if 'CUNIT3' not in self.header:
            self.header['CUNIT3'] = 'm/s'

    def __KCWI__(self):
        """Fix for KCWI."""
        self.instr = 'KCWI'
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

    def __PCWI__(self):
        """Fix for Palomar CWI."""
        self.instr = 'PCWI_IDL'
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

    def __MUSE__(self):
        """Fix for MUSE."""
        self.instr = 'MUSE_PIPE'
        # add RESTFRQ keyword to the header
        restfreq = const.c .value / (self.header['RESTWAV'] * 1E-10)
        self.header.set('RESTFRQ', restfreq)

        #  add CDELT3 keyword and convert values to frequency
        if 'CD3_3' in self.header:
            cdelt3 = self.header['CD3_3']
            self.header.set('CDELT3', cdelt3)

        # add some 'fake' beam parameters these should be first
        # updated to the seeing values of the data.
        self.header.set('BMAJ', 1.0 / 3600.)
        self.header.set('BMIN', 1.0 / 3600.)
        self.header.set('BPA',  0.0)

    def __NGVLA__(self):
        """Fix for NGVLA, assuming reduction with CASA."""
        self.instr = 'NGVLA_CASA'

    def __VLBA__(self):
        """Fix for VLBA, assuming reduction with AIPS"""
        if 'AIPS' in self.header.tostring():
            self.instr = 'VLBA_AIPS'
            self.__AIPS__()
        else:
            raise ValueError('Only VLBA/AIPS is currently supported')

    def __AIPS__(self):
        """Fix specific to AIPS to deal with the beam."""
        # rename the rest frequency (if needed)
        if 'RESTFREQ' in self.header and 'RESTFRQ' not in self.header:
            self.header.rename_keyword('RESTFREQ', 'RESTFRQ')

        # look for a line like this in the history to get beam information
        # HISTORY AIPS CLEAN  BMAJ=1.7599E-03  BMIN=1.5740E-03  BPA=2.61
        for line in self.header['History']:
            if 'BMAJ' in line and 'AIPS' in line:
                if 'BMAJ' not in self.header:
                    self.header.insert('History',
                                       ('BMAJ', float(line.split()[3])))
                if 'BMIN' not in self.header:
                    self.header.insert('History',
                                       ('BMIN', float(line.split()[5])))
                if 'BPA' not in self.header:
                    self.header.insert('History',
                                       ('BPA', float(line.split()[7])))

    def __add_beam__(self, hdu):
        """Add the beam data attribute (different for CASA)."""
        if 'CASAMBM' in self.header:
            self.beam = {'BMAJ': hdu[1].data['BMAJ'] / 3600,
                         'BMIN': hdu[1].data['BMIN'] / 3600,
                         'BPA': hdu[1].data['BPA'],
                         'CHAN': hdu[1].data['CHAN'],
                         'POL': hdu[1].data['POL']}
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
        # add beam area in arsec and pixels
        self.beam['BAREA_DEG'] = (self.beam['BMAJ'] * self.beam['BMIN']) / (8 * np.log(2)) * 2 * np.pi
        self.beam['BAREA_PIX'] = self.beam['BAREA_DEG'] / np.square(self.header['CDELT1'])

    def __fix_beam__(self, channels=None):
        """Fix the beam data attribute."""
        if channels is None:
            self.beam["BMAJ"] = np.mean(self.beam["BMAJ"])
            self.beam["BMIN"] = np.mean(self.beam["BMIN"])
            self.beam["BPA"] = np.mean(self.beam["BPA"])
            self.beam["CHAN"] = np.mean(self.beam["CHAN"], dtype=int)
            self.beam["POL"] = np.mean(self.beam["POL"], dtype=int)
            self.beam["BAREA_DEG"] = np.mean(self.beam["BAREA_DEG"])
            self.beam["BAREA_PIX"] = np.mean(self.beam["BAREA_PIX"])
        else:
            self.beam["BMAJ"] = self.beam["BMAJ"][channels]
            self.beam["BMIN"] = self.beam["BMIN"][channels]
            self.beam["BPA"] = self.beam["BPA"][channels]
            self.beam["CHAN"] = self.beam["CHAN"][channels]
            self.beam["POL"] = self.beam["POL"][channels]
            self.beam["BAREA_DEG"] = self.beam["BAREA_DEG"][channels]
            self.beam["BAREA_PIX"] = self.beam["BAREA_PIX"][channels]


# some auxilliary functions not directly part of the Qube class
def __fit_gaussian__(data, doguess=True, gausspar=None, bins=None,
                     ax=None, channel=0):
    """Fit a Gaussian to a data distribution.

    Parameters:
    -----------
    doguess : BOOLEAN, optional
        If set, the code will calculate the bin size and the intial
        estimates for the gaussian fit. The default is True.
    gausspar : LIST, optional
        If a list is given it is taken as the initial guesses for the
        Gaussian fit. This has to be set with the number of bins for
        the histogram, otherwise doguess has to be set to True, which
        will overwrite the values given here. The default is None.
    bins : INT, optional
        The number of bins to use in the histogram. If not set, doguess
        needs to be set, which calculates this value.
    """

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
    g_init = models.Gaussian1D(amplitude=gausspar[0], mean=gausspar[1], stddev=gausspar[2])
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, xval, hist)
    # make a plot of the Gaussian fitting
    if ax is not None:
        __make_sigplot__(g.stddev.value, xval, hist, g, ax, channel)
    return g


def __make_sigplot__(sigma, xval, hist, g, ax, channel):
    """Make a histogram plot of the Gaussian fit."""
    ax.plot(xval, hist, 'o')
    ax.plot(xval, g(xval), label='Gaussian')
    if np.isfinite(np.min(xval)+np.max(xval)):
        ax.set_xlim([np.min(xval), np.max(xval)])
    ax.text(0.02, 0.95, 'Gaussian sigma is: {:.3e}'.format(sigma),
            transform=ax.transAxes)
    ax.legend(loc=1)
    ax.text(0.02, 1.03, 'Channel: {}'.format(str(channel)),
            transform=ax.transAxes)


def __correct_flux__(flux, vel, limits, n_degree=2):
    """
    Correct the flux for continuum wiggles.

    This will fit the spectrum for a potential continuum residual or
    wiggles. The fitting function of the continuum is a second order
    polynomial by default. The fit will ignore the region within the limits.
    """
    finit = models.Polynomial1D(n_degree)
    fitter = fitting.LevMarLSQFitter()
    ofitter = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3, sigma=3.0)
    fit_idx = (vel < limits[0]) + (vel > limits[1])
    _o_fit, o_fit_data = ofitter(finit, vel[fit_idx], flux[fit_idx])

    return flux - o_fit_data(vel)
