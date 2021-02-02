# some hopefully useful analysis utils to run after the qubefit chain
# has been generated.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
from matplotlib.offsetbox import AnchoredText
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import corner
from copy import deepcopy as dc


class AnchoredEllipse(AnchoredOffsetbox):
    def __init__(self, transform, width, height, angle, loc,
                 pad=0.1, borderpad=0.1, prop=None, frameon=True,
                 facecolor='White', alpha=1.0, **kwargs):
        """
        Draw an ellipse the size in data coordinate of the give axes.
        pad, borderpad in fraction of the legend font size (or prop)
        """
        self._box = AuxTransformBox(transform)
        self.ellipse = (Ellipse((0, 0), width, height, angle,
                        facecolor=facecolor, alpha=alpha, **kwargs))
        self._box.add_artist(self.ellipse)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=self._box, prop=prop, frameon=frameon)


def get_beam(Qube, scale=None, wcs=None):
    """
    Grab the beam from the header

    Args:
        Qube (Qube):
        scale (float, optional):
        wcs (WCS):
    Returns:
        np.ndarray, np.ndarray, np.ndarray:  width, height and angle of the
        beam. Default is in pixels but one can apply a scale
    """
    # Scale
    if scale is None:
        if wcs is not None:
            if hasattr(wcs.wcs, 'cd'):
                cdelt = wcs.wcs.cd[0][0]
            elif hasattr(wcs.wcs, 'cdelt'):
                cdelt = wcs.wcs.cdelt[0]
            else:
                raise ValueError('wcs does not have cd or cdelt defined.')
            scale = np.abs(Qube.header["CDELT1"] / cdelt)
        else:
            scale = 1.
    # Parse
    width = Qube.beam["BMAJ"] / Qube.header["CDELT1"] * scale
    height = Qube.beam["BMIN"] / Qube.header["CDELT1"] * scale
    angle = Qube.beam["BPA"]+90.

    # Return
    return width, height, angle


def qubebeam(Qube, ax, loc=3, pad=0.5, borderpad=0.4, frameon=True,
             color='Darkslategray', alpha=1.0, wcs=None, prop=None,
             scale=None, flip=False, **kwargs):

    # Grab the Beam
    width, height, angle = get_beam(Qube, scale=scale, wcs=wcs)

    # flip the beam if requested
    if flip:
        angle = -1 * angle

    # Note: not set up to deal with different x,y, pixel scales
    beam = AnchoredEllipse(ax.transData, width=width, height=height,
                           angle=angle, loc=loc, pad=pad, borderpad=borderpad,
                           frameon=frameon, facecolor=color, alpha=alpha,
                           prop=prop, **kwargs)

    return beam


def get_pltposrange(origin, shape, scale, getposition=False, flip=False):

    position = np.array([-origin[0], shape[0]-origin[0]-1,
                         -origin[1], shape[1]-origin[1]-1])
    pltrange = np.array([position[0]-0.5, position[1]+0.5,
                         position[2]-0.5, position[3]+0.5]) * scale

    # flip the position and position in the x-direction
    if flip:
        position = np.array([position[1], position[0], position[2], position[3]])
        pltrange = np.array([pltrange[1], pltrange[0], pltrange[2], pltrange[3]])
    if getposition:
        return position
    else:
        return pltrange


def standardfig(raster=None, contour=None, newplot=False, ax=None, fig=None,
                origin=None, scale=1.0, cmap='RdYlBu_r', vrange=None,
                cbar=False, cbaraxis=None, cbarlabel=None, vscale=None,
                tickint=5.0, clevels=None, ccolor='black', beam=True,
                text=None, textprop=[dict(size=12)], textposition=[[0.5, 0.85]],
                cross=None, crosssize=1., crosscolor='black', crosslw=2.,
                pdfname=None, pltrange=None, flip=False, **kwargs):

    """ This will make a standard figure of a 2D data set, it can be used to
    either draw contours or filled images only or both. It can be used as a
    stand-alone plotting program if newplot = True or it can plot the data
    on an already defined plot. Note that the data HAS to be 2D.

    keywords:
    ---------
    raster:     qube object used to generate the raster image, leave blank or
                'None' if not desired.
    contour:    qube object used to generate the contours, leave blank or
                'None' if not desired.
    newplot:    If true then a new plot (ax and fig) is generated using
                plt.subplots
    origin:     the zero point of the axis
    scale:      pixel scale to used, if not specified will use 1
    cmap:       colormap to use for the raster image
    vrange:     z-axis range to use. Defaults to min and max of the data
    cbar:       if true a color bar will be drawn
    cbaraxis:   axis used to draw the color bar (used with GridSpec)
    cbarlabel:  label of the color bar axis
    vscale:     scale/tickinterval to use for the colorbar defaults to 5 ticks
    tickint:    interval for the ticks on the x and y axis
    clevels:    values for which to draw the contours
    ccolors:    colors of the contour (defaults to black)
    beam:       if true will draw the beam of the observation
    text:       optional text to be added to the figure
    textprop:   dictionary of the properties of the text
    textposition:    Position of the text box (1=upper right)
    cross:      if set will draw a cross at the position given [x, y]
    crosssize:  size of the cross in units of the X and Y axis
    crosscolor: color of the cross
    crosslw:    linewidth o the cross
    pdfname:    if set, the name of the PDF file (only works with newplot=True)
    pltrange:
    flip:       boolean, if set to true it will flip the x-axis.
    """

    # check if new plot needs to be generated
    if newplot:
        # some global characteristics for the plot
        font = {'weight': 'normal',
                'size': 20}
        mpl.rc('font', **font)
        mpl.rc('axes', linewidth=2)

        # define the axis
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        if (ax is None or fig is None):
            raise ValueError('ax and fig need to be set if you do not want ' +
                             'a new plot!')

    # global figure properties
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tickint))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tickint))

    # the raster image
    if raster is not None:

        # define the extent of the plot
        if origin is None:
            origin = [(raster.data.shape[0]-1) // 2,
                      (raster.data.shape[0]-1) // 2]
        if pltrange is None:
            pltrange = get_pltposrange(origin, raster.data.shape, scale,
                                       flip=flip)

        # define the z-axis range
        if vrange is None:
            vrange = [np.nanmin(raster.data), np.nanmax(raster.data)]

        # plot the raster image
        im = ax.imshow(raster.data, cmap=cmap, origin='lower',
                       extent=pltrange, vmin=vrange[0], vmax=vrange[1])

        # plot the color bar
        if cbar:
            if vscale is None:
                vscale = (vrange[1] - vrange[0]) / 5.

            if cbaraxis is not None:
                cbr = cbaraxis.colorbar(im, ticks=np.arange(-10, 10) * vscale)
            else:
                cbr = fig.colorbar(im, ticks=np.arange(-10, 10) * vscale,
                                   ax=ax)
            # label of color bar
            if cbarlabel is not None:
                cbr.ax.set_ylabel(cbarlabel, labelpad=-1)

    # contour
    if contour is not None:

        # define the extent of the plot
        if origin is None:
            origin = [(contour.data.shape[0]-1) // 2,
                      (contour.data.shape[0]-1) // 2]
        position = get_pltposrange(origin, contour.data.shape, scale,
                                   getposition=True, flip=flip)
        xc = np.linspace(position[0], position[1],
                         contour.data.shape[0]) * scale
        yc = np.linspace(position[2], position[3],
                         contour.data.shape[1]) * scale

        # deal with contour levels
        if clevels is None:
            raise ValueError('Set the contour levels (clevels) keyword.')

        ax.contour(xc, yc, contour.data, levels=clevels, colors=ccolor)

    # the beam
    if beam:
        if raster is not None:
            data = raster
        elif contour is not None:
            data = contour
        else:
            raise ValueError('To draw a beam something has to be drawn')

        ax.add_artist(qubebeam(data, ax, scale=scale, loc=3, pad=0.3, flip=flip,
                               fill=None, hatch='////', edgecolor='black'))

    # optional text
    if text is not None:
        if type(text) is str:
            text = [text]
        if type(textposition) is int:
            textposition = [textposition]
        if type(textprop) is dict:
            textprop = [textprop]

        for txt, prop, pos in zip(text, textprop, textposition):
            ax.text(pos[0], pos[1], txt, prop, transform=ax.transAxes)

    # add cross
    if cross is not None:
        if type(crosssize) == float:
            ax.plot(np.array([cross[0], cross[0]]),
                    np.array([cross[1] - crosssize, cross[1] + crosssize]),
                    lw=crosslw, color=crosscolor)
            ax.plot(np.array([cross[0] - crosssize, cross[0] + crosssize]),
                    np.array([cross[1], cross[1]]),
                    lw=crosslw, color=crosscolor)
        elif type(crosssize) == tuple or type(crosssize) == list:
            ax.plot(np.array([cross[0], cross[0]]),
                    np.array([cross[1] - crosssize[1],
                              cross[1] + crosssize[1]]),
                    lw=crosslw, color=crosscolor)
            ax.plot(np.array([cross[0] - crosssize[0],
                              cross[0] + crosssize[0]]),
                    np.array([cross[1], cross[1]]),
                    lw=crosslw, color=crosscolor)
        else:
            raise ValueError('Incorrect type for crosssize.')
    # finish the new plot (if set)
    if newplot:
        if pdfname is not None:
            plt.savefig(pdfname, format='pdf', dpi=300)
        else:
            plt.show()


def create_channelmap(raster=None, contour=None, clevels=None, zeropoint=0.,
                      channels=None, ncols=4, vrange=None, vscale=None,
                      show=True, pdfname=None, cbarlabel=None, cmap='RdYlBu_r',
                      beam=True, **kwargs):

    """ Running this function will create a quick channel map of the Qube.
    One can either plot the contours or the raster image or both. This program
    can be used as a basis for a more detailed individual plot which can take
    better care of whitespace, etc. The following keywords are valid:

    raster:     qube object used to generate the raster image, leave blank or
                'None' if not desired.
    contour:    qube object used to generate the contours, leave blank or
                'None' if not desired.
    clevels:    nested lists of contour levels to draw, list should be the same
                length as the spectral dimension of the contour qube, if a
                single list is given assumes that the contours are the same
                for all channels.
    zeropoint:  Optional shift in velocities compared to the Restfrq keyword in
                the header of the Qube.
    ncols:      number of columns to plot
    vrange:     range of the z-axis (for consistency across panels) if None,
                will take minimum maximum of the Qube
    vscale:     division of the colorbar (if none will make ~5 divisions)
    cbarlabel:  if set, label of the color bar
    cmap:       colormap to use for the plotting
    pdfname:    if set, the name of the pdf file to which the image was saved

    Returns:
        fig
    """

    # generate a temporary qube from the data
    if raster is not None:
        tqube = raster
    elif contour is not None:
        tqube = contour
    else:
        raise ValueError('Need to define either a contour or raster image.')

    # first genererate the velocity array
    VelArr = tqube._getvelocity_() - zeropoint

    # define the number of channels, rows (and columns = already defined)
    if channels is None:
        channels = np.arange(tqube.shape[0])
    nrows = np.ceil(len(channels)/ncols).astype(int)

    # define the range of the z-axis (if needed)
    if vrange is None:
        vrange = [np.nanmin(tqube.data), np.nanmax(tqube.data)]
    # now generate a grid with the specified number of columns
    fig = plt.figure(1, (8., 8. / ncols * nrows))
    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.,
                     cbar_mode='single', cbar_location='right', share_all=True)

    # now loop over the channels
    for idx, chan in enumerate(channels):

        # get the string value of the velocity
        VelStr = str(int(round(VelArr[chan]))) + ' km s$^{-1}$'

        # get the boolean of the beam (bottom left figure only)
        if (idx % ncols == 0) and (idx // ncols == int(nrows) - 1) and beam:
            beambool = True
        else:
            beambool = False

        # plot the individual channels
        if raster is not None:
            rasterimage = raster.get_slice(zindex=(chan, chan+1))
        else:
            rasterimage = None
        if contour is not None:
            contourimage = contour.get_slice(zindex=(chan, chan+1))
            # also get the contour levels
            if clevels is None:
                raise ValueError('Set the contour levels using the clevels ' +
                                 'keyword')
            elif type(clevels[0]) is list or type(clevels[0]) is np.ndarray:
                clevel = clevels[chan]
            else:
                clevel = clevels
        else:
            contourimage = None
            clevel = clevels

        standardfig(raster=rasterimage, contour=contourimage, clevels=clevel,
                    newplot=False, fig=fig, ax=grid[idx], cbar=False,
                    beam=beambool, vrange=vrange, text=[VelStr], cmap=cmap,
                    **kwargs)

    # now do the color bar
    norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
    cmapo = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cmapo.set_array([])
    if vscale is None:
        vscale = (vrange[1] - vrange[0]) / 5.
    cbr = plt.colorbar(cmapo, cax=grid.cbar_axes[0],
                       ticks=np.arange(-10, 10) * vscale)

    # label of color bar
    if cbarlabel is not None:
        cbr.ax.set_ylabel(cbarlabel, labelpad=-1)

    if pdfname is not None:
        plt.savefig(pdfname, format='pdf', dpi=300)
    elif show:
        plt.show()
    else:
        pass

    return fig, grid, channels


def diagnostic_plots(model, chainfile, burnin=0.3, channelmaps=True,
                     cornerplot=True, momentmaps=True, bestarray=False,
                     outname='model', vrangecm=None, vrangemom=None,
                     cmap0='RdYlBu_r', cmap1='Spectral_r', cmap2='copper_r',
                     maskvalue=3.0, **kwargs):

    """ This program will create several diagnostic plots of a given model
    and MCMC chain. The plots are not paper-ready but are meant to understand
    how well the MCMC chain ran. Currently the following plots are generated:

    Channel maps of the data, model and residual as well as a composite file
    which shows the data with the residual overplotted as contours
    A corner plot of the given MCMC chain converted to the correct units
    (uses the corner package)
    Moment-0, 1 and 2 images of the data and the model. For the moment 0, the
    residual contours are also plotted in the moment0-data.

    input and keywords:
    model:       This is a Qube object that contains a defined model.
    chainfile:   This is a file containting an MCMC chain as a numpy.save
                 object. The shape of the object is (nwalkers, nruns, dim+1)
                 where dim is the number of variable parameters in the MCMC
                 chain and the extra slice contains the lnprobability of the
                 given parameters in that link.
    burnin:      burnin fraction of the chain (default = 30%, i.e., 0.3)
    channelmaps: If true will plot the channels maps of the data, model and
                 residual
    cornerplot:  If true will plot the corner plot of the chain file
    momentmaps:  If true will plot the zeroth, first and second moment of the
                 data and the model
    bestarray:   If true the plots will be generated from the model with the
                 highest probability if false the median parameters will be
                 chosen.
    outname:     The name of the root used to save the pdf figures
    vrangecm:    z-axis range used to plot the channelmaps
    vrangemom:   z-axis range used to plot the moment-0 channel maps
    cmap0:       colormap to use for the plotting of the moment-0 maps
    cmap1:       colormap to use for the plotting of the moment-1 maps
    cmap2:       colormap to use for the plotting of the moment-2 maps
    maskvalue:   value (in sigma) to use to generate the mask in the
                 moment-0 image which is used in higher moment images.
                 default is 3.0.
    """

    # read in the chain data file and load it in the model then regenerate
    # the model with the wanted values (either median or best)
    Chain = np.load(chainfile)
    Chain = Chain[:, int(burnin * Chain.shape[1]):, :-1]
    Chain = Chain.reshape((-1, Chain.shape[2]))
    model.get_chainresults(chainfile, burnin=burnin)
    if not bestarray:
        model.update_parameters(model.chainpar['MedianArray'])
    else:
        model.update_parameters(model.chainpar['BestArray'])
    model.create_model()

    # create the data, model and residual cubes
    dqube = dc(model)
    mqube = dc(model)
    mqube.data = model.model
    rqube = dc(model)
    rqube.data = model.data - model.model

    # make the corner plot
    if cornerplot is True:
        # convert each parameter to a physically meaningful quantity
        # units = []
        for idx, key in enumerate(model.mcmcmap):

            # get conversion factors and units for each key
            if model.initpar[key]['Conversion'] is not None:
                conversion = model.initpar[key]['Conversion'].value
            else:
                conversion = 1.
            # units.append(model.initpar[key]['Unit'].to_string())
            # quick fix for IO
            if key == 'I0':
                Chain[:, idx] = Chain[:, idx]*1E3

            Chain[:, idx] = Chain[:, idx] * conversion
        corner.corner(Chain, labels=model.mcmcmap, quantiles=[0.16, 0.5, 0.84],
                      show_titles=True)

        plt.savefig(outname + '_cornerplot.pdf', format='pdf', dpi=300)
        plt.close()

    # make the channel maps
    if channelmaps is True:
        # define some global properties for all plots
        if vrangecm is None:
            vrangecm = [np.nanmin(dqube.data), np.nanmax(dqube.data)]
        sigma = np.sqrt(model.variance[:, 0, 0])
        clevels = [np.insert(np.arange(3, 30, 3), 0, -3)*i for i in sigma]

        # make the channel map for the data
        create_channelmap(raster=dqube, contour=dqube, clevels=clevels,
                          pdfname=outname+'_datachannelmap.pdf',
                          vrange=vrangecm, **kwargs)

        # make the channel map for the model
        create_channelmap(raster=mqube, contour=mqube, clevels=clevels,
                          pdfname=outname+'_modelchannelmap.pdf',
                          vrange=vrangecm, **kwargs)

        # make the channel map for the residual
        create_channelmap(raster=rqube, contour=rqube, clevels=clevels,
                          pdfname=outname+'_residualchannelmap.pdf',
                          vrange=vrangecm, **kwargs)

        # make the channel map for the data with residual contours
        create_channelmap(raster=dqube, contour=rqube, clevels=clevels,
                          pdfname=outname+'_combinedchannelmap.pdf',
                          vrange=vrangecm, **kwargs)

        plt.close()

    # make the moment maps
    if momentmaps is True:
        # create the moment-0 images
        dMom0 = dqube.calculate_moment(moment=0)
        mMom0 = mqube.calculate_moment(moment=0)
        rMom0 = rqube.calculate_moment(moment=0)

        # calculate the Mom0sig of the data and create the contour levels
        Mom0sig = (np.sqrt(np.nansum(model.variance[:, 0, 0])) *
                   model.__get_velocitywidth__())
        clevels = np.insert(np.arange(3, 30, 3), 0, -3) * Mom0sig
        if vrangemom is None:
            vrangemom = [-3 * Mom0sig, 11 * Mom0sig]
        mask = dMom0.mask_region(value=Mom0sig * maskvalue, applymask=False)

        # create the figure
        fig = plt.figure(1, (8., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.,
                         cbar_mode='single', cbar_location='right')

        # plot the figures
        standardfig(raster=dMom0, contour=dMom0, clevels=clevels, ax=grid[0],
                    fig=fig, vrange=vrangemom, cmap=cmap0, text='Data',
                    textprop=[dict(size=12)], **kwargs)
        standardfig(raster=mMom0, contour=mMom0, clevels=clevels, ax=grid[1],
                    fig=fig, vrange=vrangemom, cmap=cmap0, beam=False,
                    text='Model', textprop=[dict(size=12)], **kwargs)
        standardfig(raster=rMom0, contour=rMom0, clevels=clevels, ax=grid[2],
                    fig=fig, vrange=vrangemom, cmap=cmap0, text='Residual',
                    textprop=[dict(size=12)], **kwargs)
        standardfig(raster=dMom0, contour=rMom0, clevels=clevels, ax=grid[3],
                    fig=fig, vrange=vrangemom, cmap=cmap0,
                    text='Data with residual contours',
                    textprop=[dict(size=12)], **kwargs)

        # plot the color bar
        norm = mpl.colors.Normalize(vmin=vrangemom[0], vmax=vrangemom[1])
        cmapo = plt.cm.ScalarMappable(cmap=cmap0, norm=norm)
        cmapo.set_array([])
        cbr = plt.colorbar(cmapo, cax=grid.cbar_axes[0])
        cbr.ax.set_ylabel('Moment-0', labelpad=-1)
        plt.savefig(outname + '_moment0.pdf', format='pdf', dpi=300)
        plt.close()

        # create the moment-1 images
        dsqube = dqube.mask_region(mask=mask)
        dMom1 = dsqube.calculate_moment(moment=1)
        msqube = mqube.mask_region(mask=mask)
        mMom1 = msqube.calculate_moment(moment=1)

        # create the figure
        fig = plt.figure(1, (8., 5.))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.,
                         cbar_mode='single', cbar_location='right')

        # plot the figures
        vrangemom1 = [np.nanmin(dMom1.data), np.nanmax(dMom1.data)]
        standardfig(raster=dMom1, ax=grid[0], fig=fig, cmap=cmap1,
                    vrange=vrangemom1, text='Data',
                    textprop=[dict(size=12)], **kwargs)
        standardfig(raster=mMom1, ax=grid[1], fig=fig, cmap=cmap1,
                    vrange=vrangemom1, beam=False, text='Model',
                    textprop=[dict(size=12)], **kwargs)

        # plot the color bar
        norm = mpl.colors.Normalize(vmin=vrangemom1[0], vmax=vrangemom1[1])
        cmapo = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
        cmapo.set_array([])
        cbr = plt.colorbar(cmapo, cax=grid.cbar_axes[0])
        cbr.ax.set_ylabel('Moment-1', labelpad=-1)

        plt.savefig(outname + '_moment1.pdf', format='pdf', dpi=300)
        plt.close()

        # create the moment-2 images
        dsqube = dqube.mask_region(mask=mask)
        dMom2 = dsqube.calculate_moment(moment=2)
        msqube = mqube.mask_region(mask=mask)
        mMom2 = msqube.calculate_moment(moment=2)

        # create the figure
        fig = plt.figure(1, (5., 8.))
        grid = ImageGrid(fig, 111, nrows_ncols=(1, 2), axes_pad=0.,
                         cbar_mode='single', cbar_location='right')

        # plot the figures
        vrangemom2 = [np.nanmin(dMom2.data), np.nanmax(dMom2.data)]
        standardfig(raster=dMom2, ax=grid[0], fig=fig, cmap=cmap2,
                    vrange=vrangemom2, text='Data',
                    textprop=[dict(size=12)], **kwargs)
        standardfig(raster=mMom2, ax=grid[1], fig=fig, cmap=cmap2,
                    vrange=vrangemom2, beam=False, text='Model',
                    textprop=[dict(size=12)], **kwargs)

        # plot the color bar
        norm = mpl.colors.Normalize(vmin=vrangemom2[0], vmax=vrangemom2[1])
        cmapo = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
        cmapo.set_array([])
        cbr = plt.colorbar(cmapo, cax=grid.cbar_axes[0])
        cbr.ax.set_ylabel('Moment-2', labelpad=-1)

        plt.savefig(outname + '_moment2.pdf', format='pdf', dpi=300)
        plt.close()


def modeldata_comparison(cube, pdffile=None, **kwargs):

    """This will make a six-panel plot for providing an easy overview of the
    the model and the data. Panels are the zeroth moment for the model,
    the data and the residual, the first moment for the model and the data,
    and the residual in the velocity field.

    inputs:
    -------
    cube: Qubefit object that holds the data and the model

    keywords:
    ---------
    pdffile (string|default: None): If set, will save the image to a pdf file

    showmask (Bool|default: False): If set, it will show a contour of the mask
        used in creating the fit (stored in cube.maskarray).
    """

    # create the figure plots
    fig = plt.figure(1, (8., 8.))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 3), axes_pad=0)

    # create the data, model and residual cubes
    dqube = dc(cube)
    mqube = dc(cube)
    mqube.data = cube.model
    rqube = dc(cube)
    rqube.data = cube.data - cube.model

    # moment-zero data
    dMom0 = dqube.calculate_moment(moment=0)
    Mom0sig = (np.sqrt(np.nansum(cube.variance[:, 0, 0])) *
               cube.__get_velocitywidth__())
    clevels = np.insert(np.arange(3, 30, 3), 0, np.arange(-30, 0, 3)) * Mom0sig
    vrangemom = [-3 * Mom0sig, 11 * Mom0sig]
    mask = dMom0.mask_region(value=Mom0sig * 3, applymask=False)
    standardfig(raster=dMom0, contour=dMom0, clevels=clevels, ax=grid[0],
                fig=fig, vrange=vrangemom, cmap='RdYlBu_r', text='Data',
                textprop=[dict(size=12)], **kwargs)

    # moment-zero model
    mMom0 = mqube.calculate_moment(moment=0)
    standardfig(raster=mMom0, contour=mMom0, clevels=clevels, ax=grid[1],
                fig=fig, vrange=vrangemom, cmap='RdYlBu_r', text='Model',
                textprop=[dict(size=12)], **kwargs)

    # moment-zero residual
    rMom0 = rqube.calculate_moment(moment=0)
    standardfig(raster=rMom0, contour=rMom0, clevels=clevels, ax=grid[2],
                fig=fig, vrange=vrangemom, cmap='RdYlBu_r', text='Residual',
                textprop=[dict(size=12)], **kwargs)

    # moment-one data
    dsqube = dqube.mask_region(mask=mask)
    dMom1 = dsqube.calculate_moment(moment=1)
    vrangemom1 = [0.95 * np.nanmin(dMom1.data), 0.95 * np.nanmax(dMom1.data)]
    standardfig(raster=dMom1, ax=grid[3], fig=fig, cmap='Spectral_r',
                vrange=vrangemom1, text='Data',
                textprop=[dict(size=12)], **kwargs)

    msqube = mqube.mask_region(mask=mask)
    mMom1 = msqube.calculate_moment(moment=1)
    standardfig(raster=mMom1, ax=grid[4], fig=fig, cmap='Spectral_r',
                vrange=vrangemom1, beam=False, text='Model',
                textprop=[dict(size=12)], **kwargs)
    
    rMom1 = dc(mMom1)
    rMom1.data = dMom1.data - mMom1.data
    standardfig(raster=rMom1, ax=grid[5], fig=fig, cmap='Spectral_r',
                vrange=vrangemom1, beam=False, text='Residual',
                textprop=[dict(size=12)], **kwargs)
