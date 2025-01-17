# some hopefully useful analysis utils to run after the qubefit chain has been generated. This mostly
# consists out of figures that can be made.
from abc import ABC
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import corner
from copy import deepcopy as dc
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip
import warnings


def standardfig(raster=None, contour=None, rasterselect='data', contourselect='data', newplot=False, ax=None, fig=None,
                origin=None, scale=None, cmap='RdYlBu_r', vrange=None, cbar=False, cbaraxis=None, cbarlabel=None,
                cbarticks=None, tickint=1.0, clevels=None, ccolor='black', beam=True, text=None, textprop=None,
                textposition=None, cross=None, crosssize=None, crosscolor='black', crosslw=2., plotfig=None,
                plotrange=None, flip=False, beamfill=None, beamhatch='////', beamedgecolor='black', **kwargs):
    """ Create a standard figure (raster and/or contour) of a qubefit object.

    This will make a standard figure of a 2D data set, it can be used to either draw contours or filled images
    only or both. It can be used as a stand-alone plotting program if newplot = True or it can plot the data on an
    already defined plot. Note that the data HAS to be 2D to be plotted. If it is not, then you need to first slice
    the data cube using the task get_slice().

    Parameters:
    ---------
    raster:     qube object used to generate the raster image, leave blank or 'None' if not desired.
    contour:    qube object used to generate the contours, leave blank or 'None' if not desired.
    rasterselect: string telling what data to select for the raster image ('data', 'model', residual')
    rasterselect: string telling what data to select for the contours ('data', 'model', residual')
    newplot:    If true then a new plot (ax and fig) is generated using plt.subplots
    origin:     the zero point of the axis
    scale:      pixel scale to used, if not specified will use arcsec as defined in the data header
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
    text:       optional text to be added to the figure can be either a single string or list of strings
    textprop:   dictionary of the properties of the text can be either a single dict or list of dicts
    textposition:    Position of the text box either in an int or string or a list of either
    cross:      if set will draw a cross at the position given [x, y]
    crosssize:  size of the cross in units of the X and Y axis
    crosscolor: color of the cross
    crosslw:    linewidth of the cross
    pdfname:    if set, the name of the PDF file (only works with newplot=True)
    plotrange:   if set, this will set the range of the raster image to plot
    flip:       boolean, if set to true it will flip the x-axis.
    """
    # decide if a new plot needs to be generated
    if newplot:
        __fig_properties__()
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        if ax is None or fig is None:
            raise ValueError('ax and fig need to be set if you do not want a new plot!')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tickint))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tickint))
    # the raster image
    if raster is not None:
        if rasterselect == 'data':
            raster_data = raster.data
        elif rasterselect == 'model':
            raster_data = raster.model
        elif rasterselect == 'residual':
            raster_data = raster.residual
        else:
            raise ValueError('rasterselect needs to be either "data", "model" or "residual"')
        # define the extent of the plot
        if origin is None:
            origin = [(raster_data.shape[0] - 1) // 2, (raster_data.shape[0] - 1) // 2]
        if scale is None:
            scale = np.abs(raster.header['CDELT1'] * 3600)
        if plotrange is None:
            plotrange = __get_plotposrange__(origin, raster_data.shape, scale, flip=flip)
        # define the colormap range, if not set
        if vrange is None:
            vrange = [np.nanmin(raster_data), np.nanmax(raster_data)]
        # plot the raster image
        im = ax.imshow(raster_data, cmap=cmap, origin='lower', extent=plotrange, vmin=vrange[0], vmax=vrange[1])
        # plot the color bar
        if cbar:
            if cbarticks is None:
                cbarticks = np.arange(-10, 10) * (vrange[1] - vrange[0]) / 5.
            elif type(cbarticks) is float or type(cbarticks) is int:
                cbarticks = np.arange(-10, 10) * cbarticks
            if cbaraxis is None:
                cbr = plt.colorbar(im, ticks=cbarticks, ax=ax)
            else:
                cbr = plt.colorbar(im, ticks=cbarticks, cax=cbaraxis)
            if cbarlabel is not None:
                cbr.ax.set_ylabel(cbarlabel, labelpad=-1)
    # contour
    if contour is not None:
        if contourselect == 'data':
            contour_data = contour.data
        elif contourselect == 'model':
            contour_data = contour.model
        elif contourselect == 'residual':
            contour_data = contour.data - contour.model
        else:
            raise ValueError('rasterselect needs to be either "data", "model" or "residual"')
        if origin is None:
            origin = [(contour_data.shape[0]-1) // 2, (contour_data.shape[0]-1) // 2]
        if scale is None:
            scale = np.abs(contour.header['CDELT1'] * 3600)
        position = __get_plotposrange__(origin, contour_data.shape, scale, getposition=True, flip=flip)
        if raster is None and flip:
            ax.set_xlim(np.array([position[0], position[1]]) * scale)
        xc = np.linspace(position[0], position[1], contour_data.shape[-1]) * scale
        yc = np.linspace(position[2], position[3], contour_data.shape[-2]) * scale
        # deal with contour levels
        if clevels is None:
            raise ValueError('Set the contour levels (clevels) keyword.')
        ax.contour(xc, yc, contour_data, levels=clevels, colors=ccolor)
    # the beam
    if beam:
        if raster is not None:
            data = raster
        elif contour is not None:
            data = contour
        else:
            raise ValueError('To draw a beam something has to be drawn')
        ax.add_artist(get_beam(data, ax, scale=scale, flip=flip, fill=beamfill, hatch=beamhatch,
                               color=beamedgecolor, **kwargs))
    # optional text
    if text is not None:
        if textposition is None:
            textposition = [[0.5, 0.85]]
        if textprop is None:
            textprop = [dict(size=12)]
        if type(text) is str:
            text = [text]
        if type(textposition) is int or type(textposition) is str:
            textposition = [textposition]
        if type(textprop) is dict:
            textprop = [textprop]
        for txt, prop, pos in zip(text, textprop, textposition):
            ax.text(pos[0], pos[1], txt, prop, transform=ax.transAxes)
    # add cross
    if cross is not None:
        if crosssize is None:
            crosssize = [1, 1]
        ax.plot(np.array([cross[0], cross[0]]), np.array([cross[1] - crosssize[1], cross[1] + crosssize[1]]),
                lw=crosslw, color=crosscolor)
        ax.plot(np.array([cross[0] - crosssize[0], cross[0] + crosssize[0]]), np.array([cross[1], cross[1]]),
                lw=crosslw, color=crosscolor)
    # finish the new plot (if set)
    if plotfig is True:
        plt.show()
    elif type(plotfig) == str:
        plt.savefig(plotfig, format='pdf', dpi=300)
        plt.close('all')


def make_channelmap(raster=None, contour=None, zeropoint=0., cube_in_mjy=False, channels=None, ncols=4,
                    vrange=(-3, 11), vscale=1, clevels=None, cscale=None, cbar=True, cbarsize=None,
                    cbarlabel=None, cmap='RdYlBu_r', beam=True, plotfig=None, figsize=None, gridsize=None,
                    textprop=None, flip=True, tickint=None, scale=None, **kwargs):
    """
    Create a channelmap of the data, model or residual of the Qube.

    Running this function will create a channel map of the Qube. You can either select the data, the model,
    the residual or nothing to plot for both the raster image and the contour image.

    Parameters:
    -----------
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
    # define the properties of the channel maps
    try:
        tdata = raster
    except AttributeError:
        tdata = contour
    velocity_array = tdata.get_velocity() - zeropoint
    if channels is None:
        channels = np.arange(raster.shape[0])
    if type(channels) is tuple:
        channels = np.arange(channels[0], channels[1])
    vrange = np.array(vrange) * vscale
    if contour is not None and clevels is None:
        if cscale is not None:
            psqrt2 = np.power(np.sqrt(2), np.arange(15))
            clevels = np.insert(2 * psqrt2, 0, -2 * np.flip(psqrt2)) * cscale
        else:
            raise IOError('Need to supply either contour levels or set the cscale keyword')
    # now generate a grid with the specified number of columns
    nrows = np.ceil(len(channels) / ncols).astype(int)
    if figsize is None:
        figsize = (8, 8 / ncols * nrows)
    if gridsize is None:
        gridsize = (0.08, 0.06, 0.90, 0.82)
    if textprop is None:
        textprop = {'bbox': {'facecolor': 'white', 'alpha': 0.8, 'pad': 2}, 'ha': 'center'}
    if scale is None:
        scale = np.abs(tdata.header['CDELT1'] * 3600)
    if tickint is None:
        ti = scale * tdata.header['NAXIS1'] / 3
        tickint = round(ti, 1 - int(np.floor(np.log10(abs(ti)))) - 1)
    __fig_properties__()
    fig = plt.figure(1, figsize=figsize)
    grid = ImageGrid(fig, gridsize, nrows_ncols=(nrows, ncols), axes_pad=0., cbar_mode=None, share_all=True)
    # now loop over the channels to be plotted
    for idx, chan in enumerate(channels):
        # get the string value of the velocity
        velocity_string = str(int(round(velocity_array[chan]))) + ' km s$^{-1}$'
        beambool, rasterimage, contourimage = None, None, None
        if (idx % ncols == 0) and (idx // ncols == int(nrows) - 1) and beam:
            beambool = True
        if raster is not None:
            rasterimage = raster.get_slice(zindex=(chan, chan+1))
        if contour is not None:
            contourimage = contour.get_slice(zindex=(chan, chan+1))
        standardfig(raster=rasterimage, contour=contourimage, clevels=clevels, fig=fig, ax=grid[idx], cbar=False,
                    beam=beambool, vrange=vrange, text=[velocity_string], cmap=cmap, textprop=textprop, flip=flip,
                    scale=scale, tickint=tickint, **kwargs)
    # plot the colorbar
    if cbar is True and raster is not None:
        if cbarsize is None:
            cbarsize = [0.12, 0.90, 0.82, 0.03]
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vrange[0], vrange[1]), cmap=cmap)
        cbaraxes = fig.add_axes(cbarsize)
        cbar = fig.colorbar(sm, cax=cbaraxes, orientation='horizontal')
        cbar.ax.tick_params(axis='x', direction='out', top=True, bottom=False, labelbottom=False, labeltop=True)
        if cbarlabel is not None:
            cbar.ax.set_ylabel(cbarlabel, labelpad=-1)
    # add text
    if cube_in_mjy:
        fig.text(0.52, 0.97, 'Flux density (mJy beam$^{-1}$)', fontsize=13, ha='center')
    else:
        fig.text(0.52, 0.97, 'Flux density (Jy beam$^{-1}$)', fontsize=13, ha='center')
    fig.text(0.52, 0.015, '$\\Delta$ R.A. (arcsec)', fontsize=13, ha='center')
    fig.text(0.015, 0.5, '$\\Delta$ Decl. (arcsec)', fontsize=13, va='center', rotation=90)
    # save, show or return the figure
    return __returnfig__(plotfig, fig)


def make_1panelfigure(raster=None, contour=None, label='cust', cont_in_ujy=False, vrange=(-3, 11),
                      vscale=1, clevels=None, cscale=None, custtext='', plotfig=None, figsize=(4.6, 4),
                      cbar=True, flip=True, figadjust=None, **kwargs):
    """
    Create a 1-panel figure with nice axis and labels

    This will make a standard single panel figure. Unlike standard_fig, which it calls, it will generate 'nice'
    axes with labels. The plots generated using this program have been used in several publications.

    Parameters:
    ---------
    plot : STRING
        This parameter determines what to plot and can be one of the following: 'cont', 'mom0', 'mom1', 'mom2',
        or 'cust'. The default is None, which will result in an InputError. The plot keywords correspond to the
        keys in the dictionary that the get_data returns, so see this function's documentation string for more
        information.
    tick_value : FLOAT
        The tick sapcing to use along the X and Y axis. The default is 1.
    vrange : TUPLE
        Range of the color map used in the plot. Together with the in_sigma parameter, this allows the selection
        of the range in terms of the RMS of the data. The default is (-3, 11).
    """
    # define the plot properties
    vrange = np.array(vrange) * vscale
    if contour is not None and clevels is None:
        if cscale is not None:
            psqrt2 = np.power(np.sqrt(2), np.arange(15))
            clevels = np.insert(2 * psqrt2, 0, -2 * np.flip(psqrt2)) * cscale
        else:
            raise IOError('Need to supply either a contour levels or set the cscale keyword')
    # create the figure
    __fig_properties__()
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    if figadjust is None:
        figadjust = [0.08, 0.99, 0.90, 0.12]
    plt.subplots_adjust(left=figadjust[0], right=figadjust[1], top=figadjust[2], bottom=figadjust[3])
    standardfig(raster=raster, contour=contour, ax=ax, fig=fig, vrange=vrange, cbar=cbar, clevels=clevels,
                flip=flip, **kwargs)
    # Figure text
    text = {'cont': r'Continuum flux density (Jy beam$^{-1}$)',
            'mom0': r'Integrated [CII] flux (Jy km s$^{-1}$ beam$^{-1}$)',
            'mom1': r'Mean velocity (km s$^{-1}$)',
            'mom2': r'Velocity dispersion (km s$^{-1}$)',
            'cust': custtext}
    if cont_in_ujy:
        text['cont'] = r'Continuum flux density ($\mu$Jy beam$^{-1}$)'
    fig.text(0.5, 0.93, text[label], fontsize=14, color='black', ha='center')
    fig.text(0.5, 0.03, r'$\Delta$ R.A. (arcsec)', fontsize=14,
             ha='center')
    fig.text(0.03, 0.5, r'$\Delta$ Decl. (arcsec)', fontsize=14,
             va='center', rotation=90)
    # save, show or return the figure
    return __returnfig__(plotfig, fig)


def make_4panelfigure(raster=None, contour=None, label=None, cont_in_ujy=False, vrange=None, vscale=None,
                      clevels=None, cscale=None, custtext='', figsize=(8, 7.5), cbar=None, cmap=None, flip=True,
                      gridsize=None, plotfig=None, cbarticks=None, **kwargs):
    """
    Create a 4-panel figure with nice axis and labels

    This will make a standard four panel figure. The default is to plot the continuum and the first three moments,
    and show the contour for the first two. However, this figure can be customized relatively easily.

    Parameters:
    ---------
    raster : list of four QUBE instances or None
        This parameter determines what to plot and can be one of the following: 'cont', 'mom0', 'mom1', 'mom2',
        or 'cust'. The default is None, which will result in an InputError.
    contour : list of four QUBE instances or None
        This parameter determines what contours to plot (if any).
    label : list of four STRINGS
        This parameter will determine what label (if any) to use for the panel. If label is 'cust' the label will be
        taken from the custtext keyword.
    """
    # assign the lists in case they are not defined
    if cbarticks is None:
        cbarticks = [50, 0.2, 50, 50]
    if cmap is None:
        cmap = ['RdYlBu_r', 'RdYlBu_r', 'Spectral_r', 'bone_r']
    if cbar is None:
        cbar = [True, True, True, True]
    if vscale is None:
        vscale = [1, 1, 1, 1]
    if vrange is None:
        vrange = [(-3, 11), (-3, 11), (-200, 200), (0, 250)]
    if cscale is None:
        cscale = [None, None, None, None]
    if clevels is None:
        clevels = [None, None, None, None]
    if contour is None:
        contour = [None, None, None, None]
    if label is None:
        label = ['cont', 'mom0', 'mom1', 'mom2']
    if raster is None:
        raster = [None, None, None, None]
    # make the figure
    __fig_properties__()
    fig = plt.figure(figsize=figsize)
    if gridsize is None:
        gridsize = (0.08, 0.08, 0.86, 0.86)
    grid = ImageGrid(fig, gridsize, nrows_ncols=(2, 2), axes_pad=0.5, cbar_mode='each',
                     cbar_location='right', cbar_pad=0.0)
    for idx, ax, cax in zip(np.arange(4), grid, grid.cbar_axes):
        vrange[idx] = np.array(vrange[idx]) * vscale[idx]
        if contour[idx] is not None and clevels[idx] is None:
            if cscale[idx] is not None:
                psqrt2 = np.power(np.sqrt(2), np.arange(15))
                clevels[idx] = np.insert(2 * psqrt2, 0, -2 * np.flip(psqrt2)) * cscale[idx]
            else:
                raise IOError('Need to supply either a contour levels or set the cscale keyword')
        standardfig(raster=raster[idx], contour=contour[idx], ax=ax, fig=fig, cmap=cmap[idx], vrange=vrange[idx],
                    cbar=cbar[idx], cbaraxis=cax, cbarticks=cbarticks[idx], clevels=clevels[idx], flip=flip, **kwargs)
    # Figure text
    text = {'cont': r'Continuum flux density (Jy beam$^{-1}$)',
            'mom0': r'Integrated line flux (Jy km s$^{-1}$ beam$^{-1}$)',
            'mom1': r'Mean velocity (km s$^{-1}$)',
            'mom2': r'Velocity dispersion (km s$^{-1}$)',
            'cust': custtext}
    if cont_in_ujy:
        text['cont'] = r'Continuum flux density ($\mu$Jy beam$^{-1}$)'
    fig.text(0.29, 0.96, text[label[0]], fontsize=13, color='black', ha='center')
    fig.text(0.75, 0.96, text[label[1]], fontsize=13, color='black', ha='center')
    fig.text(0.29, 0.49, text[label[2]], fontsize=14, color='black', ha='center')
    fig.text(0.75, 0.49, text[label[3]], fontsize=14, color='black', ha='center')
    fig.text(0.5, 0.02, r'$\Delta$ R.A. (arcsec)', fontsize=20, ha='center')
    fig.text(0.01, 0.5, r'$\Delta$ Decl. (arcsec)', fontsize=20, va='center', rotation=90)
    # save, show or return the figure
    return __returnfig__(plotfig, fig)


def make_momentcomparisonfigure(qf_object, channels=None, plotfig=None, mom0_rms=None, mask_rms=3, quick=True,
                                tickint=None, scale=None, **kwargs):
    """
    Make a 6-panel figure comparing the integrated line flux density and velocity field for the model and data

    This will make a six-panel plot for providing an overview of the the model and the data. The top row of three
    panels are the zeroth moment (the integrated line flux density) of the model, the data and the residual.
    The bottom row shows the first moment for the model, the data and the residual of the velocity field.

   Parameters:
   -------
   qf_object: QUBEFIT object
       Qubefit object that holds the data and the model
   """
    # calculate the moments
    mom0d, mom1d, _mom2d, mask = get_moments(qf_object, channels=channels, quick=quick, mask_rms=mask_rms,
                                             mom0_rms=mom0_rms, return_mask=True)
    mom0m, mom1m, _mom2m = get_moments(qf_object, channels=channels, quick=quick, mask_rms=mask_rms, mom0_rms=mom0_rms,
                                       use_model=True, mask=mask)
    mom0r, mom1r = dc(mom0m), dc(mom1m)
    mom0r.data, mom1r.data = mom0d.data - mom0m.data, mom1d.data - mom1m.data
    psqrt2 = np.power(np.sqrt(2), np.arange(15))
    clevels = np.insert(2 * psqrt2, 0, -2 * np.flip(psqrt2)) * mom0d.header['rms']
    vrange_1 = np.array([-3, 11]) * mom0d.header['rms']
    vrange_2 = [np.nanmin(mom1m.data), np.nanmax(mom1m.data)]
    # make the figure
    if scale is None:
        scale = np.abs(qf_object.header['CDELT1'] * 3600)
    if tickint is None:
        ti = scale * qf_object.header['NAXIS1'] / 5
        tickint = round(ti, 1 - int(np.floor(np.log10(abs(ti)))) - 1)
    __fig_properties__()
    mpl.rc('contour', linewidth=0.8)
    fig, ax = plt.subplots(2, 3, figsize=(8, 5.3), sharex='all', sharey='all')
    plt.subplots_adjust(left=0.09, right=0.89, top=0.95, bottom=0.1, wspace=0.1, hspace=0.2)
    im = {1: {'raster': mom0d, 'cont': mom0d, 'ax': ax[0, 0], 'beam': True, 'text': 'Data', 'cmap': 'RdYlBu_r',
              'vrange': vrange_1},
          2: {'raster': mom0d, 'cont': mom0m, 'ax': ax[0, 1], 'beam': False, 'text': 'Model', 'cmap': 'RdYlBu_r',
              'vrange': vrange_1},
          3: {'raster': mom0d, 'cont': mom0r, 'ax': ax[0, 2], 'beam': False, 'text': 'Residual', 'cmap': 'RdYlBu_r',
              'vrange': vrange_1},
          4: {'raster': mom1d, 'cont': None, 'ax': ax[1, 0], 'beam': True, 'text': 'Data', 'cmap': 'Spectral_r',
              'vrange': vrange_2},
          5: {'raster': mom1m, 'cont': None, 'ax': ax[1, 1], 'beam': False, 'text': 'Model', 'cmap': 'Spectral_r',
              'vrange': vrange_2},
          6: {'raster': mom1r, 'cont': None, 'ax': ax[1, 2], 'beam': False, 'text': 'Residual', 'cmap': 'Spectral_r',
              'vrange': vrange_2}}
    for idx in im:
        standardfig(raster=im[idx]['raster'], contour=im[idx]['cont'], ax=im[idx]['ax'], fig=fig, cmap=im[idx]['cmap'],
                    vrange=im[idx]['vrange'], cbar=False, clevels=clevels, flip=True, tickint=tickint, scale=scale,
                    beam=im[idx]['beam'], text=im[idx]['text'], textposition=[[0.07, 0.87]],
                    textprop={'bbox': {'facecolor': 'white', 'alpha': 0.8, 'pad': 2}}, **kwargs)
    # add color bars
    sm_1 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vrange_1[0], vrange_1[1]), cmap=im[1]['cmap'])
    sm_2 = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vrange_2[0], vrange_2[1]), cmap=im[4]['cmap'])
    axins_1 = inset_axes(im[3]['ax'], width="5%", height="100%", loc='center right', borderpad=-2.5)
    axins_2 = inset_axes(im[6]['ax'], width="5%", height="100%", loc='center right', borderpad=-2.5)
    cb_1 = fig.colorbar(sm_1, cax=axins_1, orientation="vertical")
    cb_2 = fig.colorbar(sm_2, cax=axins_2, orientation="vertical")
    cb_1.ax.tick_params(axis='x', direction='out')
    cb_2.ax.tick_params(axis='x', direction='out')
    # add text
    fig.text(0.52, 0.96, r'Integrated line flux density (Jy km s$^{-1}$)', fontsize=12, color='black', ha='center')
    fig.text(0.52, 0.50, r'Velocity field (km s$^{-1}$)', fontsize=12, color='black', ha='center')
    fig.text(0.52, 0.015, '$\\Delta$ R.A. (arcsec)', fontsize=12, ha='center')
    fig.text(0.015, 0.5, '$\\Delta$ Decl. (arcsec)', fontsize=12, va='center', rotation=90)
    # save, show or return the figure
    return __returnfig__(plotfig, fig)

    pass


def make_mcmcchainfigure(qf_object, figsize=None, adjust_size=None, plotfig=None):
    """
    Make a figure of the chain for each parameter in the model

    This makes a figure of the chain for each parameter. Each line is a single walker in the chain. If you want
    to show the full chain it clearly is important to read in the full chain (i.e., burnin=0.0 needs to be set, if
    you have read the chain in with get_chainresults().
    """
    # generate a grid with two columns and enough rows to accomodate all of the parameters and probability
    ncols = 2
    nrows = np.ceil((qf_object.mcmcarray.shape[2] + 1) / ncols).astype(int)
    if figsize is None:
        figsize = (8, 1.3 * nrows)
    if adjust_size is None:
        adjust_size = [0.08, 0.98, 0.98, 0.08, 0.0, 0.2]
    __fig_properties__()
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows, ncols*3+1, left=adjust_size[0], right=adjust_size[1], top=adjust_size[2],
                           bottom=adjust_size[3], wspace=adjust_size[4], hspace=adjust_size[5])
    # now loop over the parameters that need to be plotted
    idx = 0
    for idx, par in enumerate(qf_object.mcmcmap):
        row = idx // 2
        if np.mod(idx, 2) == 0:
            ax = fig.add_subplot(gs[row, :3])
        else:
            ax = fig.add_subplot(gs[row, 4:])
        for walker in np.arange(qf_object.mcmcarray.shape[1]):
            if qf_object.initpar[par]['Conversion'] is not None:
                yval = dc(qf_object.mcmcarray[:, walker, idx] * qf_object.initpar[par]['Conversion'].value)
            else:
                yval = dc(qf_object.mcmcarray[:, walker, idx])
            ax.plot(np.arange(qf_object.mcmcarray.shape[0]), yval, alpha=0.05, color='black')
        ax.set_ylabel('{0} ({1})'.format(par, qf_object.initpar[par]['Unit'].to_string()))
        ax.set_xlabel('Step')
    # final probability plot
    if np.mod(idx + 1, 2) == 0:
        ax = fig.add_subplot(gs[(idx + 1) // 2, :3])
    else:
        ax = fig.add_subplot(gs[(idx + 1) // 2, 4:])
    for walker in np.arange(qf_object.mcmclnprob.shape[1]):
        ax.plot(np.arange(qf_object.mcmclnprob.shape[0]), qf_object.mcmclnprob[:, walker], alpha=0.05, color='black')
        ax.set_ylabel('ln_prob')
        ax.set_xlabel('Step')
    # save, show or return the figure
    return __returnfig__(plotfig, fig)


def make_cornerfigure(qf_object, plotfig=None, show_titles=True, **kwargs):
    parameters = qf_object.mcmcmap
    flat_chain = dc(qf_object.mcmcarray.reshape((-1, qf_object.mcmcarray.shape[2])))
    # convert to physical units
    for par in parameters:
        idx = qf_object.mcmcmap.index(par)
        if qf_object.initpar[par]['Conversion'] is not None:
            flat_chain[:, idx] *= qf_object.initpar[par]['Conversion'].value
    # plot using corner package
    fig = corner.corner(flat_chain, quantiles=[0.16, 0.5, 0.84], labels=parameters, color='midnightblue',
                        show_titles=show_titles, **kwargs)
    # save, show or return the figure
    return __returnfig__(plotfig, fig)


def diagnostic_plots(qf_object, chainfile=None, burnin=0.3, mcmcburnin=0.0, load_best=False, corner_figure=True,
                     momentcomparison_figure=True, channelmap_figure=True, mcmcchain_figure=True,
                     cube_rms=None, mom0_rms=None, figure_root='./model'):
    """ Create simple diagnostic plots

    This program will create several diagnostic plots of a given model and MCMC chain. The plots are likely not
    paper-ready but are meant to understand how well the MCMC chain ran. The following plots can be generated:

    Parameters:
    -----------
    corner_figure : BOOL
        Creates a corner figure of the mcmc chain in physical units. This can help understand dependencies between the
        parameters and give a quick view of the mean parameter values and 16th and 84th percentile ranges.
    momentcomparison_figure : BOOL
        Creates the integrated line flux density (moment-0) and velocity field (moment-1) for both the data and
        the model. In the final panel, the residual (data - model) is shown. This plot gives an overall view of the
        goodness of fit, but the channel map figure is more sensitive to differences between the model and the data.
    channelmap_figure : BOOL
        Creates three different channel map figures. In all cases the data is shown in the raster image. However,
        the figures have different contours plotted (data, model and residual) each at the same level. This provides
        a good view of the small differences between the data and model.
    mcmcchain_figure : BOOL
        Creates plots of the MCMC chains for each of the parameters. This figure is useful to see if and when the
        data has converged (This is most useful if burnin=0.0, so none of the chain has been discarded).

    Other Parameters
    ----------------
    chainfile : STRING or NONE
        If given, the model and model parameters will be updated with this new chainfile. This is the recommened way
        to load in the
    burnin : FLOAT
        If chainfile is given, this burn-in value will be used to discard the first part of the chain. The default is
        0.3.
    mcmcburnin : FLOAT
        If chainfile is given, this burn-in value will be used to discard the first part of the chain for plotting the
        mcmc chain only. The default is 0.0.
    load_best : BOOL
        If chainfile is given and load_best is True, then the lowest probability parameter solution will be chosen,
        if load_best is False, then the median solution is chosen
    plot_root : STRING
        The root of the file names used. For each of the figures a different string will be appended to this root name.
    """
    # do the mcmc chain first (because of the different burn-in range).
    if mcmcchain_figure:
        print('Creating the mcmc chain figure.')
        if chainfile is not None:
            qf_object.get_chainresults(chainfile, burnin=mcmcburnin, load_best=load_best)
        file_name = figure_root + '_mcmcchain.pdf'
        make_mcmcchainfigure(qf_object, plotfig=file_name)
    # read in the chain data file and load in the model (either median or lowest probability)
    if chainfile is not None:
        qf_object.get_chainresults(chainfile, burnin=burnin, load_best=load_best)
    # get the cube_rms if it is not defined
    if cube_rms is None:
        cube_rms = np.median(qf_object.calculate_sigma())
    # corner
    if corner_figure:
        print('Creating the corner figure.')
        file_name = figure_root + '_corner.pdf'
        make_cornerfigure(qf_object, plotfig=file_name)
    # moment comparison
    if momentcomparison_figure:
        print('Creating the moment comparison figure.')
        file_name = figure_root + '_momentcomp.pdf'
        make_momentcomparisonfigure(qf_object, plotfig=file_name, mom0_rms=mom0_rms)
    # channels maps
    if channelmap_figure:
        print('Creating the channel map figures.')
        file_name = figure_root + '_channelmap_datacontours.pdf'
        make_channelmap(raster=qf_object, contour=qf_object, vscale=cube_rms, cscale=cube_rms, plotfig=file_name)
        file_name = figure_root + '_channelmap_modelcontours.pdf'
        make_channelmap(raster=qf_object, contour=qf_object, contourselect='model', vscale=cube_rms, cscale=cube_rms,
                        plotfig=file_name)
        file_name = figure_root + '_channelmap_residualcontours.pdf'
        make_channelmap(raster=qf_object, contour=qf_object, contourselect='residual', vscale=cube_rms,
                        cscale=cube_rms, plotfig=file_name)


def get_moments(qf_object, channels, mask_rms=3, quick=False, mom0_rms=None, use_model=False, mask=None,
                size=None, center=None, return_mask=False):
    if center is None:
        center = [qf_object.data.shape[2] // 2, qf_object.data.shape[1] // 2]
    # full moment-0
    mom0 = qf_object.calculate_moment(moment=0, channels=channels, use_model=use_model)
    if mom0_rms is None:
        mom0_rms = mom0.calculate_sigma()
    mom0.header['rms'] = mom0_rms
    if size is not None:
        qf_object = qf_object.get_slice(xindex=(center[0] - size, center[0] + size + 1),
                                        yindex=(center[1] - size, center[1] + size + 1))
        mom0 = mom0.get_slice(xindex=(center[0] - size, center[0] + size + 1),
                              yindex=(center[1] - size, center[1] + size + 1))
    if mask is None:
        mask = mom0.mask_region(value=mom0_rms * mask_rms, applymask=False)
    qf_object_masked = qf_object.mask_region(value=0.0)
    mom1 = qf_object.calculate_moment(moment=1, channels=channels, use_model=use_model)
    mom2 = qf_object_masked.calculate_moment(moment=2, channels=channels, use_model=use_model)
    if not quick:
        mom1, mom2 = qf_object.gaussian_moment(mom1=mom1, mom2=mom2, use_model=use_model)
    mom1 = mom1.mask_region(mask=mask)
    mom2 = mom2.mask_region(mask=mask)
    if return_mask:
        return mom0, mom1, mom2, mask
    else:
        return mom0, mom1, mom2


def get_beam(qube, ax, loc='lower left', pad=0.3, borderpad=0.4, frameon=True, color='Darkslategray', alpha=1.0,
             wcs=None, prop=None, scale=None, flip=False, **kwargs):
    # Grab the Beam
    width, height, angle = __get_beamproperties__(qube, scale=scale, wcs=wcs)
    # flip the beam if requested
    if flip:
        angle = -1 * angle
    # Note: not set up to deal with different x,y, pixel scales
    beam = AnchoredEllipse(ax.transData, width=width, height=height,
                           angle=angle, loc=loc, pad=pad, borderpad=borderpad,
                           frameon=frameon, facecolor=color, alpha=alpha,
                           prop=prop, **kwargs)
    return beam


def get_lineproperties(qube, pos, sig=None, radius=None, radius_in_arcsec=True, guessmean=None, pbcor=1.,
                       cont_correct=False, lim=None, return_spec=False, guessamp=None, guesssig=None, **kwargs):
    # ignore runtime warnings in fitting
    warnings.filterwarnings("ignore")
    # calculate sigma
    bmaj = qube.beam['BMAJ'] / np.sqrt(8 * np.log(2)) / np.abs(qube.header['CDELT1'])
    bmin = qube.beam['BMIN'] / np.sqrt(8 * np.log(2)) / np.abs(qube.header['CDELT1'])
    beamarea = bmaj * bmin * 2 * np.pi
    if sig is None:
        sigmapb = qube.calculate_sigma() * pbcor
    else:
        sigmapb = sig * pbcor
    # get xy coordinates in pixels
    if type(pos) is str:
        wcs = WCS(qube.header)
        sky_coord = SkyCoord(pos, unit=(u.hourangle, u.deg))
        pos = sky_coord.to_pixel(wcs)
    else:
        pos = (np.array(pos[0]), np.array(pos[1]))
    # get radius in pixels and get spectrum, nu and v array
    if radius is None:
        int_pos = np.rint(pos).astype(int)
        snu = qube.data[:, int_pos[1], int_pos[0]]
        v = qube.get_velocity()
        if cont_correct:
            snu = __correct_flux__(snu, v, lim)
        dv = qube.get_velocitywidth()
    else:
        if radius_in_arcsec:
            rad = radius / np.abs(qube.header['CDELT1'] * 3600)
        else:
            rad = radius
        qube_masked = qube.mask_region(ellipse=[pos[0], pos[1], rad, rad, 0])
        sigmapb *= np.sqrt(np.pi * rad**2 / beamarea)
        snu, v = qube_masked.get_spec1d(continuum_correct=cont_correct, limits=lim, **kwargs)
        dv = qube_masked.get_velocitywidth()
    snupb = snu * pbcor
    # Gaussian fit of the spectrum
    if guessmean is None:
        if np.where(snupb > 2 * sigmapb)[0].size > 0:
            guessmean = v[np.median(np.where(snupb > 2 * sigmapb)[0]).astype(int)]
        else:
            guessmean = v[int(len(v) / 2)]
    if guessamp is None:
        if np.where(np.abs(v - guessmean) < dv / 2)[0].size > 0:
            guessamp = snupb[np.where(np.abs(v - guessmean) < dv / 2)[0]]
        else:
            guessamp = np.max(snupb)
    if guesssig is None:
        guesssig = 100
    gausspar = [guessamp, guessmean, guesssig]
    g_init = models.Gaussian1D(amplitude=gausspar[0], mean=gausspar[1], stddev=gausspar[2])
    fit_g = fitting.LevMarLSQFitter()
    g = fit_g(g_init, v, snupb)
    fwhm, mean_v, amp = np.sqrt(8 * np.log(2)) * g.stddev.value, g.mean.value, g.amplitude.value
    mean_nu = qube.header['restfrq'] * (1 - mean_v / 299792.458)
    snudv = 1.065 * fwhm * amp
    # calculate Gaussian fit uncertainties (Lenz & Ayres 1992, PSAP, 104, 1104)
    idx = np.where(np.abs(v - mean_v) < 5 * dv)
    meansigmapb = np.nanmean(sigmapb[idx])
    factor = np.sqrt(fwhm / dv) * (amp / meansigmapb)
    dfwhm, dmean_v, damp = fwhm / (0.60 * factor), fwhm / (1.47 * factor), amp / (0.70 * factor)
    dmean_nu = qube.header['restfrq'] / 299792.458 * dmean_v
    dsnudv = snudv / (0.70 * factor)
    # calculate the observed properties (relies on the Gaussian fit for initial estimates).
    full_idx = np.where((v > mean_v - fwhm) & (v < mean_v + fwhm))
    opt_idx = np.where((v > mean_v - 0.6 * fwhm) & (v < mean_v + 0.6 * fwhm))
    hig_idx = np.where((v > mean_v - fwhm) & (v < mean_v + fwhm) & (snupb > amp - sigmapb))
    amp_obs = np.mean(snupb[hig_idx])
    fwhm_obs = __get_obsfwhm__(snupb, dv, amp_obs)
    linedict = {'fitted': {'amp': amp, 'damp': damp,
                           'mean_v': mean_v, 'dmean_v': dmean_v,
                           'mean_nu': mean_nu, 'dmean_nu': dmean_nu,
                           'fwhm': fwhm, 'dfwhm': dfwhm,
                           'snudv': snudv, 'dsnudv': dsnudv,
                           'snr': snudv / dsnudv},
                'observed': {'amp': np.mean(snupb[hig_idx], dtype=np.float64),
                             'damp': np.sqrt(np.sum(np.square(sigmapb[hig_idx]))) / sigmapb[hig_idx].size,
                             'mean_v': np.sum(v[full_idx] * snupb[full_idx]) / np.sum(snupb[full_idx]),
                             'dmean_v': fwhm_obs / (1.47 * factor),
                             'mean_nu': (qube.header['restfrq'] *
                                         (1 - np.sum(v[full_idx] * snupb[full_idx]) / np.sum(snupb[full_idx]) / 299792.458)),
                             'dmean_nu': qube.header['restfrq'] / 299792.458 * fwhm_obs / (1.47 * factor),
                             'fwhm': fwhm_obs,
                             'dfwhm': fwhm_obs / (0.60 * factor),
                             'snudv': np.sum(snupb[full_idx]) * dv,
                             'dsnudv': np.sqrt(np.sum(np.square(sigmapb[full_idx]))) * dv,
                             'snr': np.sum(snupb[full_idx]) / np.sqrt(np.sum(np.square(sigmapb[full_idx]))),
                             'snr_opt': np.sum(snupb[opt_idx]) / np.sqrt(np.sum(np.square(sigmapb[opt_idx])))}}
    if return_spec:
        spec = {'velocity': v, 'flux': snupb, 'sigma': sigmapb}
        return linedict, spec
    else:
        return linedict


class AnchoredEllipse(AnchoredOffsetbox, ABC):
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


def __get_beamproperties__(qube, scale=None, wcs=None):
    """
    Grab the beam properties from the header in the qube class.
    optional keywords are the scale which can be set to be
    either in physical units, pixels or defaults to the units in the
    header as long as wcs is defined.
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
            scale = np.abs(qube.header["CDELT1"] / cdelt)
        else:
            scale = 1.
    # Parse
    width = qube.beam["BMAJ"] / qube.header["CDELT1"] * scale
    height = qube.beam["BMIN"] / qube.header["CDELT1"] * scale
    angle = qube.beam["BPA"]+90.
    # Return
    return width, height, angle


def __get_plotposrange__(origin, shape, scale, getposition=False, flip=False):
    # position and plotrange
    position = np.array([-origin[0], shape[1]-origin[0]-1,
                         -origin[1], shape[0]-origin[1]-1])
    pltrange = np.array([position[0]-0.5, position[1]+0.5,
                         position[2]-0.5, position[3]+0.5]) * scale
    # flip the position and position in the x-direction
    if flip:
        position = np.array([-1*position[0], -1*position[1], position[2], position[3]])
        pltrange = np.array([-1*pltrange[0], -1*pltrange[1], pltrange[2], pltrange[3]])
    if getposition:
        return position
    else:
        return pltrange


def __fig_properties__():
    mpl.rcdefaults()
    font = {'family': 'DejaVu Sans', 'weight': 'normal',
            'size': 10}
    mpl.rc('font', **font)
    mpl.rc('mathtext', fontset='dejavusans')
    mpl.rc('axes', lw=1)
    mpl.rc('xtick.major', size=4, width=1)
    mpl.rc('ytick.major', size=4, width=1)
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'


def __returnfig__(plotfig, fig):
    if plotfig is None:
        return fig
    elif type(plotfig) is str:
        plt.savefig(plotfig, format='pdf', dpi=300)
        plt.close('all')
    else:
        plt.show()


def __get_obsfwhm__(snu, dv, amp):
    fwhm_idx = np.where(snu > amp / 2)[0]
    count = 0
    for x in fwhm_idx:
        if x - 1 not in fwhm_idx:
            y = x + 1
            while y in fwhm_idx:
                y += 1
            count = max(count, y - x)
    return count * dv


def __correct_flux__(flux, vel, limits):
    """
    Correct the flux for continuum wiggles.

    This will fit the spectrum for a potential continuum residual or
    wiggles. The fitting function of the continuum is a second order
    polynomial. The fit will ignore the region within the limits.
    """
    # fit the spectrum for potential continuum residual
    # (second order polynomial)

    finit = models.Polynomial1D(2, c0=0, c1=0, c2=0)
    fitter = fitting.LevMarLSQFitter()
    ofitter = fitting.FittingWithOutlierRemoval(fitter, sigma_clip, niter=3,
                                                sigma=3.0)
    FitIdx = (vel < limits[0]) + (vel > limits[1])
    OFit, OFitData = ofitter(finit, vel[FitIdx], flux[FitIdx])

    return flux - OFit(vel)
