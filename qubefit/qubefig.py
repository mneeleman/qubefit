from qubefit.qube import Qube
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from qubefit.qfutils import standardfig
from astropy.io import ascii
from astropy.table import Table


def make_4panelfigure(tick_value=1.0, cont_range=(-3, 11), mom0_range=(-3, 11),
                      mom1_range=(-200, 200), mom2_range=(0, 250), in_sigma=True,
                      cont_ticks=50, mom0_ticks=0.2, mom1_ticks=50, mom2_ticks=50,
                      do_contour=(True, True, False, False), clevels=None,
                      cmaps=('RdYlBu_r', 'RdYlBu_r', 'Spectral_r', 'Spectral_r'),
                      cont_in_ujy=True, plot_fig=None, **kwargs):

    # get_data
    data = get_data(**kwargs)
    if cont_in_ujy:
        data['cont'].data *= 1E6
        data['cont_rms'] *= 1E6

    # define plot properties from the data
    if in_sigma:
        v_ranges = [np.array(cont_range) * data['cont_rms'],
                    np.array(mom0_range) * data['mom0_rms'],
                    mom1_range, mom2_range]
    else:
        v_ranges = [cont_range, mom0_range, mom1_range, mom2_range]
    v_ticks = [cont_ticks, mom0_ticks, mom1_ticks, mom2_ticks]
    if clevels is None and do_contour:  # standard contours start at 3sigma and increase in powers of sqrt(2)
        psqrt2 = np.power(np.sqrt(2), np.arange(15))
        clevels = [np.insert(3 * psqrt2, 0, -3 * np.flip(psqrt2)) * data['cont_rms'],
                   np.insert(3 * psqrt2, 0, -3 * np.flip(psqrt2)) * data['mom0_rms'],
                   50, 50]

    # create the figure
    __fig_properties__()
    fig = plt.figure(1, (8., 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5, cbar_mode='each',
                     cbar_location='right', cbar_pad=0.0)
    iterable = zip(grid, grid.cbar_axes, [data['cont'], data['mom0'], data['mom1'], data['mom2']],
                   v_ranges, v_ticks, cmaps, do_contour, clevels)
    for ax, cax, qf_object, v_range, v_tick, cmap, do_cont, c_level in iterable:
        if do_cont:
            standardfig(raster=qf_object, contour=qf_object, ax=ax, fig=fig, origin=data['origin'],
                        scale=data['scale'], cmap=cmap, vrange=v_range, cbar=True, cbaraxis=cax,
                        vscale=v_tick, tickint=tick_value, clevels=c_level, flip=True)
        else:
            standardfig(raster=qf_object, ax=ax, fig=fig, origin=data['origin'],
                        scale=data['scale'], cmap=cmap, vrange=v_range, cbar=True, cbaraxis=cax,
                        vscale=v_tick, tickint=tick_value, flip=True)

    # Figure text
    fig.text(0.10, 0.89, r'Continuum flux density ($\mathrm{\mu}$Jy beam$^{-1}$)',
             fontsize=12, color='black')
    fig.text(0.53, 0.89, r'Integrated [CII] flux (Jy km s$^{-1}$ beam$^{-1}$)',
             fontsize=12, color='black')
    fig.text(0.18, 0.48, r'Mean velocity (km s$^{-1}$)', fontsize=14, color='black')
    fig.text(0.57, 0.48, r'Velocity dispersion (km s$^{-1}$)', fontsize=14,
             color='black')

    fig.text(0.5, 0.04, r'$\Delta$ R.A. (arcsec)', fontsize=20, ha='center')
    fig.text(0.03, 0.5, r'$\Delta$ Decl. (arcsec)', fontsize=20, va='center',
             rotation=90)

    # save the figure
    if plot_fig is None:
        return fig
    else:
        __save_fig__(plot_fig)


def make_3panelfigure(tick_value=1.0, mom0_range=(-3, 11),
                      mom1_range=(-200, 200), mom2_range=(0, 250), in_sigma=True,
                      mom0_ticks=0.2, mom1_ticks=50, mom2_ticks=50, do_contours=(True, False, False),
                      cmaps=('RdYlBu_r', 'Spectral_r', 'Spectral_r'), plto_fig=None, **kwargs):

    # load the cube and cont, trim, and calculate the rms
    cube, mom0_rms = __load_cube__(cube_file, channels, center, size)

    # get the scale and the origin in the smaller region
    scale, new_origin = __get_scale_origin__(cube, center, size, origin)

    # get the moments of the cube
    mom0, mom1, mom2 = __get_mom__(cube, channels, mom0only=False, quick=quick)

    # define the ranges for the figure
    if in_sigma:
        v_ranges = [np.array(mom0_range) * mom0_rms, mom1_range, mom2_range]
    else:
        v_ranges = [mom0_range, mom1_range, mom2_range]
    v_ticks = [mom0_ticks, mom1_ticks, mom2_ticks]
    c_levels = [np.insert(3 * np.power(np.sqrt(2), np.arange(15)), 0, -3) * mom0_rms, 50, 50]

    # create the figure
    __fig_properties__()
    fig = plt.figure(1, (10.5, 3.6))
    plt.subplots_adjust(left=0.06, right=0.95, top=0.90, bottom=0.08)
    grid = ImageGrid(fig, 111, nrows_ncols=(1, 3), axes_pad=0.5, cbar_mode='each',
                     cbar_location='right', cbar_pad=0.0)
    iterable = zip(grid, grid.cbar_axes, [mom0, mom1, mom2], v_ranges, v_ticks, cmaps,
                   do_contours, c_levels)
    for ax, cax, qf_object, v_range, v_tick, cmap, do_contour, c_level in iterable:
        if do_contour:
            standardfig(raster=qf_object, contour=qf_object, ax=ax, fig=fig, origin=new_origin,
                        scale=scale, cmap=cmap, vrange=v_range, cbar=True, cbaraxis=cax,
                        vscale=v_tick, tickint=tick_value, clevels=c_level, flip=True)
        else:
            standardfig(raster=qf_object, ax=ax, fig=fig, origin=new_origin,
                        scale=scale, cmap=cmap, vrange=v_range, cbar=True, cbaraxis=cax,
                        vscale=v_tick, tickint=tick_value, flip=True)

    # Figure text
    fig.text(0.06, 0.88, r'Integrated [CII] flux (Jy km s$^{-1}$ bm$^{-1}$)',
             fontsize=12, color='black')
    fig.text(0.43, 0.88, r'Mean velocity (km s$^{-1}$)', fontsize=12, color='black')
    fig.text(0.70, 0.88, r'Velocity dispersion (km s$^{-1}$)', fontsize=12,
             color='black')

    fig.text(0.5, 0.03, r'$\Delta$ R.A. (arcsec)', fontsize=14, ha='center')
    fig.text(0.02, 0.5, r'$\Delta$ Decl. (arcsec)', fontsize=14, va='center',
             rotation=90)

    # save the figure
    if plot_fig is None:
        return fig
    else:
        __save_fig__(plot_fig)


def make_1panelfigure(plot=None, tick_value=1.0, vrange=(-3, 11), in_sigma=True, cb_ticks=0.2,
                      cont_in_ujy=False, do_contour=True, cmap='RdYlBu_r', ctext='', clevels=None,
                      plot_fig=None, cbar=True, fig_size=(4.6, 4), fig_adjust=[0.08, 0.99, 0.90, 0.12],
                      **kwargs):

    # define the data and text dictionary
    data = get_data(**kwargs)
    text = {'cont': r'Continuum flux density ($\mu$Jy beam$^{-1}$)',
            'mom0': r'Integrated [CII] flux (Jy km s$^{-1}$ beam$^{-1}$)',
            'mom1': r'Mean velocity (km s$^{-1}$)',
            'mom2': r'Velocity dispersion (km s$^{-1}$)',
            'cust': ctext}

    # define the plot properties
    if cont_in_ujy:
        data['cont'].data *= 1E6
        data['cont_rms'] *= 1E6
    if plot == 'cont':
        data_rms = data['cont_rms']
    elif plot == 'mom0':
        data_rms = data['mom0_rms']
    elif plot == 'cust':
        data_rms = data['cust_rms']
    else:
        data_rms = 1.
    if in_sigma:
        vrange = np.array(vrange) * data_rms
    if clevels is None and do_contour:  # standard contours start at 3sigma and increase in powers of sqrt(2)
        psqrt2 = np.power(np.sqrt(2), np.arange(15))
        clevels = np.insert(3 * psqrt2, 0, -3 * np.flip(psqrt2)) * data_rms

    # create the figure
    __fig_properties__()
    fig, ax = plt.subplots(1, 1, figsize=fig_size)
    plt.subplots_adjust(left=fig_adjust[0], right=fig_adjust[1], top=fig_adjust[2], bottom=fig_adjust[3])
    if do_contour:
        standardfig(raster=data[plot], contour=data[plot], ax=ax, fig=fig, origin=data['origin'],
                    scale=data['scale'], cmap=cmap, vrange=vrange, cbar=cbar,
                    vscale=cb_ticks, tickint=tick_value, clevels=clevels,
                    flip=True, **kwargs)
    else:
        standardfig(raster=data[plot], ax=ax, fig=fig, origin=data['origin'],
                    scale=data['scale'], cmap=cmap, vrange=vrange, cbar=cbar,
                    vscale=cb_ticks, tickint=tick_value, flip=True, **kwargs)

    # Figure text
    fig.text(0.5, 0.93, text[plot], fontsize=14, color='black', ha='center')
    fig.text(0.5, 0.03, r'$\Delta$ R.A. (arcsec)', fontsize=14,
             ha='center')
    fig.text(0.03, 0.5, r'$\Delta$ Decl. (arcsec)', fontsize=14,
             va='center', rotation=90)

    # save the figure
    if plot_fig is None:
        return fig
    else:
        __save_fig__(plot_fig)


def make_cmfigure(nrows=3, ncols=5, step=1, vrange=(-3, 11), in_sigma=True, cmap='RdYlBu_r',
                  cb_ticks=0.1, tick_value=1.0, figsize=(9, 6), cb_size=[0.10, 0.90, 0.86, 0.03],
                  gridsize=(0.08, 0.06, 0.90, 0.82), clevels=None, cube_rms=None, plot_fig=None,
                  cube_in_mjy=True, **kwargs):

    # get the data
    data = get_data(**kwargs)
    if cube_in_mjy:
        data['cube'].data *= 1E3
    
    # define the plot properties from the data
    if data['channels'] is None:
        channels = (0, data['cube'].data.shape[0])
    else:
        channels = data['channels']
    velocity_array = data['cube'].get_velocity()
    if cube_rms is None:
        cube_rms = np.median(data['cube'].calculate_sigma()[0:20])
    if in_sigma:
        vrange = np.array(vrange) * cube_rms
    if clevels is None:  # defaults to start at 2 sigma increase in powers of sqrt(2).
        psqrt2 = np.power(np.sqrt(2), np.arange(15))
        clevels = np.insert(2 * psqrt2, 0, -2 * np.flip(psqrt2)) * cube_rms

    # plot
    __fig_properties__()
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, gridsize, nrows_ncols=(nrows, ncols),
                     axes_pad=0.00, cbar_mode='none', share_all=True)
    for idx, channel in enumerate(np.arange(channels[0], channels[1], step=step)):

        # get the boolean of the beam (bottom left figure only)
        if (idx % ncols == 0) and (idx // ncols == int(nrows) - 1):
            beambool = True
        else:
            beambool = False

        # get the string value of the velocity
        velocity_string = str(int(round(velocity_array[channel]))) + ' km s$^{-1}$'

        cubeimage = data['cube'].get_slice(zindex=(channel, channel+1))
        standardfig(raster=cubeimage, contour=cubeimage, ax=grid[idx], fig=fig,
                    origin=data['origin'], scale=data['scale'], cmap=cmap, vrange=vrange,
                    cbar=False, vscale=cb_ticks, tickint=tick_value, clevels=clevels,
                    beam=beambool, flip=True, **kwargs)
        grid[idx].text(0.5, 0.85, velocity_string, transform=grid[idx].transAxes, fontsize=10,
                       color='black', bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2}, ha='center')

    # add colorbar
    img = plt.imshow(cubeimage.data, cmap=cmap, vmin=vrange[0], vmax=vrange[1])
    plt.gca().set_visible(False)
    cb_axes = fig.add_axes(cb_size)
    cb = plt.colorbar(img, cax=cb_axes, orientation='horizontal')
    cb.ax.tick_params(axis='x', direction='out', top=True, bottom=False,
                      labelbottom=False, labeltop=True)

    # add text
    fig.text(0.52, 0.97, 'Flux density (mJy beam$^{-1}$)', fontsize=13, ha='center')
    fig.text(0.52, 0.015, '$\\Delta$ R.A. (arcsec)', fontsize=13, ha='center')
    fig.text(0.015, 0.5, '$\\Delta$ Decl. (arcsec)', fontsize=13, va='center', rotation=90)

    # save the figure
    if plot_fig is None:
        return fig
    else:
        __save_fig__(plot_fig)


def make_spectrumfigure(cubefile, center, size, pbcor=None, in_mjy=True,
                        plot_fig=None, table_file=None):

    # get properties of the cube
    cube = Qube.from_fits(cubefile)
    cube_rms = cube.calculate_sigma()
    beam_area = (np.median(cube.beam['BMAJ']) *
                 np.median(cube.beam['BMIN']) /
                 (2 * np.log(2) * np.abs(cube.header['CDELT1'])**2))

    # get properties of the mask (circular only)
    ellipse = [center[0], center[1], size, size, 0]
    mask_area = np.pi * size**2
    nbeams = mask_area / beam_area

    # get the flux
    cube_mask = cube.mask_region(ellipse=ellipse)
    flux, vel = cube_mask.get_spec1d()
    rms = cube_rms * np.sqrt(nbeams)
    if pbcor:
        cube_pb = Qube.from_fits(pbcor)
        pb = (cube_pb.data[0, center[1], center[0]] /
              cube.data[0, center[1], center[0]])
        flux *= pb
        rms *= pb
    if in_mjy:
        flux *= 1E3
        rms *= 1E3

    # print to ascii file
    if table_file is not None:
        data = Table()
        data['vel(km/s)'] = vel
        if in_mjy:
            data['flux(mJy)'] = flux
            data['rms(mJy)'] = rms
        else:
            data['flux(Jy)'] = flux
            data['rms(Jy)'] = rms
        ascii.write(data, table_file, overwrite=True)

    # plot
    __fig_properties__()
    fig, ax = plt.subplots(1, 1, figsize=(4.8, 4))
    plt.subplots_adjust(left=0.12, right=0.99, top=0.98, bottom=0.12)
    plt.plot(vel, flux, '-o')
    plt.fill_between(vel, rms, -1 * rms, alpha=0.1, color='black')
    plt.axhline(0, ls='--', color='black')

    # Figure text
    fig.text(0.53, 0.03, r'Velocity (km s$^{-1}$)', fontsize=14,
             ha='center')
    if in_mjy:
        fig.text(0.03, 0.5, r'Flux density (mJy)', fontsize=14,
                 va='center', rotation=90)
    else:
        fig.text(0.03, 0.5, r'Flux density (Jy)', fontsize=14,
                 va='center', rotation=90)

    # save the figure
    if plot_fig is None:
        return fig
    else:
        __save_fig__(plot_fig)
    


def get_data(cont=None, cube=None, moments=None, cust=None, channels=None, center=None, size=None,
             origin=None, mask_rms=3, quick=False, cont_rms=None, mom0_rms=None, cust_rms=None,
             mom0_only=False):

    # initialize the data structure
    data = {'center':center, 'size':size, 'origin':origin, 'channels':channels}

    # load the data
    if cont is not None:
        __load_cont__(data, cont, cont_rms=cont_rms)
    if cube is not None:
        __load_cube__(data, cube, mom0_rms=mom0_rms)
    else:
        if moments is not None:
            if (moments[0] == True or moments[1] == True or moments[2] == True):
                __load_cube__(data, cube, mom0_rms=mom0_rms)
    if moments is not None:
        __load_moments__(data, moments, mask_rms=mask_rms, mom0_only=mom0_only, quick=quick,
                         mom0_rms=mom0_rms)
    if cust is not None:
        data['cust'] = Qube.from_fits(cust)
        if data['size'] is not None:
            data['cust'] = __get_slice__(data['cust'], data['center'], data['size'])
    data['cust_rms'] = cust_rms

    # update the data parameters
    __update_datapars__(data)

    return data

    
def __load_cube__(data, cube_file, mom0_rms=None):
    data['cube'] = Qube.from_fits(cube_file)
    if mom0_rms is None:
        temp_mom0 = data['cube'].calculate_moment(moment=0, channels=data['channels'])
        data['mom0_rms'] = temp_mom0.calculate_sigma()
    if data['size'] is not None:
        data['cube'] = __get_slice__(data['cube'], data['center'], data['size'])


def __load_cont__(data, cont_file, cont_rms=None):
    data['cont'] = Qube.from_fits(cont_file)
    if cont_rms is None:
        try:
            data['cont_rms'] = data['cont'].header['rms']
        except KeyError:
            data['cont_rms'] = data['cont'].calculate_sigma()
    else:
        data['cont_rms'] = cont_rms
    if data['size'] is not None:
        data['cont'] = __get_slice__(data['cont'], data['center'], data['size'])


def __load_moments__(data, moments, mask_rms=3, mom0_only=False, quick=False, mom0_rms=None):

    # moment-0 --always load when calculating, as it is needed for the other moments
    if moments[0] == True or moments[1] == True or moments[2] == True:
        data['mom0'] = data['cube'].calculate_moment(moment=0, channels=data['channels'])
    if type(moments[0]) == str:
        data['mom0'] = Qube.from_fits(moments[0])
   
    # moment-0 rms any of the below two will overwrite the calculated value (if present).
    if 'mom0' in data.keys():
        if 'rms' in data['mom0'].header:  # this supersedes any previously calculates mom0_rms
            data['mom0_rms'] = data['mom0'].header['rms']
        if mom0_rms is not None:
            data['mom0_rms'] = mom0_rms
    
    # calculate moments 1 and 2 (if needed)
    if moments[1] == True or moments[2] == True:
        mask = data['mom0'].mask_region(value=data['mom0_rms'] * mask_rms, applymask=False)
        cube_m = data['cube'].mask_region(value=0.0)
        mom1 = data['cube'].calculate_moment(moment=1, channels=data['channels'])
        mom2 = cube_m.calculate_moment(moment=2, channels=data['channels'])
        if not quick:
            mom1, mom2 = data['cube'].gaussian_moment(mom1=mom1, mom2=mom2)
        mom1 = mom1.mask_region(mask=mask)
        mom2 = mom2.mask_region(mask=mask)

    # moment-1
    if type(moments[1]) == bool:
        if moments[1]:
            data['mom1'] = mom1
    elif type(moments[1]) == str:
        data['mom1'] = Qube.from_fits(moments[1])
    else:
        raise ValueError('moments[1] needs to be a valid string file or True/False')

    # moment-2
    if type(moments[2]) == bool:
        if moments[2]:
            data['mom2'] = mom2
    elif type(moments[2]) == str:
        data['mom2'] = Qube.from_fits(moments[2])
    else:
        raise ValueError('moments[2] needs to be a valid string file or True/False')


def __get_slice__(qf_object, center, size):
    if center is None:
        center = (qf_object.data.shape[2] // 2, qf_object.data.shape[1] // 2)
    x_index = (center[0] - size, center[0] + size + 1)
    y_index = (center[1] - size, center[1] + size + 1)
    qf_object_small = qf_object.get_slice(xindex=x_index, yindex=y_index)
    return qf_object_small


def __update_datapars__(data):

    lst = ['cont', 'cube', 'mom0', 'mom1', 'mom2', 'cust']
    for key in lst:
        try:
            qf_object = data[key]
            data['scale'] = np.abs(qf_object.header['CDELT1'] * 3600)
            if data['center'] is None:
                data['center'] = (qf_object.header['NAXIS1'] // 2, qf_object.header['NAXIS2'] // 2)
            if data['origin'] is None:
                data['origin'] = (qf_object.header['NAXIS1'] // 2, qf_object.header['NAXIS2'] // 2)
            if data['size'] is not None:
                data['origin'] = (data['origin'][0] - data['center'][0] + data['size'],
                                  data['origin'][1] - data['center'][1] + data['size'])
        except KeyError:
            pass


def __fig_properties__():

    mpl.rcdefaults()
    font = {'family': 'DejaVu Sans', 'weight': 'normal',
            'size': 10}
    mpl.rc('font', **font)
    mpl.rc('mathtext', fontset='stixsans')
    mpl.rc('axes', lw=1)
    mpl.rc('xtick.major', size=4, width=1)
    mpl.rc('ytick.major', size=4, width=1)
    mpl.rcParams['xtick.direction'] = 'in'
    mpl.rcParams['ytick.direction'] = 'in'


def __save_fig__(plot_fig):
    if type(plot_fig) == str:
        plt.savefig(plot_fig, format='pdf', dpi=300)
        plt.close('all')
    elif type(plot_fig) == bool:
        plt.show()
