from qubefit.qube import Qube
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from qubefit.qfutils import standardfig
from astropy.io import ascii
from astropy.table import Table


def make_4panelfigure(cube_file, cont_file, channels=None, center=None,
                      size=50, origin=None, mask_rms=3, quick=False,
                      tick_value=1.0, cont_range=(-3, 11), mom0_range=(-3, 11),
                      mom1_range=(-200, 200), mom2_range=(0, 250), in_sigma=True,
                      cont_ticks=50, mom0_ticks=0.2, mom1_ticks=50, mom2_ticks=50,
                      cont_in_ujy=True, do_contours=(True, True, False, False),
                      cmaps=('RdYlBu_r', 'RdYlBu_r', 'Spectral_r', 'Spectral_r'),
                      fig_file=None):

    # load the cube and cont, trim, and calculate the rms
    cube, mom0_rms = __load_cube__(cube_file, channels, center, size)
    cont, cont_rms = __load_cont__(cont_file, cont_in_ujy, center, size)

    # get the scale and the origin in the smaller region
    scale, new_origin = __get_scale_origin__(cube, center, size, origin)

    # get the moments of the cube
    mom0 = cube.calculate_moment(moment=0, channels=channels)
    mask = mom0.mask_region(value=mom0_rms * mask_rms, applymask=False)
    cube_m = cube.mask_region(value=0.0)
    mom1 = cube.calculate_moment(moment=1, channels=channels)
    mom2 = cube_m.calculate_moment(moment=2, channels=channels)
    if not quick:
        mom1, mom2 = cube.gaussian_moment(mom1=mom1, mom2=mom2)
    mom1 = mom1.mask_region(mask=mask)
    mom2 = mom2.mask_region(mask=mask)

    # define the ranges for the figure
    if in_sigma:
        v_ranges = [np.array(cont_range) * cont_rms, np.array(mom0_range) * mom0_rms,
                    mom1_range, mom2_range]
    else:
        v_ranges = [cont_range, mom0_range, mom1_range, mom2_range]
    v_ticks = [cont_ticks, mom0_ticks, mom1_ticks, mom2_ticks]
    c_levels = [np.insert(3 * np.power(np.sqrt(2), np.arange(15)), 0, -3) * cont_rms,
                np.insert(3 * np.power(np.sqrt(2), np.arange(15)), 0, -3) * mom0_rms,
                50, 50]

    # create the figure
    __fig_properties__()
    fig = plt.figure(1, (8., 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(2, 2), axes_pad=0.5, cbar_mode='each',
                     cbar_location='right', cbar_pad=0.0)
    iterable = zip(grid, grid.cbar_axes, [cont, mom0, mom1, mom2], v_ranges, v_ticks, cmaps, do_contours, c_levels)
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
    __save_fig__(fig_file)


def make_3panelfigure(cube_file, channels=None, center=None, size=50, origin=None,
                      mask_rms=3, quick=False, tick_value=1.0, mom0_range=(-3, 11),
                      mom1_range=(-200, 200), mom2_range=(0, 250), in_sigma=True,
                      mom0_ticks=0.2, mom1_ticks=50, mom2_ticks=50, do_contours=(True, False, False),
                      cmaps=('RdYlBu_r', 'Spectral_r', 'Spectral_r'), fig_file=None):

    # load the cube and cont, trim, and calculate the rms
    cube, mom0_rms = __load_cube__(cube_file, channels, center, size)

    # get the scale and the origin in the smaller region
    scale, new_origin = __get_scale_origin__(cube, center, size, origin)

    # get the moments of the cube
    mom0 = cube.calculate_moment(moment=0, channels=channels)
    mask = mom0.mask_region(value=mom0_rms * mask_rms, applymask=False)
    cube_m = cube.mask_region(value=0.0)
    mom1 = cube.calculate_moment(moment=1, channels=channels)
    mom2 = cube_m.calculate_moment(moment=2, channels=channels)
    if not quick:
        mom1, mom2 = cube.gaussian_moment(mom1=mom1, mom2=mom2)
    mom1 = mom1.mask_region(mask=mask)
    mom2 = mom2.mask_region(mask=mask)

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
    __save_fig__(fig_file)


def make_1panelfigure(datafile, plot, channels=None, center=None,
                      size=50, origin=None, mask_rms=3, quick=False,
                      tick_value=1.0, vrange=(-3, 11), in_sigma=True,
                      ticks=0.2, cont_in_ujy=True, do_contour=True,
                      cmap='RdYlBu_r', fig_file=None, **kwargs):

    # load the cube and cont, trim, and calculate the rms
    if plot == 'cont':
        data, data_rms = __load_cont__(datafile, cont_in_ujy, center, size)
        text = r'Continuum flux density ($\mathrm{\mu}$Jy beam$^{-1}$)'
    else:
        data, data_rms = __load_cube__(datafile, channels, center, size)

    # get the scale and the origin in the smaller region
    scale, new_origin = __get_scale_origin__(data, center, size, origin)

    # get the moments of the cube and apply mask
    if plot != 'cont':
        mom0 = data.calculate_moment(moment=0, channels=channels)
    if plot == 'mom1' or plot == 'mom2':
        mask = mom0.mask_region(value=data_rms * mask_rms, applymask=False)
        data_m = data.mask_region(value=0.0)
        mom1 = data.calculate_moment(moment=1, channels=channels)
        mom2 = data_m.calculate_moment(moment=2, channels=channels)
        if not quick:
            mom1, mom2 = data.gaussian_moment(mom1=mom1, mom2=mom2)
        mom1 = mom1.mask_region(mask=mask)
        mom2 = mom2.mask_region(mask=mask)
    if plot == 'mom0':
        data = mom0
        text = r'Integrated [CII] flux (Jy km s$^{-1}$ beam$^{-1}$)'
    elif plot == 'mom1':
        data = mom1
        text = r'Mean velocity (km s$^{-1}$)'
    elif plot == 'mom2':
        data = mom2
        text =  r'Velocity dispersion (km s$^{-1}$)'

    # define the range and contour levels for the figure
    if in_sigma:
        vrange = np.array(vrange) * data_rms
    if do_contour:
        clevels = np.insert(3 * np.power(np.sqrt(2), np.arange(15)), 0, -3) * data_rms

    # create the figure
    __fig_properties__()
    fig, ax = plt.subplots(1, 1, figsize=(4.6, 4))
    plt.subplots_adjust(left=0.08, right=0.99, top=0.90, bottom=0.12)
    if do_contour:
        standardfig(raster=data, contour=data, ax=ax, fig=fig, origin=new_origin,
                    scale=scale, cmap=cmap, vrange=vrange, cbar=True,
                    vscale=ticks, tickint=tick_value, clevels=clevels,
                    flip=True, **kwargs)
    else:
        standardfig(raster=data, ax=ax, fig=fig, origin=new_origin,
                    scale=scale, cmap=cmap, vrange=vrange, cbar=True,
                    vscale=ticks, tickint=tick_value, flip=True, **kwargs)

    # Figure text
    fig.text(0.5, 0.93, text, fontsize=14, color='black', ha='center')
    fig.text(0.5, 0.03, r'$\Delta$ R.A. (arcsec)', fontsize=14,
             ha='center')
    fig.text(0.03, 0.5, r'$\Delta$ Decl. (arcsec)', fontsize=14,
             va='center', rotation=90)

    # save the figure
    __save_fig__(fig_file)


def make_cmfigure(cube_file, center=None, channels=None, size=50,
                  origin=None, nrows=3, ncols=5, step=1, vrange=(-3, 11),
                  in_sigma=True, fig_file=None, ticks=0.1, average=False,
                  tick_value=1.0, figsize=(10, 6), **kwargs):

    # load the cube and parameters:
    cube, _mom0_rms = __load_cube__(cube_file, channels, center, size)
    if average is True:
        cube.data = (cube.data + np.roll(cube.data, -1, 0)) / 2.
    cube.data *= 1E3
    cube_rms = np.median(cube.calculate_sigma()[0:20])
    scale, new_origin = __get_scale_origin__(cube, center, size, origin)
    if channels == 'None':
        channels = (0, cube.data.shape[0])
    velocity_array = cube.get_velocity()
    if in_sigma:
        vrange = np.array(vrange) * cube_rms
    
    __fig_properties__()
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, (0.08, 0.06, 0.90, 0.82), nrows_ncols=(nrows, ncols),
                     axes_pad=0.00, cbar_mode='none', share_all=True)
    for idx, channel in enumerate(np.arange(channels[0], channels[1], step=step)):

        # get the boolean of the beam (bottom left figure only)
        if (idx % ncols == 0) and (idx // ncols == int(nrows) - 1):
            beambool = True
        else:
            beambool = False

        # get the string value of the velocity
        velocity_string = str(int(round(velocity_array[channel] + 10))) + ' km s$^{-1}$'

        cubeimage = cube.get_slice(zindex=(channel, channel+1))
        clevels = np.insert(2 * np.power(np.sqrt(2), np.arange(0, 5)), 0,
                            [-4, -2.82, -2]) * cube_rms
        standardfig(raster=cubeimage, contour=cubeimage, ax=grid[idx], fig=fig,
                    origin=new_origin, scale=scale, cmap='RdYlBu_r', vrange=vrange,
                    cbar=False, vscale=ticks, tickint=tick_value, clevels=clevels,
                    beam=beambool, flip=True, **kwargs)
        grid[idx].text(0.5, 0.85, velocity_string, transform=grid[idx].transAxes, fontsize=10, color='black',
                       bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 2}, ha='center')

    # add colorbar
    img = plt.imshow(cubeimage.data, cmap='RdYlBu_r', vmin=vrange[0], vmax=vrange[1])
    plt.gca().set_visible(False)
    cbaxes = fig.add_axes([0.10, 0.90, 0.86, 0.03])
    cb = plt.colorbar(img, cax=cbaxes, orientation='horizontal')
    cb.ax.tick_params(axis='x', direction='out', top=True, bottom=False,
                      labelbottom=False, labeltop=True)
    fig.text(0.52, 0.97, 'Flux density (mJy beam$^{-1}$)', fontsize=13, ha='center')

    # add text
    fig.text(0.52, 0.015, '$\\Delta$ R.A. (arcsec)', fontsize=13, ha='center')
    fig.text(0.015, 0.5, '$\\Delta$ Decl. (arcsec)', fontsize=13, va='center', rotation=90)

    # save the figure
    __save_fig__(fig_file)


def make_spectrumfigure(cubefile, center, size, pbcor=None, in_mjy=True,
                        fig_file=None, table_file=None):

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
    __save_fig__(fig_file)


def __load_cube__(cube_file, channels, center, size):
    cube = Qube.from_fits(cube_file)
    temp_mom0 = cube.calculate_moment(moment=0, channels=channels)
    mom0_rms = temp_mom0.calculate_sigma()
    cube_small = __get_slice__(cube, center, size)
    return cube_small, mom0_rms


def __load_cont__(cont_file, cont_in_ujy, center, size):
    cont = Qube.from_fits(cont_file)
    if cont_in_ujy:
        cont.data *= 1E6
    cont_rms = cont.calculate_sigma()
    cont_small = __get_slice__(cont, center, size)
    return cont_small, cont_rms


def __get_slice__(qf_object, center, size):
    if center is None:
        center = (qf_object.data.shape[2] // 2, qf_object.data.shape[1] // 2)
    x_index = (center[0] - size, center[0] + size + 1)
    y_index = (center[1] - size, center[1] + size + 1)
    qf_object_small = qf_object.get_slice(xindex=x_index, yindex=y_index)
    return qf_object_small


def __get_scale_origin__(qf_object, center, size, origin):
    # get the scale and the origin in the smaller region
    scale = np.abs(qf_object.header['CDELT1']) * 3600
    if origin is None:
        origin = center
    new_origin = (origin[0] - center[0] + size, origin[1] - center[1] + size)
    return scale, new_origin


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


def __save_fig__(fig_file):
    if fig_file is not None:
        plt.savefig(fig_file, format='pdf', dpi=300)
    else:
        plt.show()
    plt.close('all')
