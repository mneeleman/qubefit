.. _models:

Pre-Defined Models
===============================================
Here is a list and description of the current models that have been tested, and are supplied with the code. If you wish to create your own custom model, please see the page on :ref:`newmodel`.

.. note::
   To create the models, we need to switch between the frame of the galaxy (used to create the models) and the frame of the sky (the observed frame). To distinguish between these two frames, here and elsewhere, we define the unprimed frame (e.g., r, φ and z) as the frame of the galaxy and the primed frame (e.g., r', φ', z') as the frame of the sky. That is in the cylindrical coordinate system, r and φ are within the plane of the disk, and z is perpendicular to the disk, whereas the primed coordinates r' and φ' are within the plane of the sky and z' is pointing towards us.

Thin Disk Model
-------------------------------------
This model describes the kinematic signature of emission lines that stem from an infinitely thin rotating disk, where all of velocities arise solely from rotation. The thin disk model is described in detail in Neeleman et al. 2021 ``link this``. Here we provide a more general approach to the discussion in this paper. First, to create a thin disk model requires a mathematical description of the spatial distribution of the intensity (*I*), rotational velocity (*v*:subscript:`rot`) and line-of-sight velocity dispersion (*σ*:subscript:`v`). In its most general form, each of these quantities could be varying as a function of radius (r), and azimuthal angle (φ), since the emission is constrained within the plane z=0. Therefore we have the following equations:

.. math::

   I &= I(r, \phi)\\
   v_{\rm rot} &=  v_{\rm rot}(r, \phi)\\
   \sigma_v &=  \sigma_v(r, \phi)\\

Here the functional forms that can be chosen for the profiles are defined in the :ref:`qfmodels` function. There is a list of basic functions available, such as a constant, exponential, linear or Sérsic profile, but it is straightforward to add additional profiles. The code assumes that the profiles are separable; meaning that I(r, φ) = I(r) x I(φ), and in almost all cases you probably only care for profiles that only vary with radius (i.e., I(φ) is constant). However it is relatively straightforward to relax this assumption in a new custom model (see :ref:`newmodel`).

Most profiles are determined by a single scale factor (either *R*:subscript:`d` or  *R*:subscript:`v` for radial profiles), except for the power and Sérsic profiles, which require an additional parameter. The profiles are scaled using a scaling factor, which is either *I*:subscript:`0`, *V*:subscript:`max`, or *σ*:subscript:`v` depending on which profile.

These profiles need to be transformed into the frame of the sky (the primed frame). In theory, we could do this through matrix rotations. However in practice this produces incorrect results, because the disk is infinitely thin and most of the emission will fall between the grid after rotation, and correctly distributing the flux onto the regular grid is difficult. Fortunately, the infinitely thin disk has easy-to-calculate transformations. Following the work in `Chen et al. 2005 <https://arxiv.org/abs/astro-ph/0411006>`_:

.. math:: 

   r &= r' \times \sqrt{1 + \sin^2(\phi' - \alpha) \tan^2(i)}.\\
   v_{0, z'} &= \frac{\cos(\phi'-\alpha)\sin(i)}{\sqrt{1 + \sin^2(\phi' - \alpha)\tan^2(i)}}v_{\rm rot} + v_{\rm c}(z_{\rm kin})

In the last equation, *v*:subscript:`0,z'` is the velocity along the direction perpendicular to the sky (i.e., the spectral velocity direction). The parameter *v*:subscript:`c` is the zero-point offset, which is determined by the redshift of the emission line, *α* is the position angle of the major axis, and *i* is the inclination of the disk. Assuming azimuthally constant profiles (i.e., profiles solely dependent on r), and assuming the velocity dispersion obey a 1D Gaussian distribution, yields:

.. math::

   I(r', \phi', v') = I(r') e^{(v' - v_{0, z'})^2 / 2\sigma_v^2}.

Here are r' and *v*:subscript:`0, z` are defined by the above equations, and *σ*:subscript:`v` is the one-dimensional velocity distribution. The above model cube has been created out of ten different parameters. Here is a list of the different parameters with a short description:

  * **Xcen**: The position of the center of the disk in the x-direction.
  * **Ycen**: The position of the center of the disk in the y-direction.
  * **PA**: The position angle of the major axis (equal to *α* in the above equations).
  * **Incl**: The inclination of the disk (equal to *i* in the above equations).
  * **I0**: The scale factor of the intensity. For most profiles this gives the maximum intensity of the emission at the center of the disk. This intensity is really a scaling of the whole cube, and providing an actual physical meaning is very difficult for any but the most simple models.
  * **Rd**: The scale distance of the intensity profile, both **I0** and **Rd** define the intensity profile.
  * **Vmax**: The (maximum) rotational velocity of the disk.
  * **Vcen**: The systemic velocity of the galaxy.
  * **Rv**: The scale distance of the velocity profile (and dispersion profile). The **Rv** and **Vmax** parameter define the velocity profile.
  * **Disp**: The 1D (maximum) total velocity dispersion. Together with **Rv** this determines the velocity dispersion profile.

Besides these ten main parameters, there are an additional three parameters, **IIdx**, **VIdx** and **DIdx** that might be needed if you wish to specify the intensity, velocity or dispersion profiles with one of the parametric functions.
    
.. note::

   **Note on velocity dispersions.** Having an infinitely thin disk with only azimuthal velocities (i.e., no radial or out-of-the-disk velocities) by its very nature implies a distribution that has zero velocity dispersion. This is because velocity dispersion implies motion perpendicular to the disk or along the radial direction, and addition of a velocity dispersion is therefore unphysical in an infinitely thin disk. However, physical disks have some thickness, and one can interpret the added velocity dispersion as the **maximum** amount of velocity dispersion (or total velocity dispersion) that is needed to make the model agree with the data. In practice, for a physical disk the total velocity dispersion, *σ*:subscript:`v`,  is composed of bulk radial motions, bulk motions perpendicular to the disk, as well as the velocity dispersion of the individual gas clouds that emit the emission lines. It, however, does not include the dispersion due to the line spread function of the instrument as well as beam smearing. These are factored out during the fitting procedure.

Dispersion Bulge Model
-------------------------------------
This model describes the emission line signature that arises from gravitationally bound gas that does not show bulk rotation. An example of such motion is that from stars within a classical bulge or elliptical galaxy. In such systems, the random orientation of the rotation manifest themselves as a gaussian velocity distribution around the systemic velocity of the galaxy. The dispersion-dominated bulge model is described in detail in Neeleman et al.  (2021) ``link this``. Here we summarize the discussion in this paper.

To describe the bulge model requires a description of the intensity profile (*I*) of the emission and the profile of the velocity dispersion (*σ*:subscript:`v`). In this simple model, we assume that the bulge is spherically symmetric, and therefore we can set the galaxy frame to the sky frame. In this model, we also wish to describe the intensity profile by a 2D function (and not the intrinsic 3D density distribution). We therefore have:

.. math::

   I &= I(r, \phi)\\
   \sigma_v &=  \sigma_v(r, \phi)\\

For these two profiles several shapes can be chosen. The list of available profiles are given in :ref:`qfmodels`, but it is straightforward to add your own profile to this list. Most profiles are normalized to some specific value (where possible, unity), and have a single scale distance. However, some of the profiles are parametric (i.e., the Sérsic and power profiles) and require an additional parameter, IIdx or DIdx. The profiles are normalized by the scaling factor in intensity, *I0* and velocity dispersion *Disp*.

Assuming that the velocity dispersion has a Gaussian velocity distribution around the systemic velocity of the cube yields:

.. math::

   I(r', v') = I_0 e^{-r'/R_{\rm D}} e^{(v' - v_{\rm c})^2 / 2\sigma_v^2}.

Here the primed and unprimed coordinate frame are equal. For this model, we require a total of 7 parameters:

  * **Xcen**: The position of the center of the bulge in the x-direction.
  * **Ycen**: The position of the center of the bulge in the y-direction.
  * **I0**: The scale factor of the intensity. For most profiles this gives the maximum intensity of the emission at the center of the disk. This intensity is really a scaling of the whole cube, and providing an actual physical meaning is very difficult for any but the most simple models.
  * **Rd**: The scale distance of the intensity profile, both **I0** and **Rd** define the intensity profile.
  * **Vcen**: The systemic velocity of the galaxy.
  * **Disp**: The 1D (maximum) total velocity dispersion.
  * **Rv**: The scale distance of the velocity dispersion profile. The **Rv** and **Disp** parameter together set the velocity dispersion profile.

In addition, the parameters, **IIdx** and **DIdx** are needed if a parametric profile, such as the Sérsic profile is used. 


Two Clumps Model
----------------------
This model is a combination of two bulge models. It can be used to test if the velocity gradient in marginally resolved observations can be reproduced using simple non-rotating clumps that are moving w.r.t. each other. This model is a simple linear combination of the bulge model described above, and shows how other models can be built from the simple model above.

The model requires 2 x 7 = 14 parameters to define the model. As described in the bulge model, a possible 4 additional parameters need to be defined depending on the chosen intensity or velocity dispersion profile. Because two intensity profiles are required, the :ref:`qubefit` instance must contain a list of profiles such as:

   qube.intensityprofile = [['Exponential', None, 'Step'], ['Sersic', None, None]]
   qube.dispersionprofile = [['Constant', None, None], ['Constant', None, None]]

The 14 parameters that need to be defined are:

  * **Xcen1, Xcen2**: The position of the center of the first (second) clump in the x-direction.
  * **Ycen1, Ycen2**: The position of the center of the first (second) clump in the y-direction.
  * **I01, I02**: The scale factor of the intensity of the first (second) clump.
  * **Rd1, Rd2**: The scale distance of the intensity profile of the first (second) clump, both **I0** and **Rd** define the intensity profile.
  * **Vcen1, Vcen2**: The systemic velocity of the first (second) clump
  * **Disp1, Disp2**: The 1D (maximum) total velocity dispersion of the first (second) clump.
  * **Rv1, Rv2**: The scale distance of the velocity dispersion profile of the first (second) clump. The **Rv** and **Disp** parameter together set the velocity dispersion profile.

In addition, the parameters, **IIdx1, IIdx2, DIdx1** and **DIdx2** are needed if a parametric profile, such as the Sérsic profile is used. 


Thin Spiral Model
-----------------------------------
This model builds on the thin disk model by adding in a spiral density component. This model was used in `Chittidi et al. (2021) <https://arxiv.org/abs/2005.13158>`_. In this model the spiral pattern does not affect the velocity of the gas, it is just modeled by a modification of the intensity profile compared to the thin disk:

.. math::

   I(r, \phi) = I_1(r, \phi) + I_2(r, \phi) = I_0 [I_1(r, \phi) + I_{spf} I_2(r, \phi)]

Here *I*:subscript:`1` is the density profile of the disk as in the Thin Disk model, and *I*:subscript:`2` is the density profile of the spiral pattern. In the second equality we explicitly show how the spiral pattern is scaled with the scale factor *I*:subscript:`spf`. To model the spiral pattern we take the very simple approach that the spiral pattern has a Gaussian distribution as a function of azimuthal angle at a given radius, and the wrapping pattern is proportional to the radius:

.. math::

   I_2(r, \phi) = \left\{
   \begin{array}{ll}
      \sum^{n_{sp}}_i e^{-\frac{(\phi - \phi_c)^2}{2\sigma_\phi^2}} & \text{if}~~r < R_s,~~ \text{where}~~ \phi_c = \phi_0 + \frac{2\pi i}{n_{sp}} + \alpha_{sp} r\\
      0 &\text{if}~~ r > R_s\\
   \end{array} 
   \right.

Here the spiral pattern is defined only up to an outer radius, *R*:subscript:`s`. The sum is over the number of spiral arms, *n*:subscript:`sp` where the width of the spiral arms is given by the parameter *σ*:subscript:`φ`. To calculate the central position of the *i*:superscript:`th` spiral arm, we need three parameters, the initial position of the first spiral arm, *φ*:subscript:`0`, the number of spiral arms, *n*:subscript:`sp` and the wrapping frequency, *α*:subscript:`sp`. 

The thin spiral model has besides the ten plus three parameters of the thin disk model an additional six parameters describing the spiral pattern. The list of all parameters of the model is:

  * **Xcen**: The position of the center of the disk in the x-direction.
  * **Ycen**: The position of the center of the disk in the y-direction.
  * **PA**: The position angle of the major axis (equal to *α* in the thin disk model).
  * **Incl**: The inclination of the disk (equal to *i* in the thin disk model).
  * **I0**: The scale factor of the intensity.
  * **Rd**: The scale distance of the intensity profile.
  * **Vmax**: The (maximum) rotational velocity of the galaxy.
  * **Vcen**: The systemic velocity of the galaxy.
  * **Rv**: The scale distance of the velocity profile and dispersion profile.
  * **Disp**: The 1D (maximum) total velocity dispersion.
  * **IIdx**: (optional) Intensity index for parameteric profiles.
  * **VIdx**: (optional) Velocity index for parameteric profiles.
  * **DIdx**: (optional) Dispersion index for parameteric profiles.
  * **Nspiral**: Number of spiral arms.
  * **Phi0**: The azimuthal angle of the first spiral arm.
  * **Spcoef**: The wrapping frequency of the spiral arm (*α*:subscript:`sp` in the above equations).
  * **Dphi**: Gaussian width of the spiral arm (*σ*:subscript:`φ` in the above equations).
  * **Ispf**: The fractional contribution of the spiral pattern. Depending on the profile, a value of unity would imply that the spiral density has a maximum contribution similar to the disk component.
  * **Rs**: The cut-off radius of the spiral arm. This is a sharp cutoff, but this can be changed in the model by taking a different density profile (not a step function).

