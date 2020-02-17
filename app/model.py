"""Functions for the sound field computation"""

import math

import numpy as np
import pyproj as proj
from scipy.optimize import least_squares
from scipy.special import erf
from shapely.geometry import asPoint
from shapely.geometry.polygon import Polygon


def distance_between(s, r):
    """Distance of all combinations of points in s and r.
    Parameters
    ----------
    s : ndarray, (3, ns)
    r : ndarray, (3, nr)
    Returns
    -------
    ndarray, (nr, ns)
        Distances between points
    """
    return np.linalg.norm(s[:, None, :] - r[:, :, None], axis=0)


def impedance_miki_model(f, sigma):
    """
    impedance_miki_model(f, sigma)
    Caculate the surface impedance according to Miki Model.
    Parameters
    ----------
    f : 1-D array of frequencies in [Hz].
    sigma : double flow resistivity in [Ns/m^4].
    Returns
    -------
    Zs : 1-D array of normalized ground impedance, Zs/rhoc.
    """
    Zs = (
        1
        + 5.51 * (1000 * f / sigma) ** (-0.632)
        - 8.42j * (1000 * f / sigma) ** (-0.632)
    )

    return Zs


def speed_of_sound(T):
    """
    speed_of_sound(T)
    Caculate the speed of sound according to the temperature.
    Parameters
    ----------
    T : double value of temperature in [C].
    Returns
    -------
    c : double value of speed of sound in [m/s].
    """
    c = 20.05 * np.sqrt(273.15 + T)

    return c


def geometrical_parameters(s, r):
    """
    geometrical_parameters(s, r)
    Source-receiver over ground.Calculates the distances between sources and
    receivers and the angles of incidence to the ground. It handles 2D or 3D.
    The last coordinate is always the height.
    Parameters
    ----------
    s : 2-D array [Coordinates, Number sources] of source coordinates in [m].
    r : 2-D array [Coordinates, Number receivers] of receiver coordinates
    in [m].
    Returns
    -------
    r1 : 2-D array [Number receivers, Number sources] of distances
        in the direct path in [m].
    r2 : 2-D array [Number receivers, Number sources] of distances
        in the reflected path in [m].
    theta : 2-D array [Number receivers, Number sources] of angles of
        incidence to the ground in [rad].
    """
    D = r.shape[0]  # Dimensions
    si = np.copy(s)
    si[-1, :] = -si[-1, :]  # Mirror z-axis. Image sources with ground
    uv = np.zeros(D)
    uv[-1] = 1  # Unitary vector z-axis

    r1 = distance_between(s, r)  # Direct path distance source-receivers
    r2 = distance_between(si, r)  # Reflected path distance
    dr_xy = distance_between(s[0:-1], r[0:-1])  # distance projected on ground
    theta = np.pi / 2 - np.arccos(dr_xy / r2)  # Angle of incidence

    return r1, r2, theta


def spherical_reflection_coefficient(k, r2, Zs, theta):
    """Calculate the spherical reflection coefficient for the sources and
    receivers as inputs.
    Parameters
    ----------
    Nf : int number of frequencies.
    Ns : int number of sources.
    Nr : int number of receivers.
    r2 : [Nr, Ns] distance between sources and receivers in the indirect path
        in [m].
    Zs : 1-D array [Nf] of normalized ground impedance, Zs/rhoc.
    theta : 2-D array [Nr, Ns] of angles of incidence to the ground in [rad].
    Returns
    -------
    Q : 3-D array [Nf,Nr,Ns] with spherical reflection coefficient.
    """
    assert len(k) == len(Zs)
    assert theta.shape == r2.shape

    w = (
        0.5
        * (1 - 1j)
        * np.sqrt(k[:, None, None] * r2)
        * (np.cos(theta) + 1 / Zs[:, None, None])
    )
    F = 1 - 1j * np.sqrt(np.pi) * w * np.exp(-w ** 2) * (1 - erf(1j * w))
    Zscos = Zs[:, None, None] * np.cos(theta)
    Rp = (Zscos - 1) / (Zscos + 1)
    Q = Rp + (1 - Rp) * F

    return Q


def calc_sound_field(sigma, T, f, A1, A2, s, r, q):
    """
    calc_sound_field(sigma, T, f, A1, A2, s, r, q)
    Calculate the sound field at the receiver positions.
    Parameters
    ----------
    sigma : double flow resistivity in [Ns/m^4].
    T : double value of temperature in [C].
    f : 1-D array of frequencies in [Hz].
    A1 : 3-D array [Nf,Nr,Ns] of complex directivity, direct path.
    A2 : 3-D array [Nf,Nr,Ns] of complex directivity, reflected path.
    s : 2-D array [Coordinates, Ns] of source coordinates in [m].
    r : 2-D array [Coordinates, Nr] of receiver coordinates
    in [m].
    q : 2-D array [Nf,Ns] of sources strength.
    Returns
    -------
    p_free : 3-D array [Nf,Nr,Ns] with free field pressure.
    p : 3-D array [Nf,Nr,Ns] with the contribution of each source to the total
        field (direct + reflected).
    Dp : 3-D array [Nf,Nr,Ns] with excess attenuation.
    Dr : 2-D array [Nr, Ns] of distances between s and r in [m].
    """
    Nf = len(f)
    Ns = s.shape[1]  # Number of sources
    Nr = r.shape[1]  # Number of receivers

    assert A1.shape == (Nf, Nr, Ns)
    assert A2.shape == (Nf, Nr, Ns)
    assert s.shape[0] == r.shape[0]
    assert q.shape == (Nf, Ns)

    c = speed_of_sound(T)
    k = 2 * np.pi * f / c

    # Coordinates
    r1, r2, alpha = geometrical_parameters(s, r)
    Dr = r2 - r1

    # Flow resistivity, impedance...
    if sigma is not None:
        Zs = impedance_miki_model(f, sigma)
        Q = spherical_reflection_coefficient(k, r2, Zs, alpha)

    # free field pressure
    p_free = A1 * q[:, None, :] / r1 * np.exp(-1j * k[:, None, None] * r1)

    # with ground reflection
    p = None
    if sigma is not None:
        p = p_free + q[:, None, :] * Q * A2 / r2 * np.exp(-1j * k[:, None, None] * r2)

    Dp = p / p_free if sigma is not None else None

    return p_free, p, Dp, Dr


def grid_area(polygon, dx, dy):
    """Sample interior of polygon area with square grid.

    Parameters
    ----------
    polygon : array [2, Npoints]
        Vertices defining the area where we want to create a grid
    spacing: int
        spacing of grid

    Returns
    -------
    [2, Npoints_grid]
        Points inside the polygon.
    """

    minx = polygon[0].min()
    miny = polygon[1].min()
    maxx = polygon[0].max()
    maxy = polygon[1].max()

    polygon = Polygon(polygon.T.tolist())

    # Create grid
    x_ = np.arange(minx, maxx, dx)
    y_ = np.arange(miny, maxy, dy)
    X, Y = np.meshgrid(x_, y_)

    grid_points = np.stack((X.flatten(), Y.flatten()), axis=0)
    mask_inside_polygon = np.empty(grid_points.shape[-1], dtype=bool)

    for i in range(grid_points.shape[-1]):
        mask_inside_polygon[i] = polygon.intersects(asPoint(grid_points[:, i]))
    # return grid_points[:, mask_inside_polygon]
    return grid_points, mask_inside_polygon, X.shape


def normal_to_segment(segment):
    # Gets the unitary normal to the segments
    dxdy = segment[:, 1] - segment[:, 0]
    dxdy[-1, :] = -dxdy[-1, :]
    return np.flipud(dxdy / np.linalg.norm(dxdy, axis=0))


def find_image_sources(s, wall):
    # https://reuk.github.io/wayverb/image_source.html
    # returns [Dim, Sources, Walls]

    n_wall = normal_to_segment(wall)
    im_s = s.T[:, :, None] - (
        2
        * n_wall
        * np.einsum("ik, ijk -> jk", n_wall, (s[:, :, None] - wall[:, None, 0]))[
            :, None
        ]
    )
    return np.moveaxis(im_s, 1, 0)


def segment_intersection(s1, s2):
    # finds intersection between a wall (s1) and an image source to receiver segment (s2).
    # http://www.cs.swan.ac.uk/~cssimon/line_intersection.html
    # s = [2D, 2points]
    u = (
        (s2[1, 0] - s2[1, 1]) * (s1[0, 0] - s2[0, 0])
        + (s2[0, 1] - s2[0, 0]) * (s1[1, 0] - s2[1, 0])
    ) / (
        (s2[0, 1] - s2[0, 0]) * (s1[1, 0] - s1[1, 1])
        - (s2[1, 1] - s2[1, 0]) * (s1[0, 0] - s1[0, 1])
    )
    t = (s1[0, 0] + u * (s1[0, 1] - s1[0, 0]) - s2[0, 0]) / (s2[0, 1] - s2[0, 0])
    return u, t


def clean_image_sources(s, r, wall):
    # Keep image sources which line of sight intersects with a wall.

    # s : sources (loudspeakers). [Dim, Ns]
    # r : receiver points. [Dim, Nr]
    # wall : walls. [Dim, Begin-End points, Nwalls]
    # Return for each receiver point an array of boolean saying which source is reflected by which wall
    # im_val : [Nr, Ns, Nw]

    Nr = r.shape[-1]  # Number of receivers
    Nw = wall.shape[-1]  # Number of walls
    Ns = s.shape[-1]  # Number of sources

    im_val = np.zeros((Nr, Ns, Nw), dtype=int)  # This is our output.

    im_s = find_image_sources(s, wall)
    for i in range(0, Nr):
        # Image source to receiver segments
        im_s_r = np.concatenate(
            (
                im_s[:, None],
                np.moveaxis(np.tile(r[:, i], (Ns, Nw, 1)), [0, 1, 2], [1, 2, 0])[
                    :, None
                ],
            ),
            axis=1,
        )

        for j in range(0, Nw):
            # Intersection between walls and image source-receiver segment
            u, t = segment_intersection(wall[:, :, j], im_s_r[:, :, :, j])
            im_val[i, :, j] = (0 <= u) & (1 >= u) & (0 <= t) & (1 >= t)
    return im_val


def objective_function_dipole(Lp, K1, K2, A):
    # Find the amplitude A that minimizes F
    # Lp is the sound level at the given receivers
    # K is the sum of the inverse distances source-receiver
    # (this is possible because we assume the same weights to all the sources!)
    F = Lp - 20 * np.log10(np.abs(A[0] * K1 + A[1] * K2) / 2e-6)
    return F


def objective_function(Lp, K, A):
    # Find the amplitude A that minimizes F
    # Lp is the sound level at the given receivers
    # K is the sum of the inverse distances source-receiver
    # (this is possible because we assume the same weights to all the sources!)
    F = Lp - 20 * np.log10(np.abs(A * K) / 2e-6)
    return F


def calculate_shm(s_latlon, slm_latlon, wall_latlon, r_latlon, Lp, f, T):
    """Calculates sound heat map

    Parameters
    ----------
    s_latlon : array[Dim, Ns]
        Sources coordinates in Latitutde-Longitude format [GPS].
    slm_latlon : array[Dim, Nslm]
        IoT slm coordinates in Latitutde-Longitude format [GPS].
    wall_latlon : array[Dim, Extreme points, Nwalls]
        Walls coordinates in Latitutde-Longitude format [GPS].
    r_latlon : array[Dim, Ns]
        Virtual receivers' coordinates in Latitutde-Longitude format [GPS].
    Lp : array[Nf, Nslm]
        dB SPL at SLMs
    f : array[Nf]
        Frequencies
    T : float
        Temperature in C
    A : Source strengths

    A_dir : Source direction

    Returns
    -------
    dL : array[Nf, Nr]
        dB SPL of the sound heat map
    """

    ###------------------------------###
    ### GPS to Cartesian Projections ###
    ###------------------------------###
    # We have to define how do we project from GPS to cartesian
    crs_wgs = proj.Proj(init="epsg:4326")  # assuming you're using WGS84 geographic
    crs_bng = proj.Proj(init="epsg:2154")  # Lambert93

    # Walls
    # cast your geographic coordinates to the projected system
    if wall_latlon is not None:
        x, y = proj.transform(crs_wgs, crs_bng, wall_latlon[1, :], wall_latlon[0, :])
        wall = np.concatenate((x[None, :, :], y[None, :, :]), axis=0)

    # SLMs
    # cast your geographic coordinates to the projected system
    x, y = proj.transform(crs_wgs, crs_bng, slm_latlon[1, :], slm_latlon[0, :])
    slm = np.concatenate((x[None, :], y[None, :]), axis=0)

    # Sources
    # cast your geographic coordinates to the projected system
    x, y = proj.transform(crs_wgs, crs_bng, s_latlon[1, :], s_latlon[0, :])
    s = np.concatenate((x[None, :], y[None, :]), axis=0)

    # Receiver
    # cast your geographic coordinates to the projected system
    x, y = proj.transform(crs_wgs, crs_bng, r_latlon[1, :], r_latlon[0, :])
    r = np.concatenate((x[None, :], y[None, :]), axis=0)

    Ns = s.shape[-1]
    Nslm = slm.shape[-1]
    Nr = r.shape[-1]

    ###------------------------------###
    ###   Find Sources Amplitudes    ###
    ###------------------------------###

    if wall_latlon is not None:
        # Given the Lp measured at the IoT microphones, we find the amplitude of our sources that better fits.
        im_s = find_image_sources(s, wall)  # Image sources to all the walls
        im_slm = clean_image_sources(
            s, slm, wall
        )  # "Effective" image sources to the given receivers

        # Create a masking matrix to only take into account the "effective sources".
        im_val_slm = np.concatenate(
            (np.ones((Nslm, Ns)), im_slm.reshape((Nslm, -1))), axis=-1
        )

        s_tot = np.concatenate((s, im_s.reshape((2, -1))), axis=1)
    else:
        im_val_slm = 1
        s_tot = s

    ###-------------------------------###
    ###      Calculate soundfield     ###
    ###-------------------------------###
    q = np.ones((f.shape[-1], s_tot.shape[-1]))  # weights

    # Monopole
    A1 = np.ones((f.shape[-1], slm.shape[-1], s_tot.shape[-1]))
    p_slm_1, *_ = calc_sound_field(None, T, f, A1, A1, s_tot, slm, q)
    K1 = np.sum(p_slm_1 * im_val_slm[None, :], axis=-1)
    A_sol = np.zeros(f.shape)
    for i in range(0, len(f)):
        x0 = np.ones((1))
        res_lsq = least_squares(
            lambda x: objective_function(Lp[i].flatten(), K1[i].flatten(), x), x0
        )
        A_sol[i] = res_lsq.x

    ###------------------------------###
    ###     Calculate Sound Map      ###
    ###------------------------------###

    if wall_latlon is not None:
        # Image sources for all the walls and new receivers
        im_val = clean_image_sources(s, r, wall)
        # Create a masking matrix to only take into account the "effective sources".
        im_val_tot = np.concatenate((np.ones((Nr, Ns)), im_val.reshape((Nr, -1))), axis=-1)
    else:
        im_val_tot = 1

    A_sol_all = A_sol[:, None, None] * np.ones(
        (f.shape[-1], r.shape[-1], s_tot.shape[-1])
    )
    q = np.ones((f.shape[-1], s_tot.shape[-1]))
    p_with_im_s, *_ = calc_sound_field(None, 25, f, A_sol_all, A_sol_all, s_tot, r, q)
    p_tot = np.sum(p_with_im_s * im_val_tot[None, :], axis=-1)  # Heat map in spl

    dL = 20 * np.log10(np.abs(p_tot / 2e-6))  # Heat map in dB
    return dL


@np.vectorize
def cardioid(theta, alpha=None):
    if alpha is None:
        return 1  # Image source directivity

    return (1 / 2) * (1 + np.cos((theta - math.pi) - alpha))


def directivity(xs, ys, xr, yr, Nf, alpha):
    vectorx = xs[None, :] - xr[:, None]
    vectory = ys[None, :] - yr[:, None]
    theta = np.arctan2(vectory, vectorx)
    D = cardioid(theta, alpha)
    return np.tile(D, (Nf, 1, 1))


def calculate_shm_dir(s_latlon, slm_latlon, wall_latlon, r_latlon, Lp, f, T, alpha):
    """Calculates sound heat map

    Parameters
    ----------
    s_latlon : array[Dim, Ns]
        Sources coordinates in Latitutde-Longitude format [GPS].
    slm_latlon : array[Dim, Nslm]
        IoT slm coordinates in Latitutde-Longitude format [GPS].
    wall_latlon : array[Dim, Extreme points, Nwalls]
        Walls coordinates in Latitutde-Longitude format [GPS].
    r_latlon : array[Dim, Ns]
        Virtual receivers' coordinates in Latitutde-Longitude format [GPS].
    Lp : array[Nf, Nslm]
        dB SPL at SLMs
    f : array[Nf]
        Frequencies
    T : float
        Temperature in C
    alpha : array[Ns]
        Source direction.

    Returns
    -------
    dL : array[Nf, Nr]
        dB SPL of the sound heat map
    """

    ###------------------------------###
    ### GPS to Cartesian Projections ###
    ###------------------------------###
    # We have to define how do we project from GPS to cartesian
    crs_wgs = proj.Proj(init="epsg:4326")  # assuming you're using WGS84 geographic
    crs_bng = proj.Proj(init="epsg:2154")  # Lambert93

    # Walls
    if wall_latlon is not None:
        # cast your geographic coordinates to the projected system
        xw, yw = proj.transform(crs_wgs, crs_bng, wall_latlon[1, :], wall_latlon[0, :])
        wall = np.concatenate((xw[None, :, :], yw[None, :, :]), axis=0)

    # SLMs
    # cast your geographic coordinates to the projected system
    xslm, yslm = proj.transform(crs_wgs, crs_bng, slm_latlon[1, :], slm_latlon[0, :])
    slm = np.concatenate((xslm[None, :], yslm[None, :]), axis=0)

    # Sources
    # cast your geographic coordinates to the projected system
    xs, ys = proj.transform(crs_wgs, crs_bng, s_latlon[1, :], s_latlon[0, :])
    s = np.concatenate((xs[None, :], ys[None, :]), axis=0)

    # Receiver
    # cast your geographic coordinates to the projected system
    xr, yr = proj.transform(crs_wgs, crs_bng, r_latlon[1, :], r_latlon[0, :])
    r = np.concatenate((xr[None, :], yr[None, :]), axis=0)

    Ns = s.shape[-1]
    Nslm = slm.shape[-1]
    Nr = r.shape[-1]
    Nf = f.shape[-1]

    ###------------------------------###
    ###   Find Sources Amplitudes    ###
    ###------------------------------###

    if wall_latlon is not None:
        # Given the Lp measured at the IoT microphones, we find the amplitude of our sources that better fits.
        im_s = find_image_sources(s, wall)  # Image sources to all the walls
        im_slm = clean_image_sources(
            s, slm, wall
        )  # "Effective" image sources to the given receivers

        # Create a masking matrix to only take into account the "effective sources".
        im_val_slm = np.concatenate(
            (np.ones((Nslm, Ns)), im_slm.reshape((Nslm, -1))), axis=-1
        )
        s_tot = np.concatenate((s, im_s.reshape((2, -1))), axis=1)
        Nb_img = im_s.reshape((2, -1)).shape[-1]
        alpha = np.concatenate((alpha, -99 * np.zeros(Nb_img)), axis=0)
    else:
        s_tot = s
        im_val_slm = np.array([1])

    xs_tot = s_tot[0]
    ys_tot = s_tot[1]

    ###-------------------------------###
    ###      Calculate soundfield     ###
    ###-------------------------------###
    q = np.ones((f.shape[-1], s_tot.shape[-1]))  # weights

    # Monopole
    A1 = directivity(xs_tot, ys_tot, xslm, yslm, Nf, alpha)
    p_slm_1, *_ = calc_sound_field(None, T, f, A1, A1, s_tot, slm, q)
    K1 = np.sum(p_slm_1 * im_val_slm[None, :], axis=-1)
    A_sol = np.zeros(f.shape)
    for i in range(0, len(f)):
        x0 = np.ones((1))
        res_lsq = least_squares(
            lambda x: objective_function(Lp[i].flatten(), K1[i].flatten(), x), x0
        )
        A_sol[i] = res_lsq.x

    ###------------------------------###
    ###     Calculate Sound Map      ###
    ###------------------------------###

    if wall_latlon is not None:
        # Image sources for all the walls and new receivers
        im_val = clean_image_sources(s, r, wall)
        # Create a masking matrix to only take into account the "effective sources".
        im_val_tot = np.concatenate((np.ones((Nr, Ns)), im_val.reshape((Nr, -1))), axis=-1)
    else:
        im_val_tot = np.array([1])

    A2 = directivity(xs_tot, ys_tot, xr, yr, Nf, alpha)
    A_sol_all = A_sol[:, None, None] * A2

    q = np.ones((f.shape[-1], s_tot.shape[-1]))
    p_with_im_s, *_ = calc_sound_field(None, 25, f, A_sol_all, A_sol_all, s_tot, r, q)
    p_tot = np.sum(p_with_im_s * im_val_tot[None, :], axis=-1)  # Heat map in spl

    dL = 20 * np.log10(np.abs(p_tot / 2e-6))  # Heat map in dB
    return dL
