"""Weight sound pressure levels"""

import numpy as np


def dB(p, pre=20e-6):
    """Return p as dB SPL."""
    return 20 * np.log10(np.abs(p) / np.sqrt(2) / pre)


def A(f):
    """A-weighting.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.

    Returns
    -------
    array_like, shape (nf,)
        SPL weightings

    """
    R_a = (
        12194 ** 2
        * f ** 4
        / (
            (f ** 2 + 20.6 ** 2)
            * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
            * (f ** 2 + 12194 ** 2)
        )
    )
    return 20 * np.log10(R_a) + 2


def C(f):
    """C-weighting.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.

    Returns
    -------
    array_like, shape (nf,)
        SPL weightings

    """
    f[f == 0] = np.spacing(1)
    R_c = 12194 ** 2 * f ** 2 / ((f ** 2 + 20.6 ** 2) * (f ** 2 + 12194 ** 2))
    return 20 * np.log10(R_c) + 0.06


def Z(f):
    """Z-weighting.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.

    Returns
    -------
    array_like, shape (nf,)
        SPL weightings

    """
    np.asarray(f)
    return np.zeros(f.shape)


def lta(f):
    """Long term average music spectrum of music.

    Parameters
    ----------
    f : ndarray
        Frequencies

    Returns
    -------
    ndarray with shape `f.shape`
        SPL weightings.

    Notes
    -----
    Based on

        Elowsson, A., & Friberg, A. (2017). Long-term average spectrum in
        popular music and its rela-tion to the level of the percussion. In
        142nd Audio Engineering Society International Convention 2017, AES 2017
        (pp. 1–12).

    """
    f = np.asarray(f)

    shift = -16.412  # == y1(100) == y2(100)
    p11 = -0.000907
    p12 = 0.256
    p13 = -32.942 - shift
    p21 = -0.000183
    p22 = 0.0213
    p23 = -16.735 - shift
    u = 199.35423956
    v = -293.47038447
    f_switch = 94.12926965

    x1 = u * np.log10(f[f <= f_switch]) + v
    x2 = u * np.log10(f[f >= f_switch]) + v
    y1 = p11 * x1 ** 2 + p12 * x1 + p13
    y2 = p21 * x2 ** 2 + p22 * x2 + p23

    y = np.zeros(f.shape)
    y[f <= f_switch] = y1
    y[f >= f_switch] = y2

    return y


def pink(f):
    """pink-weighting with 0 dB at 100 Hz.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.

    Returns
    -------
    array_like, shape (nf,)
        weightings

    """
    f[f == 0] = np.spacing(1)
    return -10 * np.log10(f) + 10 * np.log10(100)


def blue(f):
    """blue-weighting with 0 dB at 100 Hz.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.

    Returns
    -------
    array_like, shape (nf,)
        weightings

    """
    return -pink(f)


def grey(f):
    """grey-weighting with 0 dB at 100 Hz.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.

    Returns
    -------
    array_like, shape (nf,)
        weightings

    """
    return -A(f)


def white(f):
    """No weighting.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.

    Returns
    -------
    array_like, shape (nf,)
        weightings

    """
    return np.zeros(f.shape)


def weightfunc(weighting):
    """Return a weighting function from string."""
    if weighting == "C":
        func = C
    elif weighting == "A":
        func = A
    elif weighting == "Z":
        func = Z
    elif weighting == "pink":
        func = pink
    elif weighting == "lta":
        func = lta
    elif weighting == "white":
        func = white
    elif weighting == "blue":
        func = blue
    elif weighting == "grey":
        func = grey
    else:
        raise ValueError("Unknown weighting {}".format(weighting))

    return func


def weight(f, x, weighting="Z", normalize=None):
    """Weight x.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.
    x : ndarray, (nf, ...)
        Data to be weighted.
    weighting : str, optional
        'A', 'C', 'Z', 'lta', 'pink', 'white'
    normalize : str, optional
        None, 'energy'

    Returns
    -------
    array_like, shape (nf,)
        Weighted x

    Raises
    ------
    ValueError
        Description

    """
    f = np.atleast_1d(f)

    w = weightfunc(weighting)(f)

    x_weighted = x * np.power(10, w[:, None] / 20)

    if normalize is None:
        pass
    elif normalize == "energy":
        E_before = np.trapz(np.abs(x) ** 2, axis=0)
        E_weighted = np.trapz(np.abs(x_weighted) ** 2, axis=0)
        x_weighted = x_weighted * np.sqrt(E_before / E_weighted)
    elif isinstance(normalize, float):
        x_weighted = x / normalize
    else:
        raise ValueError("Unknown normalization {}".format(normalize))

    return x_weighted


def total_spl(f, p, weighting="Z"):
    """Compute total weighted SPL.

    Parameters
    ----------
    f : array_like, shape (nf,)
        Frequencies.
    p : ndarray, (nf, nr)
        Pressure at nr points and nf frequencies.
    weight : str, optional
        SPL weighting to be applied. 'A', 'C', or 'Z'.

    Returns
    -------
    ndarray, (nr)
        Total weighted SPL at nr points.

    """
    mean_square_weighted = np.abs(weight(f, p, weighting=weighting)) ** 2
    total_mean_square_weighted = np.trapz(mean_square_weighted, f, axis=0)
    return dB(np.sqrt(total_mean_square_weighted))
