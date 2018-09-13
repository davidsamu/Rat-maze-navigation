#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for foraging project.

@author: David Samu
"""


import os
import copy
import string
import pickle
import warnings

from itertools import product

import numpy as np
import pandas as pd

from scipy.stats import entropy as sp_entropy

eps = 1e-6


# %% System I/O functions
# -----------------------

def create_dir(f):
    """Create directory if it does not already exist."""

    d = os.path.dirname(f)
    if d and not os.path.exists(d):
        os.makedirs(d)
    return


def write_objects(obj_dict, fname):
    """Write out dictionary object into pickled data file."""

    create_dir(fname)
    pickle.dump(obj_dict, open(fname, 'wb'))


def read_objects(fname, obj_names=None):
    """Read in objects from pickled data file."""

    data = pickle.load(open(fname, 'rb'))

    # Unload objects from dictionary.
    if obj_names is None:
        objects = data  # all objects
    elif isinstance(obj_names, str):
        objects = data[obj_names]   # single object
    else:
        objects = [data[oname] for oname in obj_names]  # multiple objects

    return objects


def get_copy(obj, deep=True):
    """Returns (deep) copy of object."""

    copy_obj = copy.deepcopy(obj) if deep else copy.copy(obj)
    return copy_obj


# %% String formatting functions
# ------------------------------

def format_to_fname(s):
    """Format string to file name compatible string."""

    valid_chars = "_ .%s%s" % (string.ascii_letters, string.digits)
    fname = ''.join(c for c in s if c in valid_chars)
    fname = fname.replace(' ', '_')
    fname = fname.replace('.', '_')
    return fname


def form_str(v, njust):
    """Format float value into string for reporting."""

    vstr = ('%.1f' % v).rjust(njust)
    return vstr


# %% General function for data structure formatting
# -------------------------------------------------

def find_key(d, v, to_tuple=True):
    """Return key for value."""

    keys, vals = zip(*d.items())
    if to_tuple:  # convert to tuple to be able to check for lists and arrays
        vals = [tuple(val) for val in vals]
        v = tuple(v)
    k = keys[vals.index(v)] if v in vals else None

    return k


def sort_keys_by_val(d):
    """Return list of dictionary keys sorted by value."""
    return sorted(d, key=d.get)


def flatten(list_of_lists):
    """Flatten list of lists."""
    return [item for sublist in list_of_lists for item in sublist]


def merge(list_of_dicts):
    """Merge list of dicts into a single dict."""
    return {k: v for d in list_of_dicts for k, v in d.items()}


def vectorize(sel_idx, index):
    """
    Return indexed vector with all zero values except for selected index,
    which is set to 1.
    """

    vec = pd.Series(0., index=index)
    vec[sel_idx] = 1

    return vec


def create_long_DF(mi_df, level_names=None):
    """Create long-format Pandas DataFrame from MultiIndex DF."""

    if level_names is not None:
        mi_df.index.set_names(level_names, inplace=True)
    return mi_df.reset_index()


# %% Array shifting, wrapping and centering functions
# ---------------------------------------------------

def row_shift(arr, num, circ, vfill=np.nan):
    """Fast row-wise matrix shift implementation."""

    res = np.empty_like(arr)
    if num > 0:
        res[:num, :] = arr[-num:, :] if circ else vfill
        res[num:, :] = arr[:-num, :]
    elif num < 0:
        res[num:, :] = arr[:-num, :] if circ else vfill
        res[:num, :] = arr[-num:, :]
    else:
        res = arr
    return res


def col_shift(arr, num, circ, vfill=np.nan):
    """Fast column-wise matrix shift implementation."""

    res = np.empty_like(arr)
    if num > 0:
        res[:, :num] = arr[:, -num:] if circ else vfill
        res[:, num:] = arr[:, :-num]
    elif num < 0:
        res[:, num:] = arr[:, :-num] if circ else vfill
        res[:, :num] = arr[:, -num:]
    else:
        res = arr
    return res


def shift(M, xshf, yshf, circ, vfill=np.nan):
    """Shift matrix along both axes."""

    return row_shift(col_shift(M, xshf, circ, vfill), yshf, circ, vfill)


def shift_to_range(v, vmin, vmax, vprd):
    """
    Shift value in min - max range. Assumes that v is away at most by one
    period!
    """

    if v > vmax:
        v -= vprd
    if v < vmin:
        v += vprd

    return v


def shift_array_to_range(v, vmin, vmax, vprd):
    """
    Shift each element of array in min - max range. Assumes that v is away at
    most by one period!
    """

    v[v > vmax] -= vprd
    v[v < vmin] += vprd

    return v


def get_periodic_pos_vec(vec, mw):
    """Return periodic position vector wrapped around maximum of values."""

    prd = len(vec) * (vec[1] - vec[0])
    n, hi = len(vec), round(len(vec)/2)
    i_max = mw.argmax(axis=1)
    v = np.tile(vec, (mw.shape[0], 1))

    # Indices to add one period to.
    idx = np.where(i_max > hi)[0]
    if len(idx):
        dv = i_max - hi
        lists = [list(range(i)) for i in range(hi+1)]
        ri, ci = zip(*[(dv[i] * [i], lists[dv[i]]) for i in idx])
        ri, ci = [[i for subl in l for i in subl] for l in (ri, ci)]
        v[ri, ci] += prd

    # Indices to subtract one period from.
    idx = np.where(i_max < hi)[0]
    if len(idx):
        dv = hi - i_max
        lists = [list(range(n-i, n)) for i in range(hi+1)]
        ri, ci = zip(*[(dv[i] * [i], lists[dv[i]]) for i in idx])
        ri, ci = [[i for subl in l for i in subl] for l in (ri, ci)]
        v[ri, ci] -= prd

    return v, prd


def center_axis(pos_df, vmin, vmax, vprd, by_cols=None):
    """Center positions in DataFrame along axis."""

    if by_cols is None:
        by_cols = pos_df.columns

    vh = (vmax-vmin)/2

    p_final = pos_df[by_cols].iloc[-1]  # final positions of selected columns

    # Do half period shift first if required to prevent being stuck
    # wrapped around the edges.
    p_final_sftd = shift_array_to_range(p_final.copy(), vmin+vh, vmax+vh, vprd)
    if p_final_sftd.std() < p_final.std():
        pos_df = shift_array_to_range(pos_df, vmin+vh, vmax+vh, vprd)
        p_final = p_final_sftd

    pm = p_final.mean()

    # Shift and wrap coordinates.
    pos_df -= pm
    pos_df = shift_array_to_range(pos_df, -vh, vh, vprd)

    return pos_df


# %% Maths functions
# ------------------

def dist_mat(x, y, xvec, yvec, circular):
    """Return distance matrix from point (x0, y0) at grid coordinates."""

    xv, yv = np.meshgrid(xvec, yvec)
    if circular:
        # Get axis period params.
        xres, yres = [vec[1] - vec[0] for vec in (xvec, yvec)]
        xprd, yprd = [max(vec) - min(vec) + res
                      for vec, res in [(xvec, xres), (yvec, yres)]]
        # Take min across distances of original and all 8 adjacent positions.
        M = np.array([np.sqrt((xv-x+xs*xprd)**2 + (yv-y+ys*yprd)**2)
                      for xs, ys in product([-1, 0, 1], repeat=2)]).min(0)
    else:
        M = np.sqrt((xv-x)**2 + (yv-y)**2)  # absolute distance

    return M


def rand_mat(nrow, ncol, rng=0):
    """
    Return matrix of standard normal random data of given shape and
    clipped to +- given range rng.
    """

    if rng == 0:
        R = np.zeros((nrow, ncol))
    else:
        R = np.random.randn(nrow, ncol)
        R = np.clip(R, -rng, rng)

    R = R.squeeze()

    return R


def softmax(x, tau=1):
    """Compute softmax values for vector x at temperature tau"""

    e_x = np.exp((x - np.max(x)) / tau)
    return e_x / e_x.sum()


def pow_read_out(p, ro_pow):
    """Return power function read-out of probability vector p."""

    if ro_pow == 'ML':  # maximum likelihood estimate
        ro = np.zeros(len(p))
        ro[p.argmax()] = 1
    else:
        ro = p ** ro_pow
        ro = ro / ro.sum()

    return ro


def entropy(p, base=None):
    """Compute entropy of probability distribution p."""

    H = sp_entropy(p, base=base)
    return H


def get_row_entropy(co_occs, v_repl_nan=None):
    """Return entropy of co-occurance matrix by axis."""

    # Suppressing warnings of all zero distributions (e.g. state not visited).
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        h = entropy(co_occs)

    # Replace NaN values.
    if v_repl_nan is not None:
        h[np.isnan(h)] = v_repl_nan

    return h


def D_KL(p, q, base=None):
    """Compute  Kullback-Leibler divergence between PDs p and q."""

    D = sp_entropy(p, q, base=base)
    return D


# %% Functions to generate input stimuli.

def noisy_vis_input(vis_input, vis_alpha, vis_beta):
    """Add Beta dist noise to visual input."""

    # Add noise by sampling from Beta distr, and transform to -1/+1 range.
    err = np.random.beta(vis_alpha, vis_beta, len(vis_input))
    vis_noisy = np.sign(vis_input) * (2 * (1 - err) - 1)

    return vis_noisy


def noisy_mot_input(mot_input, mot_sig):
    """Add Gaussian noise to motor input."""

    # Add noise sampled from truncated Gaussian distr.
    err = mot_sig * rand_mat(2, 1, rng=2)
    mot_noisy = mot_input + err

    return mot_noisy
