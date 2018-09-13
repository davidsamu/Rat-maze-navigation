#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analysis functions.

@author: David Samu
"""

import numpy as np
import pandas as pd

from mb_mf_nav import utils

new_ax = np.newaxis


# %% Environment
# --------------

def get_dist_to_r_max(village, s_list=None):
    """Return distance of each location to maximually rewarded location."""

    if s_list is None:
        s_list = village.S

    s_max_r = village.max_r_state()
    d_max = {s: village.spl[s][s_max_r] for s in s_list}
    return d_max


def get_dist_to_closest_r(village, s_list=None):
    """Return distance of each location to closest rewarded location."""

    if s_list is None:
        s_list = village.S

    s_rwd = village.pos_r_states()
    s_sx_ord = {s: sorted(village.spl[s], key=village.spl[s].get)
                for s in s_list}
    d_cls = {s: village.spl[s][[sx for sx in sx_l if sx in s_rwd][0]]
             for s, sx_l in s_sx_ord.items()}
    return d_cls


# %% HC
# -----

def HC_loc_estimate(hc_state, eps=1e-9):
    """
    Return HC location estimate and inverse precision (entropy) over
    simulation time.
    """

    # Add some random noise to break ties randomly.
    # Currently relevant for visual estimate, where some locations have the
    # exact same visual stimulus. Same random values are added across columns
    # to avoid different ML estimates between them.
    rand_err = np.random.uniform(-eps, eps, hc_state.shape[0])
    hc_state_rnd = hc_state + np.tile(rand_err, (3, 1)).T

    # Maximum likelihood location estimate.
    grp_by_lvls = [lvl for lvl in hc_state.index.names if lvl != 'loc']
    g = hc_state_rnd.groupby(level=grp_by_lvls)
    maxLL = g.apply(lambda x: pd.Series([i[-1] for i in x.idxmax()]))
    maxLL.columns = hc_state.columns

    return maxLL


def HC_entropy(hc_state):
    """Return inverse precision (entropy) of location estimate."""

    grp_by_lvls = [lvl for lvl in hc_state.index.names if lvl != 'loc']
    g = hc_state.groupby(level=grp_by_lvls)
    H = g.apply(lambda x: x.apply(utils.entropy))

    return H


# %% GS
# -----

def GS_HC_norm(GS_HC, norm_ord, axis=(1, 2)):
    """Return GS - HC connectivity norm of given order."""

    if norm_ord == 1:
        gs_hc_norm = np.abs(GS_HC).sum(axis=axis)
    else:
        gs_hc_norm = (np.abs(GS_HC)**norm_ord).sum(axis=axis)**(1/norm_ord)

    return gs_hc_norm


def GS_HC_conn_mean(GS_HC, xvec, yvec, gs_circular):
    """Mean position of GS - HC connectivity per HC state."""

    # Get axis means.
    mxw = GS_HC.mean(axis=1)
    myw = GS_HC.mean(axis=2)

    # Normalize means.
    mxw /= mxw.sum(axis=1)[:, new_ax]
    myw /= myw.sum(axis=1)[:, new_ax]

    # Wrap position vectors around maximum of axis means.
    if gs_circular:
        xv, xprd = utils.get_periodic_pos_vec(xvec, mxw)
        yv, yprd = utils.get_periodic_pos_vec(yvec, myw)
    else:
        xv = xvec[new_ax, :]
        yv = yvec[new_ax, :]

    # Use them as weights on position vectors to get connectivity mean.
    mxc = np.sum(xv * mxw, axis=1)
    myc = np.sum(yv * myw, axis=1)

    # Take modulo to center on reference sheet.
    if gs_circular:
        for mc, vec, prd in [(mxc, xvec, xprd), (myc, yvec, yprd)]:
            mc = utils.shift_array_to_range(mc, vec.min(), vec.max(), prd)

    return np.array([mxc, myc])


def GS_pos_mean(gs_p, xvec, yvec, gs_circular):
    """Return weighted mean GS position estimate."""

    # Wrap position vectors around maximum of axis means.
    if gs_circular:
        mx, my = [gs_p.sum(axis=i)[new_ax, :] for i in [0, 1]]
        xv, xprd = utils.get_periodic_pos_vec(xvec, mx)
        yv, yprd = utils.get_periodic_pos_vec(yvec, my)
        xv = np.squeeze(xv)
        yv = np.squeeze(yv)
    else:
        xv = xvec
        yv = yvec

    # Take weighted mean along each axis.
    xm = np.dot(xv, gs_p.sum(axis=0))
    ym = np.dot(yv, gs_p.sum(axis=1))

    # Take modulo to center on reference sheet.
    if gs_circular:
        xm = utils.shift_to_range(xm, xvec.min(), xvec.max(), xprd)
        ym = utils.shift_to_range(ym, yvec.min(), yvec.max(), yprd)

    return xm, ym


def GS_pos_estimate(gs_state, gs_circular):
    """
    Return GS position estimate and inverse precision (entropy) over
    simulation time.
    """

    # Init params and grouping.
    g = gs_state.groupby(level=0)
    xvec = np.array(gs_state.columns)
    yvec = np.sort(np.array(gs_state.index.get_level_values('y').unique()))

    # Weighted mean position estimate.
    mpos = g.apply(lambda x: GS_pos_mean(x, xvec, yvec, gs_circular))
    mpos = pd.concat({i: pd.Series(mp, index=['x', 'y'])
                      for i, mp in mpos.items()}).unstack()

    # Inverse precision (entropy) of location estimate.
    H = g.apply(lambda x: utils.entropy(x.stack()))

    return mpos, H


# %% VC
# -----

def VC_HC_norm(VC_HC, norm_ord, axis=1):
    """Return VC - HC connectivity norm of given order."""

    if norm_ord == 1:
        vc_hc_norm = np.abs(VC_HC).sum(axis=axis)
    else:
        vc_hc_norm = (np.abs(VC_HC)**norm_ord).sum(axis=axis)**(1/norm_ord)

    return vc_hc_norm


# %% dlS
# ------

def follow_habit_moves(Q, village):
    """Get final states of each starting location by following habit moves."""

    # Following path as long as next max Q value is larger than current reward.
    final_s = {}
    for s in village.S:
        sx = s
        s_visited = []
        while village.R[sx] < Q[sx].max():
            u_poss = village.sus[sx].keys()  # only checking valid actions
            u_max = Q.loc[u_poss, sx].idxmax()
            sx = village.sus[sx][u_max]
            if sx in s_visited:  # if already been there, stop
                break
            s_visited.append(sx)  # save visited states
        final_s[s] = sx

    return final_s


# %% PFC
# ------

def action_seq(PFC, village):
    """Analysis of PFC action sequence plans after IBP."""

    # Real final state and reward of each sequence.
    vs, vr = zip(*[village.follow_path(u_seq, gamma=PFC.gamma)
                   for u_seq in PFC.u_seq])

    # Is sequence valid?
    is_valid = np.array([si is not None for si in vs]).mean()

    # Is reward estimate accurate?
    r_err = np.array([rpfc - ri for ri, rpfc in zip(vr, PFC.r_exp)]).mean()

    res = dict(is_valid=is_valid, r_err=r_err)

    return res
