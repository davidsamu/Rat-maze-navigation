#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper functions to set up environment and model.

@author: David Samu
"""

import itertools
from random import shuffle

import numpy as np
import pandas as pd

from mb_mf_nav.village import Village, RandomEnv
from mb_mf_nav.brain_model import GridSystem, Hippocampus, VisualCortex
from mb_mf_nav.brain_model import VentralStriatum, DorsolateralStriatum
from mb_mf_nav.brain_model import PrefrontalCortex
from mb_mf_nav import analysis, plotting, process, utils

new_ax = np.newaxis


# %% Functions to init simulation parameters
# ------------------------------------------

def init_lrn_kws(model_par_kws, new_paths=None, del_paths=None, new_rwds=None,
                 lrn_conns=None, lrn_rwds=None, lrn_pars=None):
    """Init learning kws."""

    mod_conns_set = model_par_kws['hc_pars']['conns_set']
    mod_rwd_set = model_par_kws['st_pars']['learned']
    env_changed = new_paths is not None or del_paths is not None
    rwd_changed = new_rwds is not None

    # Init learning flags.
    if lrn_conns is None:
        lrn_conns = not mod_conns_set or env_changed
    if lrn_rwds is None:
        lrn_rwds = not mod_rwd_set or lrn_conns or rwd_changed

    # Collect params.
    lrn_kws = dict(lrn_conns=lrn_conns, lrn_rwds=lrn_rwds, lrn_pars=lrn_pars)

    return lrn_kws


def init_co_occ_pars(n_hc_s, n_village_s, nsteps=200):
    """Init data structures to track location - HC state co-occurances."""

    co_occs = np.zeros((n_hc_s, n_village_s))
    co_occ_list = nsteps * [[]]
    co_occ_pars = dict(nsteps=nsteps, idx=-1, co_occs=co_occs,
                       co_occ_list=co_occ_list)
    return co_occ_pars


# %% Functions to init environment and model
# ------------------------------------------

def init_env(env_pars):
    """Init enviroment."""

    if env_pars['random']:

        # Create random discrete navigation environment.
        village = RandomEnv(**env_pars)
        plotting.plot_village(village)
        fdir_temp = process.get_res_dir_name(**env_pars)

    else:

        # Use manually designed Village as navigation environment.
        village = Village(res=env_pars['res'])
        fdir_temp = process.get_res_dir_name(False)

    print('\nn states: {}, n paths: {}\n'.format(len(village.S),
                                                 village.n_paths()))

    # Get environment limits and size.
    env_pars.update(village.get_dimensions())

    return village, fdir_temp


def update_env(village, new_paths=None, del_paths=None, new_rwds=None):
    """Update environment with new paths and/or reward values."""

    # Add shortcuts.
    if new_paths is not None:
        for s1, s2 in new_paths:
            village.add_path(s1, s2)

    # Add obstacles.
    if del_paths is not None:
        for s1, s2 in del_paths:
            village.remove_path(s1, s2)

    # Revalue some states.
    r_orig = utils.get_copy(village.R)
    if new_rwds is not None:
        for loc, rwd in new_rwds.items():
            village.change_reward(loc, rwd)

    # Get file prefix for saving results.
    f_pref = ('learned_env' if new_paths is None and del_paths is None else
              'shortcuts' if new_paths is not None and del_paths is None else
              'obstacles' if new_paths is None and del_paths is not None else
              'shortcuts_obstacles')
    f_pref += '__' + ('learned_rwd' if new_rwds is None else 'revalued')

    return r_orig, f_pref


def restore_env(village, new_paths=None, del_paths=None, r_orig=None):
    """Restore environment."""

    # Remove added shortcuts.
    if new_paths is not None:
        for s1, s2 in new_paths:
            village.remove_path(s1, s2)

    # Remove added obstacles.
    if del_paths is not None:
        for s1, s2 in del_paths:
            village.add_path(s1, s2)

    # Revalue some states.
    if r_orig is not None:
        village.R = r_orig


def init_locs(village, n_reset, goals=None, starts=None, goal_type='max'):
    """Return goal and start location for each trial."""

    goals, starts = init_start_goal(village, goals, starts, goal_type='max')
    start_locs = utils.flatten([itertools.repeat(x, n_reset) for x in starts])
    shuffle(start_locs)   # init random starting location for each trial

    return goals, start_locs


def init_start_goal(village, s_goal=None, s_start=None, goal_type='max'):
    """Init goal location and starting locations."""

    if s_goal is None:
        s_goal = ([village.max_r_state()] if goal_type == 'max' else
                  village.pos_r_states())

    if s_start is None:
        s_start = [s for s in village.S if s not in s_goal]

    return s_goal, s_start


def init_GS_params(env_dims, gs_pars):
    """Init GS params."""

    # Get params.
    res = env_dims['res']
    xmin, xmax, xsize = [env_dims[k] for k in ['xmin', 'xmax', 'xsize']]
    ymin, ymax, ysize = [env_dims[k] for k in ['ymin', 'ymax', 'ysize']]
    gs_pad, gs_step_res = gs_pars['pad'], gs_pars['step_res']

    # Get derived GS params.
    gs_nx, gs_ny = [int(gs_step_res*(size+2*gs_pad)/res+1)  # GS resolution
                    for size in [env_dims['xsize'], env_dims['ysize']]]
    gs_xmin, gs_xmax = xmin - gs_pad, xmax + gs_pad  # GS position limits
    gs_ymin, gs_ymax = ymin - gs_pad, ymax + gs_pad
    gs_xvec = np.linspace(gs_xmin, gs_xmax, gs_nx)  # index vectors of
    gs_yvec = np.linspace(gs_ymin, gs_ymax, gs_ny)  # GS positions

    # Unite params.
    gs_p = dict(nx=gs_nx, xmin=gs_xmin, xmax=gs_xmax, xvec=gs_xvec,
                ny=gs_ny, ymin=gs_ymin, ymax=gs_ymax, yvec=gs_yvec)
    gs_pars.update(gs_p)

    return


def init_model(village, stim_pars, conn_pars, gs_pars, hc_pars, st_pars,
               pfc_pars, as_dict=True):
    """Create model using optimal connectivity."""

    # Init some params.
    S = village.S
    vfeats = village.vfeatures

    # Init sensory - HC connectivities.
    if hc_pars['conns_set']:  # set connectivities to 'optimal' values
        GS_HC = init_GS_HC(S, village.S_pos, **gs_pars, **hc_pars)
        VC_HC = init_VC_HC(village.V, S)
    else:                     # connectivities to be learned
        GS_HC = init_rand_GS_HC(S, gs_pars['nx'], gs_pars['ny'])
        VC_HC = init_rand_VC_HC(S, len(vfeats))

    # Init model regions.
    GS = GridSystem(**gs_pars, **stim_pars)
    VC = VisualCortex(vfeats, **stim_pars)
    HC = Hippocampus(S, GS_HC, VC_HC, **hc_pars)
    vS = VentralStriatum(HC.ns, **st_pars)
    dlS = DorsolateralStriatum(HC.ns, len(village.U), **st_pars)
    PFC = PrefrontalCortex(**pfc_pars)

    # Init vS reward estimates and dlS habits.
    if st_pars['learned']:  # already learned: initialize to 'optimal' values
        vS.r = init_R(village)
        dlS.Q = init_Q(village, st_pars['gamma'])

    ret = (dict(GS=GS, VC=VC, HC=HC, vS=vS, dlS=dlS, PFC=PFC) if as_dict else
           [GS, VC, HC, vS, dlS, PFC])

    return ret


def get_conn_on_df(prd_len, full_only=False):
    """Return connectivity configuration DataFrame."""

    conns = ['VC-HC', 'GS-HC', 'HC-GS']
    conn_on = ([(1, 1, 1)] if full_only else   # full connectivity only
               [(1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 1, 1), (1, 1, 1)])
    conn_on = pd.DataFrame(conn_on, columns=conns)
    idxs = sum([prd_len*[i] for i in range(len(conn_on))], [])
    conn_on = conn_on.loc[idxs].reset_index(drop=True)

    return conn_on


# %% Functions to init GS - HC connectivity
# -----------------------------------------

def init_GS_HC(S, S_pos, xvec, yvec, gs_hc_sharp, circular, **kws):
    """Init GS to HC connectivity with 'optimal' closeness weighted values."""

    # Maximum distance on grid cell map to normalize each connectivity to the
    # same scale.
    rng = [max(xvec)-min(xvec), max(yvec)-min(yvec)]
    rng = [rng[0]/2, rng[1]/2] if circular else rng
    dmax = np.linalg.norm(rng)

    # Connection weight is inversely proportional to the distance between
    # the state's and the grid cell's location.
    xv, yv = np.meshgrid(xvec, yvec)
    GS_HC = np.zeros((len(S_pos), len(yvec), len(xvec)))
    for i, s in enumerate(S):
        x, y = S_pos[s]
        gs_hc = utils.dist_mat(x, y, xvec, yvec, circular)  # distance from loc
        gs_hc = (dmax - gs_hc) / dmax  # normalized closeness
        gs_hc = gs_hc ** gs_hc_sharp   # sharpen connectivity profile
        GS_HC[i, :, :] = gs_hc / gs_hc.sum()

    return GS_HC


def init_rand_GS_HC(S, nx, ny, alpha=2, beta=2, norm=2):
    """Init random GS to HC connectivity."""

    GS_HC = np.array([np.random.beta(alpha, beta, (ny, nx)) for s in S])

    if norm is not None:
        GS_HC /= analysis.GS_HC_norm(GS_HC, norm)[:, new_ax, new_ax]

    return GS_HC


# %% Functions to init VC - HC connectivity
# -----------------------------------------

def init_VC_HC(V, s_names):
    """Init VC to HC connectivity with optimal visual - state mapping."""

    VC_HC = np.array([V[s] for s in s_names], dtype=float)
    return VC_HC


def init_rand_VC_HC(s_names, nvfeats, alpha=2, beta=2, norm=2):
    """Init random GS to HC connectivity."""

    # Sample values from beta distribution.
    V = {s: np.random.beta(alpha, beta, nvfeats) for s in s_names}
    # Flip sign of some values randomly.
    V = {s: (2 * np.random.binomial(1, .5, len(V[s])) - 1) * V[s] for s in V}

    VC_HC = init_VC_HC(V, s_names)

    if norm is not None:
        VC_HC /= analysis.VC_HC_norm(VC_HC, norm)[:, new_ax]

    return VC_HC


# %% Functions to init reward estimate and habits in Striatum
# -----------------------------------------------------------

def init_R(village):
    """Init reward estimate in vS."""

    R = np.array([village.R[loc] for loc in village.S], dtype=float)
    return R


def init_Q(village, gamma):
    """Init Q-value of (s, u) pairs."""

    # Q(s, u) = rmax * gamma**nstep(sx, s_rmax)
    # Simplified, but optimal assuming a single reward state.

    nstep = analysis.get_dist_to_r_max(village)

    Q = np.zeros((len(village.U), len(village.S)))
    for i_s, s in enumerate(village.S):
        for i_u, u in enumerate(village.U):
            if u in village.sus[s]:
                sx = village.sus[s][u]
                Q[i_u, i_s] = gamma**nstep[sx]

    rmax = max(village.R.values())
    Q *= rmax

    return Q
