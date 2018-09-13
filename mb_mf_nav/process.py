#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper functions processing and formatting simulation results.

@author: David Samu
"""

import numpy as np
import pandas as pd

from mb_mf_nav import analysis, utils


# %% Functions to init and record trial data of simulations.

def init_data(stats_list):
    """Init object to store step and trial data."""

    data = {stats_name: {} for stats_name in stats_list}
    return data


def get_rec_stats(village, u, s, r, stim, GS, HC, hc_ro, vS, dlS, data,
                  co_occ_pars, idx=None):
    """Record trial data and model state after a step during navigation."""

    res = {}

    if 'anim_data' in data:
        # Externally observable variables: action, state, reward, x-y position.
        x, y = village.animal_coords()
        res['anim_data'] = {'u': u, 's': s, 'r': r, 'x': x, 'y': y}

    if 'stim_data' in data:
        # Stimuli: motor feedback and visual input, both noise-free and noisy.
        res['stim_data'] = stim

    if 'gs_state' in data:
        # Instantaneous GS activity.
        res['gs_state'] = GS.P.copy()

    if 'hc_state' in data:
        # Instantaneous HC state estimate (full).
        res['hc_state'] = utils.get_copy(HC.s)

    if 'hc_ro' in data:
        # Instantaneous HC state estimate (read-out only).
        res['hc_ro'] = hc_ro.copy()

    if 'vs_state' in data:
        # Full vS reward estimate across all states.
        res['vs_state'] = vS.r.copy()

    if 'dls_state' in data:
        # Full dlS habit matrix across all states and actions.
        res['dls_state'] = dlS.Q.copy()

    if 'co_occs' in data:
        # Real location - HC state co-occurance frequency in last n steps.
        co_occ_pars['idx'] = update_co_occ_mat(hc_ro, village.S.index(s),
                                               **co_occ_pars)
        res['co_occs'] = co_occ_pars['co_occs'].copy()

    if 'gs_hc_pos' in data:
        # Mean position of GS - HC connectivity, per HC unit.
        C, xv, yv, circ = HC.GS_HC, GS.xvec, GS.yvec, GS.circular
        res['gs_hc_pos'] = analysis.GS_HC_conn_mean(C, xv, yv, circ)

    if 'gs_hc_max' in data:
        # Max value of GS - HC connectivity, per HC unit (proxy of spread).
        # Entropy calculation takes a lot of time --> approximated by max.
        # gs_hc_h = utils.entropy(HC.GS_HC.reshape((len(HC.s_names), -1)).T)
        res['gs_hc_max'] = HC.GS_HC.max(axis=(1, 2))

    # Put collected results into data object.
    for k, v in res.items():
        if idx is not None:
            data[k][idx] = v
        else:
            data[k] = v

    return


# %% Functions to format simulation results
# -----------------------------------------

def format_GS_HC_rec(res, s_hc, GS_HC):
    """Format recordings of GS - HC connectivity."""

    # GS - HC x-y mean position per HC unit.
    gs_hc_pos = pd.concat({i: pd.DataFrame(kv, index=['x', 'y'], columns=s_hc)
                           for i, kv in res['gs_hc_pos'].items()}).unstack()

    # GS - HC entropy per HC unit.
    gs_hc_h = pd.Series([utils.entropy(gs_hc.flatten()) for gs_hc in GS_HC],
                        index=s_hc)

    # GS - HC maximum value per HC unit.
    gs_hc_max = pd.Series([gs_hc.max() for gs_hc in GS_HC], index=s_hc)

    return gs_hc_pos, gs_hc_h, gs_hc_max


def format_learning(res, s_real, s_hc, GS_HC):
    """Process and format recorded learning results."""

    # Format results.
    res = pd.DataFrame(res).T
    res.index.name = 'step'

    # Pre-calculate some stats.
    s_real_h = pd.concat({i: pd.Series(kv, index=s_real)
                          for i, kv in res['s_real_h'].items()}).unstack()
    s_hc_h = pd.concat({i: pd.Series(kv, index=s_hc)
                        for i, kv in res['s_hc_h'].items()}).unstack()
    vc_hc_snr = pd.concat({i: pd.Series(kv, index=s_hc)
                           for i, kv in res['vc_hc_snr'].items()}).unstack()

    # GS - HC connectivity related stats.
    gs_hc_pos, gs_hc_h, gs_hc_max = format_GS_HC_rec(res, s_hc, GS_HC)

    return res, s_real_h, s_hc_h, vc_hc_snr, gs_hc_pos, gs_hc_h, gs_hc_max


def summarize_rec_data(data):
    """Return summary stats of recorded data."""

    # Warning: not all collectible data has a summary stats implemented below!
    # See get_rec_stats() above!

    stats = {}

    if 'hc_ro' in data:
        # Entropy across HC units average over samples.
        hc_ro_arr = np.array(list(data['hc_ro'].values()))
        stats['H HC ro'] = utils.entropy(hc_ro_arr.T).mean()

    if 'vs_state' in data:
        # Sum of vS reward estimates change (from first to last sample).
        vs_state = data['vs_state']
        stats['d vS'] = sum(vs_state[max(vs_state.keys())] - vs_state[0])

    if 'co_occs' in data:
        # Mean entropy of real location and HC state co-occurance frequencies.
        co_occs = data['co_occs'][max(data['co_occs'].keys())]
        stats['H HC co'] = np.nanmean(get_hc_co_occ_entropy(co_occs))
        stats['H loc co'] = np.nanmean(get_loc_co_occ_entropy(co_occs))

    return stats


def format_rec_data(tr_data, village, HC, gs_pars, vfeatures, idx_pars=[]):
    """Format recorded simulation data."""

    print('\nFormatting recorded data...')
    ret_list = []

    gs_xvec, gs_yvec = [utils.get_copy(gs_pars[v]) for v in ['xvec', 'yvec']]

    if 'anim_data' in tr_data:

        anim_data = pd.DataFrame(tr_data['anim_data']).T
        ret_list.append(anim_data)

    if 'stim_data' in tr_data:

        # Motor input.
        mot_keys = ['umot', 'vmot']
        mot_data = {i: pd.DataFrame({k: d[k] for k in mot_keys}).unstack()
                    for i, d in tr_data['stim_data'].items() if d is not None}
        mot_data = pd.DataFrame(mot_data).T
        mot_data.columns.set_levels(['x', 'y'], level=1, inplace=True)
        ret_list.append(mot_data)

        # Visual input.
        vis_keys = ['ovis', 'vvis']
        vis_data = {i: pd.DataFrame({k: d[k] for k in vis_keys}).unstack()
                    for i, d in tr_data['stim_data'].items() if d is not None}
        vis_data = pd.DataFrame(vis_data).T
        vis_data.columns.set_levels(vfeatures, level=1, inplace=True)
        ret_list.append(vis_data)

    if 'gs_state' in tr_data:

        gs_state = {k: pd.DataFrame(gs, columns=gs_xvec, index=gs_yvec)
                    for k, gs in tr_data['gs_state'].items()}
        gs_state = pd.concat(gs_state)
        gs_state.columns.rename('x', inplace=True)
        gs_state.index.rename('y', level=-1, inplace=True)
        ret_list.append(gs_state)

    if 'hc_state' in tr_data:

        hc_state = {k: pd.DataFrame(hc, index=HC.s_names)
                    for k, hc in tr_data['hc_state'].items()}
        hc_state = pd.concat(hc_state)
        hc_state.index.rename('loc', level=-1, inplace=True)
        hc_state = hc_state[HC.s_types]  # reorder columns
        ret_list.append(hc_state)

    if 'vs_state' in tr_data:

        vs_state = {k: pd.Series(vc, index=HC.s_names)
                    for k, vc in tr_data['vs_state'].items()}
        vs_state = pd.concat(vs_state)
        ret_list.append(vs_state)

    if 'dls_state' in tr_data:

        dls_state = {k: pd.DataFrame(dls, columns=HC.s_names, index=village.U)
                     for k, dls in tr_data['dls_state'].items()}
        dls_state = pd.concat(dls_state)
        dls_state.columns.rename('s', inplace=True)
        dls_state.index.rename('u', level=-1, inplace=True)
        ret_list.append(dls_state)

    # Set index level names.
    idx_lvl_names = idx_pars + ['step']
    for df in ret_list:
        levels = (None if not isinstance(df.index, pd.core.index.MultiIndex)
                  else list(range(len(idx_lvl_names))))
        df.index.set_names(idx_lvl_names, level=levels, inplace=True)

    return ret_list


# %% Functions to report simulation setup and progress
# ----------------------------------------------------

def report_sim_setup(fpref, lrn_kws, goals):
    """Report simulation setup."""

    print('\n\n' + fpref.replace('__', ', ',).replace('_', ' '))
    print('lrn_conns: {}, lrn_rwds: {}'.format(lrn_kws['lrn_conns'],
                                               lrn_kws['lrn_rwds']))
    print('goals: {}\n\n'.format(goals))


def report_progr(i, n, freq=1000):
    """Report progress of simulation."""

    if not i % freq:
        print('\t{} / {}'.format(str(i).rjust(5), n))
        # print('{}%'.format(int(100*istep/nsteps)))


def report_learning_progress(istep, nsteps, VC_HC, GS_HC, norm, mrh, mhh,
                             rop=None):
    """Report progress during learning simulations."""

    rep = '{}%'.format(int(100*istep/nsteps)).rjust(4)
    l_vc_hc = analysis.VC_HC_norm(VC_HC, norm).mean()
    l_gs_hc = analysis.GS_HC_norm(GS_HC, norm).mean()
    prog = '{} | VC-HC: {:.3f}, GS-HC: {:.3f}'.format(rep, l_vc_hc, l_gs_hc)
    prog += ' | H real: {:.2f}, H HC: {:.2f}'.format(mrh, mhh)
    if rop is not None:
        prog += ' | ro pow: {:.1f}'.format(rop)
    print(prog)


# %% Functions to get names of files and folders
# ----------------------------------------------

def sim_dir_name(fdir, nsteps, stim_pars, hc_pars, gs_pars, str_pars):
    """Return parameterized folder name of simulation."""

    fn = ('nsteps_{}_GSHCsharp_{}'.format(nsteps, hc_pars['gs_hc_sharp']) +
          '_hcpow_{}'.format(hc_pars['ro_pow']) +
          '_msig_{}'.format(int(stim_pars['mot_sig'])) +
          '_vbeta_{}'.format(stim_pars['vis_beta']) +
          '_lambda_{:.1f}'.format(str_pars['gamma']))
    fn = fdir + 'navigation/' + utils.format_to_fname(fn) + '/'
    return fn


def get_res_dir_name(random, nx=None, ny=None, p_state=None, p_path=None,
                     **kws):
    """Return folder name for simulation results."""

    if random:
        fdir = 'random_env/'
        env_dir = 'nx{}_ny{}_pstate{}_ppath{}'.format(nx, ny, p_state, p_path)
        fdir += utils.format_to_fname(env_dir) + '/'
    else:
        fdir = 'village/'

    return fdir


def conn_config_name(conn_config):
    """Return name of connectivity configuration."""

    n_conn_config = [cname for cname, on in conn_config.items() if on]
    n_conn_config = ' + '.join(n_conn_config)
    return n_conn_config


# %% Functions to process learning simulations
# --------------------------------------------

def get_co_occ_mat(s_hc_ml, n_s_real, n_s_hc):
    """Return world state - HC state co-occurance matrix."""

    co_occs = np.zeros((n_s_hc, n_s_real))
    for idx, n in s_hc_ml.items():
        co_occs[idx] = n

    return co_occs


def update_co_occ_mat(hc_ro, i_s_real, co_occs, co_occ_list, idx, nsteps):
    """Update world state - HC state co-occurance matrix."""

    # Go to next element in circular array.
    idx = (idx + 1) % nsteps

    # Remove estimate from left hand side of time window.
    if len(co_occ_list[idx]):
        i_s_real_left, hc_ro_left = co_occ_list[idx]
        co_occs[:, i_s_real_left] -= hc_ro_left

    # Add current estimate to matrix and list.
    co_occs[:, i_s_real] += hc_ro
    co_occ_list[idx] = [i_s_real, hc_ro]

    # Fix float imprecision.
    co_occs[(co_occs < 0) & (co_occs > -1e-10)] = 0

    return idx


def record_learning(village, s, GS, HC, hc_ro, dVC_HC, dGS_HC, co_occs, norm):
    """Collect data from a single step of connectivity learning."""

    # Animal's real and estimated position and location + uncertainty.
    x, y = village.animal_coords()
    gs_x, gs_y = analysis.GS_pos_mean(GS.P, GS.xvec, GS.yvec, GS.circular)
    gs_h = utils.entropy(GS.P.flatten())
    hc_ml = HC.s_names[hc_ro.argmax()]
    hc_h = utils.entropy(hc_ro)

    # Entropy between real state and HC state during last n steps.
    s_real_h = get_loc_co_occ_entropy(co_occs)
    s_hc_h = get_hc_co_occ_entropy(co_occs)

    # Connectivity stats.
    vc_hc_snr = np.mean(np.abs(HC.VC_HC), 1) / np.std(np.abs(HC.VC_HC), 1)
    gs_hc_pos = analysis.GS_HC_conn_mean(HC.GS_HC, GS.xvec, GS.yvec,
                                         GS.circular)
    # Entropy calculation below takes a lot of time --> approximated by max.
    # gs_hc_h = utils.entropy(HC.GS_HC.reshape((len(HC.s_names), -1)).T)
    gs_hc_max = HC.GS_HC.max(axis=(1, 2))

    # Connectivity change.
    dVC_HC_max = analysis.VC_HC_norm(dVC_HC, norm).max()
    dGS_HC_max = analysis.GS_HC_norm(dGS_HC, norm).max()

    # Compensate for the magnitude difference between VC and GS input (GS is a
    # PD, VC is not).
    dVC_HC_max /= dVC_HC.shape[1]

    res = {'s': s, 'x': x, 'y': y,
           'gs_x': gs_x, 'gs_y': gs_y, 'gs_h': gs_h,
           'hc_ro': hc_ro, 'hc_ml': hc_ml, 'hc_h': hc_h,
           's_real_h': s_real_h, 's_hc_h': s_hc_h,
           'vc_hc_snr': vc_hc_snr,
           'gs_hc_pos': gs_hc_pos, 'gs_hc_max': gs_hc_max,
           'dVC_HC_max': dVC_HC_max, 'dGS_HC_max': dGS_HC_max}

    return res


def get_co_occ_matrix(res, last_n, s_real, s_hc):
    """Get number of real and estimated state co-occurances."""

    s = np.array(res['s'])[-last_n:]
    hc_ml = np.array(res['hc_ml'])[-last_n:]
    co_occs = np.zeros((len(s_hc), len(s_real)))
    for si, hc_mli in zip(s, hc_ml):
        co_occs[s_hc.index(hc_mli), s_real.index(si)] += 1
    co_occs /= co_occs.sum()

    return co_occs


def norm_co_occ_matrix(co_occs, axis=0):
    """Normalize co-occurance matrix."""

    co_sum = co_occs.sum(axis=axis)
    co_occs_norm = np.divide(co_occs, co_sum, out=np.zeros_like(co_occs),
                             where=(co_sum != 0))
    return co_occs_norm


def get_s_order(co_occs, s_hc=None):
    """
    Return learned HC state order from co-occurance matrix by taking best
    matching pairs world location - HC state pairs.
    """

    # Greedy approach: just go through items from max to min.
    free_rows, free_cols = [list(range(n)) for n in co_occs.shape]
    s_ord = -np.ones(co_occs.shape[0], dtype=int)

    co_normed = norm_co_occ_matrix(co_occs)
    isrtd = np.unravel_index(co_normed.argsort(axis=None)[::-1], co_occs.shape)
    for irow, icol in zip(isrtd[0], isrtd[1]):
        # If neither row nor column has been taken yet, it's a match!
        if irow in free_rows and icol in free_cols:
            s_ord[icol] = irow
            free_rows.remove(irow)
            free_cols.remove(icol)
            if not len(free_rows) or not len(free_cols):
                break

    # Unmatched HC states go to the end.
    s_ord[s_ord == -1] = free_rows

    # Also sort state name list, if provided.
    s_name_srtd = np.array(s_hc)[s_ord] if s_hc is not None else None

    return s_ord, s_name_srtd


def get_loc_co_occ_entropy(co_occs, v_repl_nan=None):
    """Return entropy of each village location state across HC activations."""

    return utils.get_row_entropy(co_occs, v_repl_nan)


def get_hc_co_occ_entropy(co_occs, v_repl_nan=None):
    """Return entropy of each HC state across village locations."""

    return utils.get_row_entropy(co_occs.T, v_repl_nan)
