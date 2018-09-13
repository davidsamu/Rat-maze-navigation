#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper functions for model simulation.

@author: David Samu
"""

import numpy as np
import pandas as pd

from mb_mf_nav import analysis, plotting, process, setup, utils

new_ax = np.newaxis


# %% Functions to perform complete simulations
# --------------------------------------------

def IBP_vs_HAS(village, mod_par_kws, ibp_pars, lrn_pars, step_stats, trl_stats,
               new_paths=None, del_paths=None, new_rwds=None, goals=None,
               starts=None, lrn_conns=None, lrn_rwds=None, n_reset=100,
               max_steps=100, nav_types=None, fdir_dm=None, restore_vill=True):
    """
    Run a bunch of trials from given starting locations using either IBP or HAS
    to navigate, collect and plot results. Alteration of environment (adding
    shortcuts or obstacles) is possible.
    """

    # Create model using original (learned) environment.
    mod_kws = setup.init_model(village, **mod_par_kws)

    # Update environment with new paths and rewards.
    r_orig, fpref = setup.update_env(village, new_paths, del_paths, new_rwds)

    # Set up learning params.
    lrn_kws = setup.init_lrn_kws(mod_par_kws, new_paths, del_paths, new_rwds,
                                 lrn_conns, lrn_rwds, lrn_pars)

    # Params and objects for world - HC state co-occurance calculation.
    co_pars = None
    if 'co_occs' in step_stats or 'co_occs' in trl_stats:
        co_pars = setup.init_co_occ_pars(len(mod_kws['HC'].s_names),
                                         len(village.S), nsteps=500)

    # Types of navigation to use.
    if nav_types is None:
        nav_types = ['optimal', 'HAS', 'IBP', 'random']

    # Set up goal states and start states.
    goals, start_locs = setup.init_locs(village, n_reset, goals, starts,
                                        goal_type='max')

    # Report config.
    process.report_sim_setup(fpref, lrn_kws, goals)

    dres, dmodels, dcoocs = {}, {}, {}
    for nav in nav_types:
        print(nav + '\n')

        model_kws = utils.get_copy(mod_kws)
        co_occ_pars = utils.get_copy(co_pars)
        nav_kws = dict(village=village, stim_pars=mod_par_kws['stim_pars'],
                       conn_pars=mod_par_kws['conn_pars'], nav=nav,
                       max_steps=max_steps, goals=goals)
        trial_kws = dict(step_stats=step_stats, trl_stats=trl_stats,
                         co_occ_pars=co_occ_pars, ibp_pars=ibp_pars)
        trial_kws.update(utils.merge([model_kws, nav_kws, lrn_kws]))
        dres[nav] = run_trial_batch(start_locs, **trial_kws)
        dmodels[nav] = model_kws
        dcoocs[nav] = co_occ_pars['co_occs']

    # Format results.
    res = pd.concat({nav: pd.DataFrame(dres[nav]).T for nav in dres})
    res = res.apply(pd.to_numeric, errors='ignore')
    res = utils.create_long_DF(res, ['nav type', 'trial'])

    # Plot results.
    fdir_dm += fpref + '/'
    ttl = ''  # fpref.replace('__', ', ',).replace('_', ' ')
    plotting.plot_trajs_cooc_habits_by_type(village, res, dmodels, dcoocs,
                                            ttl=ttl, fdir=fdir_dm)
    plotting.plot_rolling_stats_by_type(res, ttl=ttl, fdir=fdir_dm)
    for per_loc in [True, False]:
        plotting.plot_learned_navig_by_type(res, village, per_loc=per_loc,
                                            ttl=ttl, fdir=fdir_dm)

    # Save model and recorded results.
    sim_data = {'village': village, 'res': res, 'dmodels': dmodels,
                'dcoocs': dcoocs, 'mod_kws': mod_kws, 'lrn_kws': lrn_kws,
                'ibp_pars': ibp_pars, 'co_pars': co_pars, 'goals': goals,
                'start_locs': start_locs, 'new_paths': new_paths,
                'del_paths': del_paths, 'new_rwds': new_rwds,
                'n_reset': n_reset, 'max_steps': max_steps}
    utils.write_objects(sim_data, fdir_dm+'sim_data.pickle')

    # Restore environment.
    if restore_vill:
        setup.restore_env(village, new_paths, del_paths, r_orig)

    return sim_data


def run_trial_batch(start_locs, **trial_kws):
    """Run a batch of trials."""

    res = {}
    for i, start_loc in enumerate(start_locs):

        # Report progress.
        process.report_progr(i, len(start_locs), 100)

        # Run trial, record results.
        res[i] = run_trial(start_loc=start_loc, **trial_kws)

    return res


def run_trial(village, start_loc, goals, max_steps, VC, GS, HC, vS, dlS, PFC,
              nav, lrn_conns, lrn_rwds, stim_pars, conn_pars, lrn_pars,
              ibp_pars, step_stats, trl_stats, co_occ_pars, sum_step_res=True):
    """Run a single trial."""

    # Init data objects and model for trial.
    step_data = process.init_data(step_stats)
    trl_data = process.init_data(trl_stats)
    init_trial(village, VC, GS, HC, PFC, start_loc)

    # Collect step results.
    hc_ro = HC.state_read_out()
    process.get_rec_stats(village, None, village.s, None, None, GS, HC, hc_ro,
                          vS, dlS, step_data, co_occ_pars, idx=0)

    # Run trial.
    for istep in range(max_steps):

        # Decide which action to take next.
        u = select_action(village, nav, VC, GS, HC, PFC, vS, dlS, ibp_pars,
                          excl_return=True)

        # Take action, receive stimuli and reward.
        s, r, vmot, vvis, stim = act_sense(village, u, **stim_pars)
        # Update model state.
        hc_ro = model_update(VC, GS, HC, PFC, u, vmot, vvis, **conn_pars)

        # Update connectivity.
        if lrn_conns:
            conn_upd(HC.VC_HC, HC.GS_HC, hc_ro, GS.P, vvis, **lrn_pars)

        # Adjust reward expectations at each step.
        if lrn_rwds:
            vS.update(hc_ro, r)

        # Collect step results.
        process.get_rec_stats(village, u, s, r, stim, GS, HC, hc_ro, vS, dlS,
                              step_data, co_occ_pars, istep+1)

        # At goal state.
        if s in goals:

            # Update habits at reward location (but nowhere else).
            if lrn_rwds:
                habit_update(village, GS, VC, HC, PFC, dlS, r)

            # Terminate trial.
            break

    # Collect trial results.
    process.get_rec_stats(village, u, s, r, stim, GS, HC, hc_ro, vS, dlS,
                          trl_data, co_occ_pars)

    # Process collected step results to return.
    if sum_step_res:
        res = process.summarize_rec_data(step_data)
        res.update(trl_data)
        res.update({'n steps': istep+1, 'start loc': start_loc})
        ret = res
    else:
        ret = (step_data, trl_data) if len(trl_stats) else step_data

    return ret


# %% Functions to perform parts of simulations
# --------------------------------------------

def init_trial(village, VC, GS, HC, PFC, loc):
    """Initialize animal location and activity of brain model."""

    # Reset environment.
    village.replace_animal(loc)
    ovis, vvis = village.observation(VC.alpha, VC.beta)

    # Reset brain model.
    VC.reset()
    GS.reset()
    HC.reset()
    PFC.reset()

    # Starting GS position estimate from current location (noiseless).
    xpos, ypos = village.animal_coords()
    GS.reset_to_pos(xpos, ypos)

    # Sample visual scene.
    VC.update(vvis)

    # Init HC state recognition.
    HC.full_update(VC.v, GS.P)


def select_action(village, nav, VC, GS, HC, PFC, vS, dlS, ibp_pars,
                  excl_return=True):
    """Select an action to take."""

    if nav == 'HAS':

        # Habitual action selection using dlS state - action mapping.
        u = dlS.habit_action(HC.state_read_out(), village.U)

    elif nav == 'IBP':

        # Action selection by imagination-based planning (PFC controlled).
        u = IBP(PFC, VC, GS, HC, vS, village, ibp_pars)

    elif nav == 'random':

        # Sample random action as assesed by model.
        u = PFC.sample_action(VC.features, VC.v, None, village.U)

    elif nav == 'random_env':

        # Sample random action from environment (guaranteed to be executable).
        u = np.random.choice(village.poss_u(excl_return=excl_return))

    elif nav == 'optimal':

        # Select (one of) absolute optimal action(s) from environment.
        u = np.random.choice(village.optimal_u(excl_return=excl_return))

    else:

        # Error case.
        print('Navigation type "{}" not understood'.format(nav))
        u = None

    return u


def act_sense(village, u, mot_sig, vis_alpha, vis_beta):
    """Take action and receive stimuli."""

    # Take action u.
    s, r = village.move_animal(u)

    # Generate input to brain model.
    umot, vmot = village.motor_feedback(u, mot_sig)  # motor feedback
    ovis, vvis = village.observation(vis_alpha, vis_beta)  # visual input
    stim = dict(umot=umot, vmot=vmot, ovis=ovis, vvis=vvis)

    return s, r, vmot, vvis, stim


def model_update(VC, GS, HC, PFC, u, vmot, vvis, vc_hc=1, gs_hc=1, hc_gs=1):
    """Perform one VC - GS - HC update cycle and update PFC's WM."""

    # VC - GS - HC cycle.
    VC.update(vvis)
    GS.motor_update(vmot)          # update GS by noisy motor feedback
    HC.full_update(VC.v, GS.P, vc_hc, gs_hc)  # update HC state estimate
    hc_ro = HC.state_read_out()
    if hc_gs:
        GS.HC_update(HC.GS_HC, hc_ro)  # HC - GS feedback

    # PFC WM update.
    PFC.update_action_history(u)

    return hc_ro


def step_internal_model(village, GS, HC, VC, u, vc_hc_fb, hc_gs_fb):
    """Simulate a single step in internal model."""

    # GS: Simulate motor feedback and roll position estimate forward.
    umot, vmot = village.motor_feedback(u, GS.int_mot_sig)
    GS.motor_update(vmot)

    # HC: Update location estimate by new GS position estimate.
    HC.full_update(None, GS.P)
    hc_ro = HC.state_read_out()

    # VC: Synthesise (imagine) visual scene at estimated location.
    VC.reconstr_scene(HC.VC_HC.T, hc_ro)

    # Optionally feedback VC to HC and HC to GS.
    if vc_hc_fb:
        HC.visual_input(VC.v)
        HC.integrate_inputs()
        hc_ro = HC.state_read_out()
    if hc_gs_fb:
        GS.HC_update(HC.GS_HC, hc_ro)

    return hc_ro


def curious_exploration(GS, HC, village, u_list, mot_sig):
    """Return action determined by curiousity-drivent exploration."""

    # TODO: remove dependence on village and make this part of an internal
    # PFC - MC - GS - HC loop.

    gs_p = GS.P.copy()  # save current GS activity

    # Go through each action, roll GS, infer HC, check uncertainty (entropy).
    s_uncertain = {}
    for u in u_list:
        umot, emot, vmot = village.motor_feedback(u, mot_sig)
        GS.motor_update(umot)
        HC.grid_input(GS.P)
        s_uncertain[u] = utils.entropy(HC.s['GS'])
        GS.P = gs_p  # restore original GS activity

    # Select action leading to maximally uncertain state.
    u_max_uncert = max(s_uncertain, key=s_uncertain.get)

    return u_max_uncert


def store_activity(VC, GS, HC):
    """Store current VC, GS and HC activity."""

    vc_v = VC.v.copy()
    gs_p = GS.P.copy()
    hc_s = utils.get_copy(HC.s)

    a_model = dict(vc_v=vc_v, gs_p=gs_p, hc_s=hc_s)

    return a_model


def restore_activity(VC, GS, HC, vc_v, gs_p, hc_s):
    """Restore VC, GS and HC activity."""

    VC.v = vc_v.copy()
    GS.P = gs_p.copy()
    HC.s = utils.get_copy(hc_s)


# %% Functions related to habit learning.

def habit_update(village, GS, VC, HC, PFC, dlS, r):
    """Update habits."""

    a0_model = store_activity(VC, GS, HC)

    # Replay: iteratively recreate visited states and update habit values.
    for i, u in enumerate(PFC.hist[::-1]):
        u_opp = village.get_opp_dir(u)
        hc_ro = step_internal_model(village, GS, HC, VC, u_opp,
                                    vc_hc_fb=True, hc_gs_fb=True)
        r_disc = dlS.get_discounted_r(r, i)
        dlS.habit_update(hc_ro, village.U.index(u), r_disc)

    restore_activity(VC, GS, HC, **a0_model)


# %% Functions related to Imagination Based Planning (IBP).

# Everything that happens inside these functions is internal to the brain
# model.

def IBP(PFC, VC, GS, HC, vS, village, ibp_pars):
    """Perform a full IBP cycle."""

    # Store activity at current location, reset PFC and get VC-HC.
    a0_model = store_activity(VC, GS, HC)
    PFC.reset_ibp()

    # Sample n_seq potential trajectories, each consisting of n_ahead steps,
    # simulated internal model and accumulated expected reward.
    pars = dict(PFC=PFC, VC=VC, GS=GS, HC=HC, vS=vS, village=village,
                ibp_pars=ibp_pars, a0_model=a0_model)
    for iseq in range(PFC.n_seq):
        ipb_traj(ulist=None, **pars)

    # Also re-simulate current action plan and add to trajectory samples.
    if len(PFC.plan):
        ipb_traj(ulist=PFC.plan, **pars)

    # Select plan and next action.
    PFC.select_plan()
    u = PFC.get_next_action()

    return u


def ipb_traj(PFC, VC, GS, HC, vS, village, ibp_pars, a0_model, ulist=None):
    """Simulate a full IBP trajectory sample (list of actions)."""

    if ulist is None:
        ulist = PFC.n_ahead * [None]

    PFC.new_sample()
    for u in ulist:
        ibp_step(PFC, VC, GS, HC, vS, village, ibp_pars, u)

    restore_activity(VC, GS, HC, **a0_model)


def ibp_step(PFC, VC, GS, HC, vS, village, ibp_pars, u=None):
    """Sample a single IBP step and evaluate result."""

    if u is None:
        # PFC: Transform visual scene to action probs and sample one action.
        u = PFC.sample_action(VC.features, VC.v, village.U_opp, village.U)

    # Store action in WM.
    PFC.store_action_in_wm(u)

    # Roll internal model of environment forward.
    hc_ro = step_internal_model(village, GS, HC, VC, u, **ibp_pars)

    # vS: Evaluate expected reward at location estimate.
    PFC.update_sample_reward(vS.reward_estim(hc_ro))


# %% Functions to learn connectivity
# ----------------------------------

def decay_weights(C, hc_ro_na, w_dec, w_fac=1):
    """Decay connections weights."""

    if w_dec != 0:
        C *= w_fac * (1 - w_dec * (1 - hc_ro_na))
    return C


def oja(VC_HC, GS_HC, hc_ro, gs_p, vvis, lr_gs, lr_vc, norm, w_dec):
    """Update connectivity using Oja's rule."""

    # Oja's rule for small learning rates is equivalent of
    # weight normalization on linear units.
    # dw = lr * y * (x - y*w)

    # The x - y*w term can be interpreted in generative modelling terms as the
    # difference between the input x and its top-down reconstruction y*w, i.e.
    # change weights (of recognized features / states) as long as they can't
    # reconstruct their input.

    # VC -> HC
    hc_ro_na = hc_ro[:, new_ax]
    dVC_HC = hc_ro_na * (vvis - hc_ro_na * VC_HC)
    VC_HC += lr_vc * dVC_HC
    decay_weights(VC_HC, hc_ro_na, w_dec)

    # GS -> HC
    hc_ro_na2 = hc_ro[:, new_ax, new_ax]
    dGS_HC = hc_ro_na2 * (gs_p - hc_ro_na2 * GS_HC)
    GS_HC += lr_gs * dGS_HC
    decay_weights(GS_HC, hc_ro_na2, w_dec)

    return dVC_HC, dGS_HC


def normed_hebb(VC_HC, GS_HC, hc_ro, gs_p, vvis, lr_gs, lr_vc, norm, w_dec):
    """Update connectivity using normalized Hebb rule."""

    # This is currently unable to learn connectivity, should be debugged?

    # Using normalized Hebbian learning.
    # dw = lr * y * x
    # w' = (w + dw) / ||w + dw||_i  , where i can be any norm order

    if w_dec != 0:
        VC_pre_sum = analysis.VC_HC_norm(VC_HC, norm)[:, new_ax]
        GS_pre_sum = analysis.GS_HC_norm(GS_HC, norm)[:, new_ax, new_ax]
    else:
        VC_pre_sum = 1
        GS_pre_sum = 1

    # VC -> HC
    hc_ro_na = hc_ro[:, new_ax]
    upd_VC_HC = VC_HC + lr_vc * hc_ro_na * vvis
    # Normalize (synaptic competition / renormalization / metabolic limit).
    upd_VC_HC /= analysis.VC_HC_norm(upd_VC_HC, norm)[:, new_ax]
    decay_weights(upd_VC_HC, hc_ro_na, w_dec, VC_pre_sum)
    # Save updated connectivity.
    dVC_HC = upd_VC_HC - VC_HC
    VC_HC[:] = upd_VC_HC

    # GS -> HC
    hc_ro_na2 = hc_ro[:, new_ax, new_ax]
    upd_GS_HC = GS_HC + lr_gs * hc_ro_na2 * gs_p
    # Normalize.
    upd_GS_HC /= analysis.GS_HC_norm(upd_GS_HC, norm)[:, new_ax, new_ax]
    decay_weights(upd_GS_HC, hc_ro_na2, w_dec, GS_pre_sum)
    # Save updated connectivity.
    dGS_HC = upd_GS_HC - GS_HC
    GS_HC[:] = upd_GS_HC

    return dVC_HC, dGS_HC


def conn_upd(VC_HC, GS_HC, hc_ro, gs_p, vvis, rule, lr_gs, lr_vc, norm, w_dec):
    """Update connectivity."""

    f = oja if rule == 'oja' else normed_hebb
    return f(VC_HC, GS_HC, hc_ro, gs_p, vvis, lr_gs, lr_vc, norm, w_dec)
