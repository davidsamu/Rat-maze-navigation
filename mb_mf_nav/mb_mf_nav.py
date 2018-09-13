#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Imagination-based planning and model-free decision making.

@author: David Samu
"""



import os
import sys

from itertools import product

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

fproj = '/home/david/Modelling/MB_MF_Nav/'
sys.path.insert(1, fproj)

from mb_mf_nav.brain_model import GridSystem, Hippocampus, VisualCortex
from mb_mf_nav.brain_model import VentralStriatum, DorsolateralStriatum
from mb_mf_nav.brain_model import PrefrontalCortex
from mb_mf_nav import analysis, plotting, process, setup, simul, utils

res_dir = fproj + 'results/'
os.chdir(res_dir)


# %% Set up environment and model.

# Navigation environment.
env_pars = dict(res=100, random=False, nx=10, ny=10, p_state=0.7, p_path=0.9)
village, fdir = setup.init_env(env_pars)

# Stimulus params.
stim_pars = dict(mot_sig=0.2*env_pars['res'],  # mod err: 0.3, low: 0.2
                 vis_alpha=2, vis_beta=5)  # mod err: 4, low: 5

# Model params.
gs_pars = dict(circular=True, activ_sprd=30, init_sprd=5, step_res=5, pad=100)
setup.init_GS_params(env_pars, gs_pars)
hc_pars = dict(gs_hc_sharp=8, ro_pow=5)
st_pars = dict(learn=False, alpha=0.05, gamma=0.9)
pfc_pars = dict(n_seq=5, n_ahead=5, u_soft_tau=.1, gamma=0.9)


# Create model.
GS_HC, VC_HC, GS, VC, HC, vS, dlS, PFC = setup.init_model(village, stim_pars,
                                                          gs_pars, hc_pars,
                                                          st_pars, pfc_pars)
# plotting.plot_model_setup(HC, GS, VC, vS, dlS, village,
#                           hc_pars['gs_hc_sharp'], fdir)


# %% Random navigation.

# Iterate through:
# IBP and HAS
# nrand trials

# Place animal to random location.
# TODO

# Init.
nsteps = 1000
simul.init_trial(village, VC, GS, HC)
tr_data = process.init_trial_data()

# Run animal.
for istep in range(nsteps):

    # Report progress
    process.navig_progr_report(istep, nsteps)

    # Interact with environment.
    u, s, r, vmot, vvis, stim = simul.act_sense(village, 'random', stim_pars)

    # Update model state.
    hc_ro = simul.VC_GS_HC_update(VC, GS, HC, vmot, vvis)

    # Record trial data and model state.
    process.rec_step(village, u, s, r, stim, GS, HC, vS, dlS, istep, tr_data)


# Format recorded data.
ret = process.format_rec_data(tr_data, village, HC, gs_pars, VC.features)
anim_data, mot_data, vis_data, gs_state, hc_state, vs_state, dls_state = ret
fd = process.sim_dir_name(fdir, nsteps, stim_pars, hc_pars, gs_pars, st_pars)

# Do analysis.
res = plotting.plot_navig_res(village, HC, vS, dlS, anim_data, vis_data,
                              mot_data, gs_state, hc_state, vs_state,
                              dls_state, gs_pars, fd)
gs_mpos, gs_H, hc_maxLL, hc_H = res


# %% Random navigation.

# TODO: incorporate old code from below to new code above!

# Simulation params.

# Add lesion to model.
# Turn each connectivity on / off: VC -> HC, GS -> HC, HC -> GS
prd_len = 10000


# Init animal location and brain model activity.
excl_return = False
wrapper.init_trial(village, GS, HC, gs_init_spread, mot_sig)

vS.reset()
dlS.reset()


# Plotting.
pause = 3
do_plotting = False
if do_plotting:
    state_fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    axs = pd.Series(axs.flatten(), index=['village', 'GS_mot', 'HC', 'GS_hc'])
    plt.subplots_adjust(left=.06, bottom=.07, right=.97, top=.97,
                        wspace=0.2, hspace=0.3)
    plotting.plot_village(village, fig=state_fig, ax=axs['village'])
    plotting.plot_step(axs, village, GS.as_DF(), HC, GS, istep=0)


# Recording.
anim_data = {}
mot_data = {}
vis_data = {}
gs_state = {}
hc_state = {}
vs_state = {}
dls_state = {}

# Run animal.
for istep in range(nsteps):

    # Report progress
    if not istep % 500:
        print('{}%'.format(int(100*istep/nsteps)))

    # State of each connectivity (on / off).
    vc_hc, gs_hc, hc_gs = conn_on.loc[istep]

    # Interact with environment.
    u, s, r, stim = wrapper.act_sense(village, 'random', mot_sig,
                                      vis_alpha, vis_beta, excl_return)
    umot, emot, vmot, ovis, vvis = stim

    # Update model state.
    hc_s_ro = wrapper.GS_HC_update(GS, HC, vmot, vvis, hc_ro_pow,
                                   vc_hc, gs_hc, hc_gs)

    # Update reward expectation.
    vS.rwd_update(hc_s_ro, r)

    # Update habitual action selection function.
    # TODO: new dlS model: HC - dlS connectivity: state -> action mapping
    #       learning through reverse replay by dlS (or PFC or HC) rewinding HC
    #       state history, rather than storing it explicitely in dlS
    #
    dlS.update(village.U.index(u), hc_s_ro, r)

    # Record environment and model state and do plot
    # ----------------------------------------------

    # Record trial data and model state.
    x, y = village.animal_coords()
    anim_data[istep] = {'u': u, 's': s, 'r': r, 'x': x, 'y': y}

    mot_data[istep] = {'umot': umot, 'emot': emot, 'vmot': vmot}
    vis_data[istep] = {'ovis': ovis, 'vvis': vvis}

    gs_state[istep] = GS.P.copy()
    hc_state[istep] = deepcopy(HC.s)
    vs_state[istep] = vS.r.copy()
    dls_state[istep] = dlS.Q.copy()

    # Plot animal's location and model state.
    # This is outdated! GS.P is passed after HC update, no prior-HC-update
    # version is kept now...
    if do_plotting:
        plotting.plot_step(axs, village, GS.P, HC, GS, istep+1, pause)


# Format recorded data.
ret = wrapper.format_rec_data(anim_data, mot_data, vis_data, gs_state,
                              hc_state, vs_state, dls_state, VC.features,
                              GS.xvec.copy(), GS.yvec.copy(),
                              HC.s_types, HC.s_names, village.U)
anim_data, mot_data, vis_data, gs_state, hc_state, vs_state, dls_state = ret


# Save recorded data.
sim_data = {'village': village, 'GS': GS, 'HC': HC, 'VC': VC,
            'anim_data': anim_data, 'mot_data': mot_data, 'vis_data': vis_data,
            'gs_state': gs_state, 'hc_state': hc_state, 'vs_state': vs_state,
            'dls_state': dls_state, 'conn_on': conn_on, 'nsteps': nsteps,
            'mot_sig': mot_sig, 'vis_alpha': vis_alpha, 'vis_beta': vis_beta,
            'gs_hc_sharpness': gs_hc_sharp, 'hc_ro_pow': hc_ro_pow}

fn = utils.sim_dir_name(nsteps, gs_hc_sharp, mot_sig, vis_beta, dlS.lambd)
dirname = 'navigation/conn_config/' + fn + '/'
fname = dirname + 'sim_data.pickle'
utils.write_objects(sim_data, fname)


# Do analysis.
gs_mpos, gs_H = analysis.GS_pos_estimate(gs_state, gs_circular)
hc_maxLL = analysis.HC_loc_estimate(hc_state)
hc_H = analysis.HC_entropy(hc_state)

plotting.plot_village(village, vS, dlS, ffig=dirname+'village.png')
plotting.plot_env_sim(anim_data, vis_data, mot_data, dirname)
plotting.plot_GS_HC_sim_res(anim_data, gs_state, hc_state, gs_mpos, gs_H,
                            hc_maxLL, hc_H, conn_on, dirname,
                            window=20, step=10)
plotting.plot_confusion_matrices(anim_data, hc_state, hc_maxLL,
                                 conn_on, dirname)
plotting.plot_reward_learning(village.R, vs_state, dls_state, dirname)
plotting.plot_habits(dlS.as_DF(HC.s_names, village.U), village, dirname)


# %% Location estimate accuracy by noise level.

# Simulation params.
nsteps = 5000  # number of steps taken for each noise level combination
# List of parameter values to sweap through.

# GS - HC connectivity sharpness.
# gs_hc_sharp_list = [1, 1.5, 2, 2.5, 3, 5, 10]
gs_hc_sharp_list = [3]

# HC read-out power (strength of autoassociation / soft-WTA).
# hc_ro_pow_list = [1, 2, 3, 4, 6, 8, 12]
hc_ro_pow_list = [1, 3, 10]

# Visual input noise (higher beta -> lower noise).
vis_beta_list = [2, 2.5, 3, 3.5, 4, 5, 8]
# vis_beta_list = [3, 8]

# Motor noise magnitude.
msfac = [0.02, 0.03, 0.04, 0.05, 0.075, 0.125, 0.25]
# msfac = [0.02, 0.10]
mot_sig_list = [v * np.array([xsize, ysize]) for v in msfac]

par_names = ['gs_hc_sharp', 'hc_ro_pow', 'vis_beta', 'mot_sig']

# Connectivity config.
vc_hc = 1
gs_hc = 1
hc_gs = 1

# Pre-simulation report.
nconfig = (len(gs_hc_sharp_list) * len(hc_ro_pow_list) *
           len(vis_beta_list) * len(mot_sig_list))
print('\nRunning {} configurations\n'.format(nconfig))

anim_data = {}
hc_state = {}

for gs_hc_sharp_i in gs_hc_sharp_list:
    print('gs_hc_sharpness: {}'.format(gs_hc_sharp_i))

    GS_HC = wrapper.init_GS_HC(village.S, village.S_pos, GS.xvec, GS.yvec,
                               gs_hc_sharp_i)
    HC = Hippocampus(village.S, GS_HC, VC_HC)

    for hc_ro_pow_i in hc_ro_pow_list:
        print('\thc_ro_pow: {}'.format(hc_ro_pow_i))

        for vis_beta_i in vis_beta_list:
            print('\t\tvis_beta: {}'.format(vis_beta_i))

            for mot_sig_i in mot_sig_list:

                # Report progress
                print('\t\t\tmmot_sig: {}'.format(mot_sig_i.mean()))

                # Init animal location and brain model activity.
                wrapper.init_trial(village, GS, HC, 'C', gs_init_spread,
                                   mot_sig_i)

                # Run animal.
                for istep in range(nsteps):

                    # Interact with environment.
                    u, s, r, stim = wrapper.act_sense(village, 'random',
                                                      mot_sig_i, vis_alpha,
                                                      vis_beta_i)
                    umot, emot, vmot, ovis, vvis = stim

                    # Update model state.
                    wrapper.GS_HC_update(GS, HC, vmot, vvis, hc_ro_pow_i,
                                         vc_hc, gs_hc, hc_gs)

                    # Record trial data and model state.
                    idx = (gs_hc_sharp_i, hc_ro_pow_i, vis_beta_i,
                           mot_sig_i.mean(), istep)
                    anim_data[idx] = {'u': u, 's': s, 'r': r}
                    hc_state[idx] = deepcopy(HC.s)


# Format recorded data.
ret = wrapper.format_rec_data(anim_data=anim_data, hc_state=hc_state,
                              hc_s_types=HC.s_types, hc_s_names=HC.s_names,
                              idx_pars=par_names)
anim_data, hc_state = ret


# Save recorded data.
sim_data = {'village': village, 'GS': GS, 'HC': HC, 'VC': VC,
            'anim_data': anim_data, 'hc_state': hc_state, 'nsteps': nsteps,
            'mot_sig_list': mot_sig_list, 'vis_alpha': vis_alpha,
            'vis_beta_list': vis_beta_list,  'hc_ro_pow_list': hc_ro_pow_list,
            'gs_hc_sharp_list': gs_hc_sharp_list}


params = [('gshc', len(gs_hc_sharp_list)), ('hcro', len(hc_ro_pow_list)),
          ('vis', len(vis_beta_list)), ('mot', len(mot_sig_list)),
          ('nsteps', nsteps)]
dname = '_'.join([pname+'_'+str(npar) for pname, npar in params])
dname = 'navigation/param_sweep/' + dname + '/'
fname = dname + 'data.pickle'
utils.write_objects(sim_data, fname)


# Do analysis.
hc_maxLL = analysis.HC_loc_estimate(hc_state)

# Determine which param combinations to plot.
pars = pd.Series([list(hc_state.index.get_level_values(idx_name).unique())
                  for idx_name in par_names], index=par_names)
npars = pars.apply(len)
pnames = npars.sort_values().index[:-2]
vcombs = list(product(*[list(pars[p]) for p in pnames]))
select_pars_list = [dict(zip(pnames, vc)) for vc in vcombs]

# Do plotting.
for select_pars in select_pars_list:

    # Select specific parameter values along two swept dimensions.
    keys, vals = zip(*list(select_pars.items()))
    sel_hc_maxLL = hc_maxLL.xs(vals, level=keys).copy()
    sel_anim_data = anim_data.xs(vals, level=keys).copy()

    # Accuracy of HC location estimate.
    hc_corr = sel_hc_maxLL.apply(lambda col: col == sel_anim_data['s'])
    hc_corr = hc_corr.astype(int)

    # Do plotting.
    title = ', '.join([k + ': ' + str(v) for k, v in select_pars.items()])
    fname = '_'.join([k+str(v) for k, v in select_pars.items()])
    fname = dname + utils.format_to_fname(fname) + '.png'
    plotting.plot_navig_param_sweep(hc_corr, 'accuracy', fname, title)


# %% Imagination based planning.


# Visual and HC feedback on / off
# -------------------------------
vc_feedback = True
hc_feedback = True


# Params to sweep through
# -----------------------

# Action selection threshold, setting it to 'max' will select most plausibly
# executable action.
pfc_u_th_list = [0.5, 0.6, 0.7, 0.8, 'max']  # 'max'  # or 0.5
pfc_u_th_list = ['max']

# HC read-out power (strength of autoassociation / soft-WTA).
hc_ro_pow_list = [1, 2, 3, 4, 6, 8, 12]
hc_ro_pow_list = [5]

# Visual scene reconstruction error (higher beta -> lower noise).
vis_beta_list = [3, 4, 5, 6, 8, 10, 100]
# vis_beta_list = [5, 20, 100]

# Motor noise magnitude.
msfac = [0.0, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10]
# msfac = [0.00, 0.05, 0.10]
mot_sig_list = [v * np.array([xsize, ysize]) for v in msfac]

par_names = ['pfc_u_th', 'hc_ro_pow', 'vis_beta', 'mot_sig']

# Simulate trajectory.
n_reset = 100
n_sampl = 1
n_ahead = 5   # TODO: make this precision dependent?

# Variable to prevent back-tracking along trajectory.
opp_dir = {u: village.get_opp_dir(u) for u in village.U}

# Set expected reward of vS to precise value.
# TODO: This should be turned off when simulating learning / habit formation!
vS.r = village.R.copy()

HC_VC = HC.VC_HC.T

# Pre-simulation report.
nconfig = (len(pfc_u_th_list) * len(hc_ro_pow_list) *
           len(vis_beta_list) * len(mot_sig_list))
print('\nRunning {} configurations\n'.format(nconfig))

swp_plans = {}

for pfc_u_th_i in pfc_u_th_list:
    print('pfc_u_th: {}'.format(pfc_u_th_i))

    for hc_ro_pow_i in hc_ro_pow_list:
        print('\thc_ro_pow: {}'.format(hc_ro_pow_i))

        for vis_beta_i in vis_beta_list:
            print('\t\tvis_beta: {}'.format(vis_beta_i))

            VC.beta = vis_beta_i

            for mot_sig_i in mot_sig_list:

                # Report progress
                print('\t\t\tmmot_sig: {}'.format(mot_sig_i.mean()))

                plans = {}

                for loc in village.S:

                    # Init.
                    # print(loc)
                    loc_plans = {}

                    # Resetting at the same location is done to regenerate
                    # visual and motor input, so noise on them varies across
                    # each new reset.
                    for ireset in range(n_reset):

                        # Using 'external' motor and visual noise levels here!

                        # Init animal location and brain model activity.
                        wrapper.init_trial(village, GS, HC, loc,
                                           gs_init_spread, mot_sig)

                        # Init visual input at starting location.
                        ovis, vvis = village.observation(vis_alpha, vis_beta)

                        # Init HC location estimate.
                        HC.full_update(vvis, GS.P)

                        # Save activity of GS and HC at current real location
                        # to restore before each sample.
                        GS_P_start = GS.P.copy()
                        HC_s_start = deepcopy(HC.s)

                        # PFC: acting as controller (scene understanding,
                        # action sampling, policy selection) and working memory
                        # supporting above functions (store decision-related
                        # variables, e.g. action sequences, expected reward)
                        PFC = PrefrontalCortex(n_sampl)

                        # Imagination-based planning by sampling and evaluating
                        # internally generated action sequences.
                        for isamp in range(n_sampl):

                            # Start a new action sequence sample.
                            u_seq = n_ahead * [None]
                            r_exp = 0
                            # Restore GS and HC activity to real current
                            # location/position.
                            GS.P = GS_P_start.copy()
                            HC.s = deepcopy(HC_s_start)

                            excl_list = []
                            for iahead in range(n_ahead):

                                # Step 1
                                # ------

                                # Synthesise (imagine / reconstruct) visual
                                # scene at estimated location.

                                if iahead != 0:
                                    s = HC.state_read_out(pwr=hc_ro_pow_i)
                                    vis_rec = VC.reconstr_scene(s, HC_VC, True)
                                else:               # at starting location
                                    vis_rec = vvis  # observe environment

                                # Optionally feedback VC to HC and HC to GS.
                                if vc_feedback:  # VC -> HC
                                    HC.visual_input(vis_rec)
                                    HC.integrate_inputs()
                                if hc_feedback:  # HC -> GS
                                    hc_ro = HC.state_read_out(pwr=hc_ro_pow_i)
                                    hc_gs_fb = wrapper.HC_GS_feedback(HC.GS_HC,
                                                                      hc_ro)
                                    GS.HC_update(hc_gs_fb)

                                # Step 2
                                # ------

                                # Extract possible actions from (real or
                                # reconstructed) visual scene.

                                # Zero out excluded actions first if maximum is
                                # to be selected.
                                vfeat = VC.features
                                vvis = vis_rec.copy()
                                if pfc_u_th_i == 'max':
                                    excl_idx = [i for i, vf in enumerate(vfeat)
                                                if vf in excl_list]
                                    vvis[excl_idx] = 0
                                u_list = PFC.get_poss_acts(vfeat, vvis,
                                                           pfc_u_th_i)

                                # Step 3
                                # ------

                                # Select an action (currently randomly).

                                # TODO: consider making this reward-directed?
                                u = PFC.sample_action(u_list, excl_list)
                                if u is None:  # no action found
                                    break      # terminate sample
                                u_seq[iahead] = u

                                # Step 4
                                # ------

                                # Simulate motor feedback, roll GS position
                                # estimate forward to new position estimate.

                                # Get motor feedback.
                                mots = village.motor_feedback(u, mot_sig_i)
                                umot, emot, vmot = mots
                                # Roll GS position estimate.
                                GS.motor_update(vmot)

                                # Step 5
                                # ------

                                # Update HC location estimate by new GS
                                # position estimate.

                                HC.full_update(vvis, GS.P, vc_hc=0, gs_hc=1)

                                # Step 6
                                # ------

                                # Evaluate expected reward at location
                                # estimate.

                                r_exp = 0
                                # hc_s_ro = HC.state_read_out(pwr=hc_ro_pow)
                                # r_exp += hc_s_ro.dot(vS.r)

                                # Prevent opposite of previous action to force
                                # exploration of environment.
                                excl_list = [opp_dir[u]]

                            # Has path been sampled yet?
                            # Consider taking another sample if so!
                            # seq_repd = PFC.WM['u_seq'].apply(lambda us: us == u_seq).any()

                            # Store sample results in PFC's WM.
                            PFC.u_seq.append(u_seq)
                            PFC.r_exp.append(r_exp)

                        # Save PFC WM for analysis.
                        loc_plans[ireset] = deepcopy(PFC.u_seq)

                    # Save results from location.
                    plans[loc] = loc_plans

                # Format plans to DataFrame.
                plans = pd.DataFrame(plans).stack()
                idx = plans.index.rename(['reset', 'loc'])
                plans = pd.DataFrame([kv for kv in plans], index=idx).stack()
                idx = plans.index.rename(['reset', 'loc', 'sample'])
                plans = pd.DataFrame([kv for kv in plans], index=idx)
                plans.columns.name = 'step'

                # Record results.
                th = str(pfc_u_th_i) if 'max' in pfc_u_th_list else pfc_u_th_i
                idx = (th, hc_ro_pow_i, vis_beta_i, mot_sig_i.mean())
                swp_plans[idx] = plans

# Format recorded results.
swp_plans = pd.concat(swp_plans)
swp_plans.index.set_names(par_names, level=range(len(par_names)), inplace=True)

# Check whether they are complete and valid.
f = analysis.is_full_valid_action_seq
loc_idx = np.where(pd.Series(swp_plans.index.names).str.match('loc'))[0][0]
is_valid = swp_plans.apply(lambda x: f(village, x.name[loc_idx], x), axis=1)
is_valid.name = 'is_valid'

# Save recorded data.
sim_data = {'village': village, 'GS': GS, 'HC': HC, 'VC': VC,
            'n_reset': n_reset, 'n_sampl': n_sampl, 'n_ahead': n_ahead,
            'mot_sig_list': mot_sig_list, 'vis_alpha': vis_alpha,
            'vis_beta_list': vis_beta_list,  'hc_ro_pow_list': hc_ro_pow_list,
            'pfc_u_th_list': pfc_u_th_list, 'swp_plans': swp_plans,
            'is_valid': is_valid, 'vc_feedback': vc_feedback,
            'hc_feedback': hc_feedback}

params = [('pfcth', len(pfc_u_th_list)), ('hcro', len(hc_ro_pow_list)),
          ('vis', len(vis_beta_list)), ('mot', len(mot_sig_list)),
          ('nahead', n_ahead), ('n_reset', n_reset),
          ('vcfb', int(vc_feedback)), ('hcfb', int(hc_feedback))]
dname = '_'.join([pname+'_'+str(npar) for pname, npar in params])
dname = 'IBN/param_sweep/' + dname + '/'
fname = dname + 'data.pickle'
utils.write_objects(sim_data, fname)


# Determine which param combinations to plot.
pars = pd.Series([list(is_valid.index.get_level_values(idx_name).unique())
                  for idx_name in par_names], index=par_names)
npars = pars.apply(len)
pnames = npars.sort_values().index[:-2]
vcombs = list(product(*[list(pars[p]) for p in pnames]))
select_pars_list = [dict(zip(pnames, vc)) for vc in vcombs]

# Do plotting.
for select_pars in select_pars_list:

    # Select specific parameter values along two swept dimensions.
    keys, vals = zip(*list(select_pars.items()))
    sel_is_valid = pd.DataFrame(is_valid.xs(vals, level=keys).copy())

    title = ', '.join([lbl + ' ' + ('on' if on else 'off')
                       for lbl, on in [('VC-HC', vc_feedback),
                                       ('HC-GS', hc_feedback)]])
    sel_is_valid.columns = [title]

    # title = ', '.join([k + ': ' + str(v) for k, v in select_pars.items()])
    suptitle = ''
    fname = '_'.join([k+str(v) for k, v in select_pars.items()])
    fname = dname + utils.format_to_fname(fname) + '.png'
    plotting.plot_navig_param_sweep(sel_is_valid, '', fname, suptitle)


# Function of preplay (Foster 2017):
#  - sampling based trajectory evaluation for deliberate choice, OR
#  - parsing fixed action sequences to reduce dimensionality of nagivation
#    problem ("options"), akin to hierarchical decision making?


# %%LEARN GM: POSITION - OBSERVATION ASSOCIATION

# TODO: improve learning by normalizing pre- and/or post-synaptic weights...

# Village: lr=0.25, w_dec=0.003, hc_ro_pow=5
# Rnd 10x10, p_state 0.7, p_path 0.9: lr=0.4, w_dec=0.0006, hc_ro_pow=5

# Learning params.
lr_vc = 0.45  # VC - HC learning rate
lr_gs = 0.45  # GS - HC learning rate

rule = 'oja'   # learning rule: normalized Hebb ('hebb') or Oja's rule ('oja')
norm = 2        # degree of norm to use to init and update connectivities

w_dec = 0.0005   # weight decay factor: w_ij = (1 - w_dec * (1-hc_ro_j)) * w_ij
lr_dec = 0.0   # learning rate decay power, set to 0 to turn off decay
ro_dec = 0   # HC read-out power decay, set to 0 to turn off decay

# Random or curiousity-driven exploration?
exploration = 'random'
# exploration = 'curious'

nsteps = 30000


# Init brain model with random connectivity
# -----------------------------------------

# HC
hc_ro_pow = 5  # 'ML' for maximum likelihood
hc_ns_fac = 5
ns = int(hc_ns_fac * len(village.S))  # number of HC units (candidate states)

# GS
gs_activ_spread = 30
gs_circular = True


# Params of initial VC - HC and GS - HC connectivity.
# Use .5 & 5 and 1 & 10 to get a skewed distributions with the majority of
# values near zero and a few values being high, to perhaps facilitite learning?
vc_hc_a, vc_hc_b = 2, 2
gs_hc_a, gs_hc_b = 2, 2

# Turn connectivity on/off.
vc_hc = 1
gs_hc = 1
hc_gs = 1

vfeats = village.vfeatures

# GS
GS = GridSystem(gs_xmin, gs_xmax, gs_nx, gs_ymin, gs_ymax, gs_ny,
                gs_circular, gs_activ_spread)
gs_xsize, gs_ysize = GS.xmax-GS.xmin, GS.ymax-GS.ymin

# Init random "states" (repertiore of HC place cell assemblies).
S = ['s' + str(i) for i in range(ns)]

# Set input noise level.
vis_alpha, vis_beta = 2, 4  # visual noise params
mot_sig = 0.2 * np.array([env_res, env_res])  # motor noise STD
gs_init_spread = 5  # initial spread in GS

# Init connectivities randomly.
VC_HC = wrapper.init_rand_VC_HC(S, len(vfeats), vc_hc_a, vc_hc_b, norm)
GS_HC = wrapper.init_rand_GS_HC(S, GS.nx, GS.ny, gs_hc_a, gs_hc_b, norm)

# To check similarity with optimal connectivity matrices.
optGS_HC = wrapper.init_GS_HC(village.S, village.S_pos, GS.xvec, GS.yvec,
                              gs_hc_sharp)
optVC_HC = wrapper.init_VC_HC(village.V, village.S)

# HC: state representation
HC = Hippocampus(S, GS_HC, VC_HC)
hc_ro_pow_i = hc_ro_pow

# VC: observations
VC = VisualCortex(vfeats, vis_alpha, vis_beta)

# Init animal location and brain model activity.
wrapper.init_trial(village, GS, HC, gs_init_spread, mot_sig)

# For world - HC state co-occurance ratio calculation.
n_co_occ = 1000
co_occs = np.zeros((len(S), len(village.S)))
i_co = -1
coocc_list = n_co_occ * [[]]

res = {}
save_every = 10

# Model init report.
print('\nExploration type: {}, learning rule: {}'.format(exploration, rule))
print(('Units: GS: {}x{} ({})'.format(GS.nx, GS.ny, GS.nx*GS.ny) +
       ', HC: {}x{} ({})\n'.format(hc_ns_fac, len(village.S), ns)))

# Run animal.
for istep in range(nsteps):

    # Report progress
    if not istep % 250:
        s_real_h, s_hc_h = ([res[istep-1][v].mean() for v in ('s_real_h',
                                                              's_hc_h')]
                            if istep > 0 else [0, 0])
        ro_pow = hc_ro_pow_i if ro_dec != 0 else None
        utils.report_learning_progress(istep, nsteps, HC.VC_HC, HC.GS_HC, norm,
                                       s_real_h, s_hc_h, ro_pow)

    # Interact with environment.
    # Select random action from current state, move animal and receive reward.
    # Turning back is disabled! In order to make exploration more effective and
    # TD-learning more efficient...
    if exploration == 'curious':
        u_list = village.poss_u(excl_return=False)
        u_type = wrapper.curious_exploration(GS, HC, village, u_list, mot_sig)
    else:
        u_type = 'random'

    u, s, r, stim = wrapper.act_sense(village, u_type, mot_sig,
                                      vis_alpha, vis_beta)
    umot, emot, vmot, ovis, vvis = stim

    # Update model state.
    if hc_ro_pow != 'ML' and ro_dec > 0:
        hc_ro_pow_i = 4 + (hc_ro_pow-4) * ((nsteps-istep) / nsteps) ** ro_dec
    hc_ro = wrapper.GS_HC_update(GS, HC, vmot, vvis, hc_ro_pow_i,
                                 vc_hc, gs_hc, hc_gs)

    # Update connectivity.
    lr_fac = ((nsteps-istep) / nsteps) ** lr_dec
    lr_gs_i = lr_fac * lr_gs
    lr_vc_i = lr_fac * lr_vc
    dVC_HC, dGS_HC = wrapper.conn_upd(HC.VC_HC, HC.GS_HC, hc_ro, GS.P, vvis,
                                      rule, lr_gs_i, lr_vc_i, norm, w_dec)


    # Update co-occurance matrix.
    i_s_real = village.S.index(s)
    i_co = wrapper.update_co_occ_mat(co_occs, hc_ro, i_s_real, coocc_list,
                                     i_co, n_co_occ, istep > n_co_occ)

    # Record data.
    if (istep+1) % save_every == 0:
        res[istep] = wrapper.record_learning(village, s, GS, HC, hc_ro,
                                             dVC_HC/lr_fac, dGS_HC/lr_fac,
                                             co_occs, norm)


# Final report.
i_max = max(res.keys())
s_real_h, s_hc_h = [res[i_max][v].mean() for v in ('s_real_h', 's_hc_h')]
ro_pow = hc_ro_pow_i if ro_dec != 0 else None
utils.report_learning_progress(istep+1, nsteps, HC.VC_HC, HC.GS_HC, norm,
                               s_real_h, s_hc_h, ro_pow)

# Format recorded results.
r = wrapper.proc_learning(res, village.S, S, HC.GS_HC)
res, s_real_h, s_hc_h, vc_hc_snr, gs_hc_pos, gs_hc_h, gs_hc_max = r
s_iord, s_name_srtd = analysis.get_s_order(co_occs, HC.s_names)


# Plot result.
fdir = 'learning/' + fdir_temp

plotting.plot_village(village, ffig=fdir+'env.png')
plotting.plot_learning_stats(res, s_real_h, s_hc_h, gs_hc_pos, vc_hc_snr,
                             gs_circular, gs_xsize, gs_ysize, fdir)
plotting.plot_gs_hc_mean_traj(gs_hc_pos, gs_hc_h, gs_hc_max, GS, 100, True,
                              len(village.S), fdir)
plotting.plot_learned_vc_hc(optGS_HC, optVC_HC, VC_HC, VC.features, village.S,
                            HC.s_names, co_occs, s_name_srtd, fdir)
# plotting.plot_GS_HC_conn(HC, GS.xvec, GS.yvec, s_iord,
#                          ffig=fdir+'gs_hc_learned.png')


# %% Parameter sweep analysis of learning process.

# TODO: this section needs updating with modified code from above section!

# Learning params.
lr_vc_list = [0.001, 0.003, 0.01, 0.03]  # VC - HC
lr_gs_list = [0.01, 0.03, 0.1, 0.3]  # GS - HC
rule = 'oja'  # learning rule: normalized Hebb ('hebb') or Oja's rule ('oja')


# Init brain model with random connectivity
# -----------------------------------------

# Params.
vc_hc_a, vc_hc_b = 1000, 1000  # params of initial VC - HC connectivity
gs_hc_a, gs_hc_b = 2, 2  # params of initial GS - HC connectivity
ns = len(village.S)  # number of units in HC (candidate states)
n_hc_s_list = [int(1*ns), int(2*ns), int(4*ns)]

hc_ro_pow_list = [3, 10, 30]

n_reset = 20
nsteps = 10000
n_co_occ = 1000


# Turn connectivity on/off.
vc_hc = 1
gs_hc = 1
hc_gs = 1


par_names = ['lr_vc', 'lr_gs', 'n_hc_s', 'hc_ro_pow']

# Pre-simulation report.
nconfig = (len(lr_vc_list) * len(lr_gs_list) *
           len(n_hc_s_list) * len(hc_ro_pow_list))
print('\nRunning {} configurations\n'.format(nconfig))

real_h = {}
hc_h = {}

for lr_vc_i in lr_vc_list:
    print('lr_vc: {}'.format(lr_vc_i))

    for lr_gs_i in lr_gs_list:
        print('\tlr_gs: {}'.format(lr_gs_i))

        for n_hc_s_i in n_hc_s_list:
            print('\t\tn_hc_s: {}'.format(n_hc_s_i))

            for hc_ro_pow_i in hc_ro_pow_list:
                print('\t\t\thc_ro_pow: {}'.format(hc_ro_pow_i))

                for i_reset in range(n_reset):

                    # GS
                    GS = GridSystem(gs_xmin, gs_xmax, gs_nx,
                                    gs_ymin, gs_ymax, gs_ny)

                    # Init random "states" (repertiore of HC place cell
                    # assemblies).
                    S = ['s' + str(i) for i in range(n_hc_s_i)]

                    # Set noise level.
                    vis_alpha, vis_beta = 2, 8  # visual noise params
                    mot_sig = 0.02 * np.array([env_res, env_res])  # motor noise
                    gs_init_spread = 1  # initial spread in GS

                    # Init connectivities randomly.
                    VC_HC = wrapper.init_rand_VC_HC(S, len(village.vfeatures),
                                                    vc_hc_a, vc_hc_b)
                    GS_HC = wrapper.init_rand_GS_HC(S, GS.nx, GS.ny,
                                                    gs_hc_a, gs_hc_b)

                    # HC: state representation
                    HC = Hippocampus(S, GS_HC, VC_HC)

                    # VC: observations
                    VC = VisualCortex(village.vfeatures, vis_alpha, vis_beta)

                    # Init animal location and brain model activity.
                    wrapper.init_trial(village, GS, HC, 'C',
                                       gs_init_spread, mot_sig)

                    # Run animal.
                    s_hc_ml = {}
                    for istep in range(nsteps):

                        # Interact with environment..
                        u, s, r, stim = wrapper.act_sense(village, 'random',
                                                          mot_sig, vis_alpha,
                                                          vis_beta)
                        umot, emot, vmot, ovis, vvis = stim

                        # Update model state.
                        hc_ro = wrapper.GS_HC_update(GS, HC, vmot, vvis,
                                                     hc_ro_pow_i, vc_hc,
                                                     gs_hc, hc_gs)

                        # Update connectivity.
                        dVC_HC, dGS_HC = wrapper.conn_upd(rule, HC.VC_HC,
                                                          HC.GS_HC, hc_ro,
                                                          GS.P, vvis,
                                                          lr_gs_i, lr_vc_i)

                        # Record results.
                        if nsteps - istep <= n_co_occ:
                            s_pair = (hc_ro.argmax(), village.S.index(s))
                            if s_pair not in s_hc_ml:
                                s_hc_ml[s_pair] = 0
                            s_hc_ml[s_pair] += 1

                    # Collect results.
                    co_occs = wrapper.get_co_occ_mat(s_hc_ml, len(village.S),
                                                     len(HC.s_names))
                    s_real_h, s_hc_h = analysis.get_co_occ_entropy(co_occs)
                    idx = (lr_vc_i, lr_gs_i, n_hc_s_i, hc_ro_pow_i, i_reset)
                    real_h[idx] = s_real_h.mean()
                    hc_h[idx] = s_hc_h.mean()


# Format recorded data.
real_h = pd.Series(real_h)
hc_h = pd.Series(hc_h)
for h_res in [real_h, hc_h]:
    h_res.index.names = par_names + ['n_reset']

# Save results.
sim_data = {'real_h': real_h, 'hc_h': hc_h}

params = [('lrvc', len(lr_vc_list)), ('lrgs', len(lr_gs_list)),
          ('hcro', len(hc_ro_pow_list)), ('nreset', n_reset),
          ('nsteps', nsteps)]
dname = '_'.join([pname+'_'+str(npar) for pname, npar in params])
dname = 'learning/param_sweep/' + dname + '/'
fname = dname + 'data.pickle'
utils.write_objects(sim_data, fname)


# Determine which param combinations to plot.
pars = pd.Series([list(real_h.index.get_level_values(idx_name).unique())
                  for idx_name in par_names], index=par_names)
npars = pars.apply(len).sort_values()
split_by = npars.index[-3]
pnames = npars.sort_values().index[:-3]
vcombs = list(product(*[list(pars[p]) for p in pnames]))
select_pars_list = [dict(zip(pnames, vc)) for vc in vcombs]

# Do plotting.
for select_pars in select_pars_list:

    # Select specific parameter values along two swept dimensions.
    keys, vals = zip(*list(select_pars.items()))
    sel_real_h = real_h.xs(vals, level=keys).copy()
    sel_hc_h = hc_h.xs(vals, level=keys).copy()

    # Do plotting.
    title = ', '.join([k + ': ' + str(v) for k, v in select_pars.items()])
    fname = '_'.join([k+str(v) for k, v in select_pars.items()])
    fname = dname + utils.format_to_fname(fname) + '.png'
    plotting.plot_learning_param_sweep(sel_real_h, sel_hc_h, split_by,
                                       fname, title)
