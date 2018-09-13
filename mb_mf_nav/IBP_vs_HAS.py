#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Navigation by Imagination-based planning and model-free decision making.

@author: David Samu
"""


import os
import sys

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
env_pars = dict(res=100, random=False, nx=6, ny=6, p_state=0.7, p_path=0.9)
village, fdir = setup.init_env(env_pars)

# Stimulus params.
stim_pars = dict(mot_sig=0.2*env_pars['res'],  # mod err: 0.3, low: 0.2
                 vis_alpha=2, vis_beta=5)  # mod err: 4, low: 5

# Model params.
gs_pars = dict(circular=True, step_res=5, pad=100, activ_sprd=30)
setup.init_GS_params(env_pars, gs_pars)
hc_pars = dict(gs_hc_sharp=8, ro_pow=5, conns_set=True)
st_pars = dict(learned=True, alpha=0.5, gamma=0.6, tau=0.1)
pfc_pars = dict(n_seq=5, n_ahead=village.diam, u_soft_tau=0.3, gamma=0.9)

# Connnectivity params (init, learning and being on/off).
lrn_pars = dict(rule='oja', norm=2, w_dec=0, lr_gs=0.25, lr_vc=0.25)
conn_pars = dict(vc_hc=1, gs_hc=1, hc_gs=1)

# IBP params.
ibp_pars = dict(vc_hc_fb=True, hc_gs_fb=True)

# Params meta dict.
mod_par_kws = dict(stim_pars=stim_pars, conn_pars=conn_pars, gs_pars=gs_pars,
                   hc_pars=hc_pars, st_pars=st_pars, pfc_pars=pfc_pars)

fdir_dm = fdir + 'IBP_vs_HAS/'


# For plotting setup: village and model.
if False:
    GS, VC, HC, vS, dlS, PFC = setup.init_model(village, **mod_par_kws,
                                                as_dict=False)
    plotting.plot_model_setup(HC, GS, VC, vS, dlS, village,
                              hc_pars['gs_hc_sharp'], fdir)


# %% Navigation in learned and altered environment
# ------------------------------------------------

# IBP: Imagination-based planning by sampling and evaluating internally
# generated action sequences.

# HAS: Habitual action selection, using state - action mappings of dlS.

# Simulation params.
n_reset = 20
new_paths = [('CN', 'water'), ('CN', 'toys'),
             ('CE', 'toys'), ('CE', 'food'),
             ('CS', 'food'), ('CS', 'peer'),
             ('CW', 'peer'), ('CW', 'water')]
# new_paths = [('CW', 'water'), ('CS', 'food')]
del_paths = [('CS', 'S'), ('CE', 'E')]  # this messes things up somehow...
del_paths = [('S', 'food'), ('E', 'food')]
# del_paths = [('S', 'food')]
new_rwds = dict(food=0, water=5)

# Which stats to collect after each step and each trial?
step_stats = ['hc_ro', 'co_occs']
trl_stats = ['gs_hc_pos', 'gs_hc_max']

ibp_has_kws = dict(village=village, mod_par_kws=mod_par_kws,
                   step_stats=step_stats, trl_stats=trl_stats,
                   ibp_pars=ibp_pars, lrn_pars=lrn_pars, n_reset=n_reset,
                   fdir_dm=fdir_dm)


# Learned environment and rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws)


# Learned environment but revalued rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws, new_rwds=new_rwds)


# Modified environment (shortcuts added), learned rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws, new_paths=new_paths)


# Modified environment (obstacles added), learned rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws, del_paths=del_paths)


# Modified environment (shortcuts and obstacles added), learned rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws, new_paths=new_paths,
                           del_paths=del_paths)


# Modified environment (shortcuts added), revalued rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws, new_paths=new_paths,
                           new_rwds=new_rwds)


# Modified environment (obstacles added), revalued rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws, del_paths=del_paths,
                           new_rwds=new_rwds)


# Modified environment (shortcuts and obstacles added), revalued rewards.
all_res = simul.IBP_vs_HAS(**ibp_has_kws, new_paths=new_paths,
                           del_paths=del_paths, new_rwds=new_rwds)


# %% Unknown environment and rewards.

# TODO : sort out file naming + report for full conn + rwd learning!

nav_types = ['optimal', 'HAS', 'IBP', 'random']
nav_types = ['random_env']
all_res = simul.IBP_vs_HAS(**ibp_has_kws, nav_types=nav_types)


# %% To test single trial run.

lrn_conns = not hc_pars['conns_set']
lrn_rwds = not st_pars['learned']
max_steps = 100

# Params to run a simulation.
start_loc = 'C'
goals = [village.max_r_state()]
nav = 'random'
GS, VC, HC, vS, dlS, PFC = setup.init_model(village, **mod_par_kws,
                                            as_dict=False)

step_stats = ['hc_ro', 'co_occs']
trl_stats = ['gs_hc_pos', 'gs_hc_max']

sum_step_res = False
co_occ_pars = setup.init_co_occ_pars(len(HC.s_names), len(village.S),
                                     nsteps=500)

# Single trial.
res = simul.run_trial(village, start_loc, goals, max_steps, VC, GS, HC, vS,
                      dlS, PFC, nav, lrn_conns, lrn_rwds, stim_pars, conn_pars,
                      lrn_pars, ibp_pars, step_stats, trl_stats, co_occ_pars,
                      sum_step_res)

# Bunch of trials.
n_reset = 100
goals, start_locs = setup.init_locs(village, n_reset)
for i, start_loc in enumerate(start_locs):
    process.report_progr(i, len(start_locs), 10)
    res = simul.run_trial(village, start_loc, goals, max_steps, VC, GS, HC, vS,
                          dlS, PFC, nav, lrn_conns, lrn_rwds, stim_pars,
                          conn_pars, lrn_pars, ibp_pars, step_stats, trl_stats,
                          co_occ_pars, sum_step_res)

plotting.plot_village(village, vS, dlS, ffig=None)


# %% To test batch trial run.

# Additional params for batch run.
n_reset = 50
goals, start_locs = setup.init_locs(village, n_reset)
starts = None
new_paths = None
del_paths = None
new_rwds = None
nav_types = None

# Batch of trials.
res = simul.run_trial_batch(start_locs, village=village, goals=goals,
                            max_steps=max_steps, VC=VC, GS=GS, HC=HC, vS=vS,
                            dlS=dlS, PFC=PFC, nav=nav, lrn_conns=lrn_conns,
                            lrn_rwds=lrn_rwds, stim_pars=stim_pars,
                            lrn_pars=lrn_pars, vc_hc_fb=vc_hc_fb,
                            hc_gs_fb=hc_gs_fb, step_stats=step_stats,
                            trl_stats=trl_stats, co_occ_pars=co_occ_pars)


# Extract entropy of imagined HC states, plot them as function of look-ahead
# step.
ibp_hc_h = np.concatenate([res[i]['ibp_hc_h'] for i in res])
mi = pd.MultiIndex.from_product([range(i) for i in ibp_hc_h.shape],
                                names=['step', 'sample', 'ahead'])
ibp_hc_h = pd.Series(ibp_hc_h.flatten(), index=mi, name='H HC')
ibp_hc_h = utils.create_long_DF(ibp_hc_h)
ax = sns.pointplot(x='ahead', y='H HC', data=ibp_hc_h)
# ax.set_ylim([0, None])


# %% Obstacle avoidance
# ---------------------


# Check HC surprise
#  - before vs after adding shortcuts
#  - changed vs unchanged locations

#   - is there a jump in uncertainty? that would make automatic switch between
#     systems possible...


# Surprise condition criteria (triggering switch from HAS to IBP):
#   - habit dictating an impossible action
#         + state mis-recognized, habit mis-learned
#   - highly uncertain HC state recognition
#         + location representation not yet well learned, highly noisy input,
#           change in environment
#   - reward expectation error
#         + vS reward estimate not well learned, location's reward has changed

# How to make forward switch from IBP to HAS?
# - track uncertainty in HAS? E.g. if during replay at reward state reached by
#   IBP, HAS would have selected the same trajectory?
# - one action dominates all other, i.e. high certainty in action maximizing
#   value. This could be problematic early in habit learning...


# TODO: new dlS model: HC - dlS connectivity: state -> action mapping
    #       learning through reverse replay by dlS (or PFC or HC) rewinding HC
    #       state history, rather than storing it explicitely in dlS
    #

# %% For Barccsyn18 poster.

# Figure 1: Village and model state example
# -----------------------------------------

village, fdir = setup.init_env(env_pars)
GS, VC, HC, vS, dlS, PFC = setup.init_model(village, **mod_par_kws,
                                            as_dict=False)
simul.init_trial(village, VC, GS, HC, dlS, 'food')

state_fig, axs = plt.subplots(1, 3, figsize=(10, 2.8))
axs = pd.Series(axs.flatten(), index=['village', 'GS', 'HC'])
plt.subplots_adjust(left=.0, bottom=.0, right=1, top=1,
                    wspace=0.3, hspace=0.4)
plotting.plot_village(village, add_loc_r=False, add_title=False,
                      # xlim=[GS.xmin, GS.xmax], ylim=[GS.ymin, GS.ymax],
                      add_animal=True, fig=state_fig, ax=axs['village'])
plotting.plot_step(axs, village, GS, HC, add_ttl=False)
axs['village'].set_title('Navigation environment')
[ax.set_title(ax.get_title(), pad=10) for ax in axs]

plotting.save_fig('Barccsyn18/model_state.png', state_fig, tight_layout=False)


# Figure 2: Random navigation by different connectivity
# -----------------------------------------------------

# Params
start_loc = 'C'
goals = []
max_steps = 1000
nav = 'random_env'
lrn_conns, lrn_rwds = False, False
vc_hc_fb, hc_gs_fb = False, False  # during IBP only!

step_stats = ['anim_data', 'stim_data', 'hc_state']
trl_stats = []
co_occ_pars = []

# Model
GS, VC, HC, vS, dlS, PFC = setup.init_model(village, **mod_par_kws,
                                            as_dict=False)
simul.init_trial(village, VC, GS, HC, dlS, start_loc)

cc_df = setup.get_conn_on_df(max_steps, full_only=False)
cc_confs = cc_df.drop_duplicates()

anim_data, hc_state = {}, {}
for cc in cc_confs.index:
    conn_pars = {k.replace('-', '_').lower(): v
                 for k, v in cc_confs.loc[cc].items()}
    res = simul.run_trial(village, start_loc, goals, max_steps, VC, GS, HC,
                          vS, dlS, PFC, nav, lrn_conns, lrn_rwds, stim_pars,
                          conn_pars, lrn_pars, ibp_pars, step_stats, trl_stats,
                          co_occ_pars, sum_step_res=False)
    res = process.format_rec_data(res, village, HC, gs_pars, VC.features)
    anim_data[cc] = res[0]
    hc_state[cc] = res[3]
    start_loc = village.s

anim_data = pd.concat(anim_data)
anim_data.reset_index(drop=True, inplace=True)

hc_state = pd.concat(hc_state)
mi = [(int(i/len(HC.s_names)), loc)
      for i, loc in enumerate(hc_state.index.get_level_values(2))]
hc_state.index = pd.MultiIndex.from_tuples(mi, names=['trial', 'loc'])

hc_maxLL = analysis.HC_loc_estimate(hc_state)


plotting.plot_GS_HC_sim_res(anim_data, hc_state, None, None,
                            hc_maxLL, None, 'Barccsyn18/', cc_df,
                            window=20, step=10)


# Figure 3: Village with rewards and values
# -----------------------------------------

plotting.plot_village(village, vS, dlS, add_title=False, figsize=(5, 5),
                      ffig='Barccsyn18/village.png')


# Figure 4: Village with rewards and values, after revaluing
# ----------------------------------------------------------

all_res = simul.IBP_vs_HAS(**ibp_has_kws, new_rwds=new_rwds)
vS, dlS = [all_res['dmodels']['IBP'][reg] for reg in ['vS', 'dlS']]
village_sim = all_res['village']
plotting.plot_village(village_sim, vS, dlS, add_title=False, figsize=(5, 5),
                      ffig='Barccsyn18/village_revalued_IBP.png')


# Figure 5: Village with rewards and values, after adding shortcuts + obstacles
# -----------------------------------------------------------------------------

all_res = simul.IBP_vs_HAS(**ibp_has_kws, new_paths=new_paths,
                           del_paths=del_paths, restore_vill=False)
vS, dlS = [all_res['dmodels']['IBP'][reg] for reg in ['vS', 'dlS']]
village_sim = all_res['village']
plotting.plot_village(village_sim, vS, dlS, add_title=False, figsize=(5, 5),
                      ffig='Barccsyn18/village_shortcuts_obstacles_IBP.png')

