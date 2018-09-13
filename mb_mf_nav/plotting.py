#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scrip to collect plotting functions.

@author: David Samu
"""

from itertools import product

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from mb_mf_nav import analysis, process, setup, utils


# %% Generic plotting functions.

def init_ax_fig(ax=None, fig=None, **fig_kws):
    """Init axis and figure."""

    if ax is None:
        fig, ax = plt.subplots(1, 1, **fig_kws)
    return fig, ax


def refresh(pause=1e-6):
    """Refresh figure."""

    plt.draw()
    plt.pause(pause)


def hide_axis_labels(ax, hidex=True, hidey=True):
    """Hide axis labels."""

    if hidex:
        ax.set_xlabel('')
    if hidey:
        ax.set_ylabel('')


def hide_tick_marks(ax, axis='both'):
    """Hide tick marks on axes."""
    ax.tick_params(axis=axis, which='both', length=0)


def hide_spine(ax, side):
    """Hide spine on given side (bottom, left, top or right)."""
    ax.spines[side].set_visible(False)


def hide_top_right_spines(ax):
    """Hide spines on top and right sides of axis."""
    hide_spine(ax, 'top')
    hide_spine(ax, 'right')


def rot_xtick_labels(ax, rot=45, ha='right'):
    """Rotate labels on x axis."""
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rot, ha=ha)


def rot_ytick_labels(ax=None, rot=45, va='top'):
    """Rotate labels on y axis."""
    ax.set_yticklabels(ax.get_yticklabels(), rotation=rot, va=va)


def hide_tick_labels(axis):
    """Hide tick labels on axis."""

    for lbl in axis.get_ticklabels():
        lbl.set_visible(False)


def keep_nth_tick_label(axis, keep_every=2):
    """Keep only every nth tick label on axis."""

    for i, lbl in enumerate(axis.get_ticklabels()):
        if i % keep_every != 0:
            lbl.set_visible(False)


def save_fig(ffig, fig=None, dpi=300, close=True, tight_layout=True):
    """Save composite (GridSpec) figure to file."""

    # Init figure and folder to save figure into.
    utils.create_dir(ffig)

    if fig is None:
        fig = plt.gcf()

    if tight_layout:
        fig.tight_layout()

    fig.savefig(ffig, dpi=dpi, bbox_inches='tight')

    if close:
        plt.close(fig)


def add_conn_config_markers(cconfs, ax, color='grey', alpha=1, lw=1, ls='--'):
    """Add connectivity configuration markers to axes."""

    ymin, ymax = ax.get_ylim()
    lbl_yloc = ymin + 1.02 * (ymax - ymin)  # used to be 0.98
    for cconf, istep in cconfs.items():
        ax.axvline(istep, color=color, alpha=alpha, lw=lw, ls=ls)
        ax.text(istep, lbl_yloc, cconf, fontsize='small', va='top', ha='left')


# %% Connectivities.

def plot_VC_HC_conn(VC_HC, s_names=None, vfeatures=None, lw=0.25, cbar=False,
                    ffig=None):
    """Plot VS - HC connectivity."""

    fig, ax = init_ax_fig(figsize=(5, 5))

    VC_HC = pd.DataFrame(np.round(VC_HC, 2), index=s_names, columns=vfeatures)
    annot_mat = VC_HC.astype(str)
    # Custom formatting below.
    annot_mat[VC_HC == 1] = '-1'
    annot_mat[VC_HC == 1] = '1'
    sns.heatmap(VC_HC, linewidth=lw, ax=ax, cbar=cbar,
                annot=annot_mat, fmt='')
    ax.set_title('VC -> HC')
    ax.set_xlabel('')
    ax.set_ylabel('')
    hide_tick_marks(ax)

    # Save figure.
    if ffig is not None:
        save_fig(ffig, fig)


def plot_GS_HC_conn(HC, xvec, yvec, s_iord=None, lw=0, cbar=False, square=True,
                    ffig=None):
    """Plot GS-HC connectivity maps."""

    # Init params.
    if s_iord is None:
        s_iord = list(range(HC.ns))
    nrow = int(np.sqrt(HC.ns))
    ncol = int(np.ceil(HC.ns / int(np.sqrt(HC.ns))))
    vmin = HC.GS_HC.min()
    vmax = HC.GS_HC.max()

    # Init axes.
    fig, axs = plt.subplots(nrow, ncol, figsize=(3*ncol, 3*nrow))
    # plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.95,
    #                     wspace=0.3, hspace=0.3)
    axs = axs.flatten()
    for i in range(HC.ns, len(axs)):
        fig.delaxes(axs[i])

    # Plot connectivity.
    for i, (s_idx, ax) in enumerate(zip(s_iord, axs)):

        # Get connectivity between GS and given HC state.
        loc = HC.s_names[s_idx]
        gs_hc_s = HC.GS_HC[s_idx, :, :]

        # Create DataFrame to use Seaborn's nice tick label formatting.
        hc_gs = pd.DataFrame(gs_hc_s, columns=xvec, index=yvec)
        sns.heatmap(hc_gs, vmin=vmin, vmax=vmax, linewidth=lw, square=square,
                    cbar=cbar, ax=ax)
        ax.invert_yaxis()
        # Format axes.
        ax.set_title(loc)
        hide_tick_marks(ax)
        if i != ncol * (nrow-1):
            hide_axis_labels(ax)
            hide_tick_labels(ax.xaxis)
            hide_tick_labels(ax.yaxis)

    fig.suptitle('GS - HC connectivity', y=1.05)

    # Save figure.
    if ffig is not None:
        save_fig(ffig, fig)


def plot_model_setup(HC, GS, VC, vS, dlS, village, gs_hc_sharp, fdir=''):
    """Plot model setup."""

    fdir += 'model/'

    # GS - HC
    fname_gs_hc = fdir + 'GS_HC_sharp_{}.png'.format(gs_hc_sharp)
    plot_GS_HC_conn(HC, GS.xvec, GS.yvec, ffig=fname_gs_hc)

    # VC - HC
    plot_VC_HC_conn(HC.VC_HC, HC.s_names, VC.features, ffig=fdir+'VS_HC.png')

    # vS
    plot_vS(vS, HC.s_names, ffig=fdir+'vS_r.png')

    # dlS
    plot_dlS(dlS, HC.s_names, village.U, ffig=fdir+'dlS_Q.png')

    # Village with vS r- and dlS Q-values added.
    plot_village(village, vS, dlS, ffig=fdir+'village_r_Q.png')


# %% Functions to plot environment-relation plots.

def plot_step(axs, village, GS, HC, istep=0, pause=1e-6, add_ttl=True):
    """Plot one simulation step."""

    # Init.
    ttl = 'step ' + str(istep) if add_ttl else ''
    loc_ord = pd.DataFrame(village.S_pos).T.sort_values(by=[0, 1]).index

    # Update animal position.
    village.redraw_animal()
    x, y = village.animal_coords()
    vill_ttl = (ttl + ', {}, pos: {:.0f} / {:.0f}'.format(village.s, x, y)
                if add_ttl else '')
    axs['village'].set_title(vill_ttl, loc='left')
    refresh(pause)

    # GS after motor update, before HC feedback.
    plot_GS(GS, axs['GS'], ttl, keep_every=5, int_vec=True)
    refresh(pause)

    # HC
    plot_HC(HC, axs['HC'], ttl, loc_ord)
    refresh(pause)


def plot_village(village, vS=None, dlS=None, add_title=True, add_animal=False,
                 add_loc_r=True, ax_equal=True, xlim=None, ylim=None, fig=None,
                 ax=None, figsize=None, ffig=None):
    """Plot village."""

    # Init figure.
    if figsize is None:
        figsize = (6, 6)
    fig, ax = init_ax_fig(ax, fig, figsize=figsize)

    fs = 'small'

    # Add routes.
    d = 0.3
    Q = dlS.as_DF(village.S, village.U) if dlS is not None else None
    qmin = Q.min().min() if dlS is not None else None
    qmax = Q.max().max() if dlS is not None else None
    qrng = qmax-qmin if qmax != qmin else 1
    for s, us in village.sus.items():
        for u, sx in us.items():
            s_x, s_y = village.S_pos[s]
            sx_x, sx_y = village.S_pos[sx]
            ax.plot([s_x, sx_x], [s_y, sx_y], alpha=0.25, c='g')
            # Add route labels (habit values).
            if Q is not None:
                q = Q.loc[u, s]
                lbl = '{:.1f}'.format(q)
                xy = [(1-d)*s_x + d*sx_x, (1-d)*s_y + d*sx_y]
                dr = ('r' if s_x < sx_x else 'l')  # direction of arrow
                fc = str((qmax-q)/qrng)  # label face color
                bbox = dict(boxstyle=dr+'arrow,pad=0.2', fc=fc, ec='0.5')
                col = 'w' if float(fc) < 0.5 else 'k'
                if s_x == sx_x:
                    rot = 90 if s_y > sx_y else -90
                elif (dr == 'r' and s_y < sx_y) or (dr == 'l' and s_y > sx_y):
                    rot = 45
                elif (dr == 'r' and s_y > sx_y) or (dr == 'l' and s_y < sx_y):
                    rot = -45
                else:
                    rot = 0
                ax.annotate(lbl, xy, color=col, ha='center', va='center',
                            size=fs, bbox=bbox, rotation=rot)

    # Add location labels.
    r_vS = vS.as_Ser(village.S) if vS is not None else None
    bbox = dict(boxstyle='round', fc='0.95', ec='0.5')
    for loc, xy in village.S_pos.items():
        lbl = loc + ('\n{}'.format(village.R[loc]) if add_loc_r else '')
        if r_vS is not None:
            lbl += ' | {:.1f}'.format(r_vS[loc])
        ax.annotate(lbl, xy, ha='center', va='center', size=fs, bbox=bbox)

    # Add animal.
    if add_animal:
        if village.animal_artist.axes is not None:
            village.animal_artist.remove()
        ax.add_artist(village.animal_artist)

    # Format axis.
    ax.axis('off')
    if ax_equal:
        ax.axis('equal')

    # Set axis limits.
    pad = 0.0
    S_pos = np.array(list(village.S_pos.values()))
    if xlim is None:
        xmin, xmax = [S_pos[:, 0].min(), S_pos[:, 0].max()]
        xpad = pad * (xmax - xmin)
        xlim = [xmin-xpad, xmax+xpad]
    if ylim is None:
        ymin, ymax = [S_pos[:, 1].min(), S_pos[:, 1].max()]
        ypad = pad * (ymax - ymin)
        ylim = [ymin-ypad, ymax+ypad]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Add title.
    if add_title:
        nstates, npaths = len(village.S), village.n_paths()
        ttl = 'n states: {}, n paths: {}'.format(nstates, npaths)
        ax.set_title(ttl)

    # Save figure.
    if ffig is not None:
        save_fig(ffig, fig)

    return fig, ax


def plot_env_sim(anim_data, vis_data, mot_data, dirname):
    """Plot environment-related simulation results."""

    # Set up figure.
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.95,
                        wspace=0.3, hspace=0.3)

    # Distribution of locations visited and actions taken.
    for name, col, ax in [('s', 'm', axs[0, 0]), ('u', 'g', axs[0, 1])]:
        anim_data.groupby(name).size().plot.bar(color=col, title=name, ax=ax)
        rot_xtick_labels(ax, rot=45)
        ax.set_xlabel('')
        ax.set_ylabel('n')

    # Visual input error.
    ax = axs[1, 0]
    vis_err = (vis_data['vvis'] - vis_data['ovis']).abs()
    vis_err.plot.hist(alpha=0.5, ax=ax, title='abs. visual input error distr')
    ax.set_xlabel('error')
    ax.set_ylabel('n')
    ax.legend().set_title('')

    # Distribution of motor feedback error.
    ax = axs[1, 1]
    mot_err = mot_data['vmot'] - mot_data['umot']
    mot_err.plot.hist(alpha=0.5, ax=ax, title='motor feedback error distr')
    ax.set_xlabel('error')
    ax.set_ylabel('n')

    # Save figure.
    fname = dirname + 'env.png'
    save_fig(fname, fig)


# %% Functions to plot model component states
# -------------------------------------------

def plot_GS(GS, ax=None, ttl='', keep_every=2, cbar=False, clear=True,
            vmin=0, vmax=None, int_vec=False):
    """Plot GS estimate (activity) of position map."""

    fig, ax = init_ax_fig(ax, figsize=(5, 5))

    if clear:
        ax.clear()

    # Get activity DF.
    GS_df = GS.as_DF()
    if int_vec:
        GS_df.index = GS_df.index.astype(int)
        GS_df.columns = GS_df.columns.astype(int)

    # Plot activity.
    sq = GS.xres == GS.yres
    sns.heatmap(GS_df, vmin=vmin, vmax=vmax, cbar=cbar, square=sq, ax=ax)
    ax.invert_yaxis()
    # Add info title.
    x, y = analysis.GS_pos_mean(GS.P, GS.xvec, GS.yvec, GS.circular)
    ttl = ttl + ', ' if len(ttl) else ''
    # title = 'GS ' + ttl + 'position estimate [{:.0f} / {:.0f}]'.format(x, y)
    title = 'GS position estimate'
    ax.set_title(title)
    # Format axes.
    hide_tick_marks(ax)
    for axis in [ax.xaxis, ax.yaxis]:
        keep_nth_tick_label(axis, keep_every)
    rot_ytick_labels(ax, rot=0, va='center')


def plot_HC(HC, ax=None, ttl='', loc_ord=None, clear=True):
    """Plot HC estimate."""

    fig, ax = init_ax_fig(ax)

    if clear:
        ax.clear()

    if loc_ord is None:
        loc_ord = HC.s_names

    hc_s = HC.as_DF()
    # mll_est = [src + ': ' + loc for src, loc in hc_s.idxmax().items()]
    # ttl = ttl + ', ' if len(ttl) else ''
    # title = 'HC ' + ttl + ' / '.join(mll_est)
    title = 'HC location estimate '
    hc_s.loc[loc_ord].plot.bar(ax=ax, ylim=[0, 1], title=title)
    rot_xtick_labels(ax, rot=45, ha='right')
    ax.set_xlabel('')
    ax.set_ylabel('p')
    ax.legend().set_title('input type')
    hide_top_right_spines(ax)


def plot_vS(vS, s_names, ax=None, ffig=None):
    """Plot Q function of dlS."""

    fig, ax = init_ax_fig(ax)

    R = vS.as_Ser(s_names)
    R.plot.bar(ax=ax)

    # Save figure.
    if ffig is not None:
        save_fig(ffig, fig)


def plot_dlS(dlS, s_names, u_names, ax=None, ffig=None):
    """Plot Q function of dlS."""

    fig, ax = init_ax_fig(ax, figsize=(5, 5))

    Q = dlS.as_DF(s_names, u_names)
    sns.heatmap(Q, linewidth=0.1, square=True, annot=True, cbar=False, ax=ax)
    ax.xaxis.tick_top()
    rot_xtick_labels(ax, rot=90, ha='center')

    # Save figure.
    if ffig is not None:
        save_fig(ffig, fig)


# %% Plotting random exploration results
# --------------------------------------

def plot_navig_res(village, HC, vS, dlS, anim_data, vis_data, mot_data,
                   gs_state, hc_state, vs_state, dls_state, gs_pars, fd):
    """High level function to plot all navigation results."""

    gs_mpos, gs_H = analysis.GS_pos_estimate(gs_state, gs_pars['circular'])
    hc_maxLL = analysis.HC_loc_estimate(hc_state)
    hc_H = analysis.HC_entropy(hc_state)

    plot_village(village, vS, dlS, ffig=fd+'village.png')
    plot_env_sim(anim_data, vis_data, mot_data, fd)
    plot_GS_HC_sim_res(anim_data, gs_state, hc_state, gs_mpos, gs_H,
                       hc_maxLL, hc_H, fd, window=20, step=10)
    plot_confusion_matrices(anim_data, hc_state, hc_maxLL, fd)
    plot_reward_learning(village.R, vs_state, dls_state, fd)
    plot_habits(dlS.as_DF(HC.s_names, village.U), village, fd)

    return gs_mpos, gs_H, hc_maxLL, hc_H


def plot_GS_HC_sim_res(anim_data, hc_state, gs_mpos, gs_H, hc_maxLL,
                       hc_H, dirname, conn_on=None, window=20, step=10):
    """Plot GS and HC simulation results."""

    # Set up figure.
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.95,
                        wspace=0.2, hspace=0.6)

    axs_iter = iter(axs.flatten())

    # Set up connectivity config variables.
    if conn_on is None:
        conn_on = setup.get_conn_on_df(len(anim_data.index), full_only=True)
    conn_config = conn_on.apply(process.conn_config_name, axis=1)
    conn_config.name = 'conn_config'
    cfs = conn_config.unique()
    cconfs = pd.Series([conn_config[conn_config == cf].index[0] for cf in cfs],
                       index=cfs)
    all_conn_on_steps = conn_on.sum(axis=1) == len(conn_on.columns)
    conn_config_ord = conn_config.unique()
    from_ord = list(hc_state.columns)

    # GS
    # --

#    # Accuracy and entropy (inverse precision) of GS position estimate.
#    pos_err = (gs_mpos - anim_data[['x', 'y']]).pow(2).sum(axis=1).pow(1/2)
#
#    # Position estimate error time series.
#    ax = next(axs_iter)
#    pos_err_mean = pos_err.rolling(window, min_periods=1, center=True).mean()
#    pos_err_mean[::step].plot(title='rolling GS position estimate error',
#                              color='red', ax=ax)
#    add_conn_config_markers(cconfs, ax)
#    ax.set_ylabel('error')
#
#    # Entropy time series.
#    ax = next(axs_iter)
#    lims = '[{:.1f} - {:.1f}]'.format(gs_H.min(), gs_H.max())
#    H_mean = gs_H.rolling(window, min_periods=1, center=True).mean()
#    H_mean[::step].plot(title='rolling GS entropy ' + lims, color='red', ax=ax)
#    add_conn_config_markers(cconfs, ax)
#    ax.set_ylabel('H')

    # HC
    # --

    # Accuracy and entropy (inverse precision) of HC location estimate.
    hc_corr = hc_maxLL.apply(lambda col: col == anim_data['s']).astype('int')

    # Accuracy time series.
    ax = next(axs_iter)
    hc_corr_mean = hc_corr.rolling(window, min_periods=1, center=True).mean()
    hc_corr_mean[::step].plot(title='rolling HC accuracy', ax=ax)
    ax.set_title('HC state recognition accuracy', y=1.08)
    add_conn_config_markers(cconfs, ax)
    ax.set_ylabel('prop. correct')
    ax.legend().set_title('')
    ax.set_ylim([0, 1])
    hide_top_right_spines(ax)

#    # Entropy time series.
#    ax = next(axs_iter)
#    H_mean = hc_H.rolling(window, min_periods=1, center=True).mean()
#    H_mean[::step].plot(title='rolling HC entropy', ax=ax)
#    add_conn_config_markers(cconfs, ax)
#    ax.set_ylabel('H')
#    ax.legend().set_title('')

    # Accuracy mean.
    ax = next(axs_iter)
    hc_corr_con = pd.concat([hc_corr, conn_config], axis=1)
    hc_corr_con_mean = hc_corr_con.groupby(conn_config.name).mean()
    lhc_corr_con = hc_corr_con_mean.stack().reset_index()
    lhc_corr_con.columns = [conn_config.name, 'from', 'prop. correct']
    sns.barplot(data=lhc_corr_con, x='from', y='prop. correct', order=from_ord,
                hue=conn_config.name, hue_order=conn_config_ord, ax=ax)
    ax.axhline(1, color='grey', alpha=1, lw=1, ls='--')
    # ax.set_title('HC accuracy')
    ax.set_xlim([None, 3.8])
    ax.set_ylim([0, 1])
    ax.set_xlabel('')
    ax.legend(ncol=1).set_title('')
    hide_tick_marks(ax, axis='x')
    hide_top_right_spines(ax)

#    # Entropy mean.
#    ax = next(axs_iter)
#    hc_H_conn = pd.concat([hc_H, conn_config], axis=1)
#    lhc_H_conn = hc_H_conn.melt(id_vars=[conn_config.name], value_name='H',
#                                var_name='from')
#    sns.boxplot(data=lhc_H_conn, x='from', y='H', hue=conn_config.name,
#                order=from_ord, hue_order=conn_config_ord, ax=ax)
#    ax.set_title('HC entropy')
#    ax.set_xlabel('')
#    ax.legend().set_title('')
#    hide_tick_marks(ax, axis='x')
#
#    # State estimate correlation (co-occurance of ML choices).
#    ax = next(axs_iter)
#    hc_maxLL_real = hc_maxLL.copy()
#    hc_maxLL_real['real'] = anim_data['s']
#    hc_maxLL_all_conn = hc_maxLL_real.loc[all_conn_on_steps]
#    psame = {(c1, c2): (hc_maxLL_all_conn[c1] == hc_maxLL_all_conn[c2]).mean()
#             for c1, c2 in product(hc_maxLL_all_conn.columns, repeat=2)}
#    psame = pd.Series(psame).unstack()
#    psame = psame.loc[hc_maxLL_real.columns][hc_maxLL_real.columns]
#    psame = psame.iloc[1:, :-1]
#    mask = np.ones_like(psame)
#    mask[np.tril_indices_from(mask)] = False
#    sns.heatmap(psame, mask=mask, annot=True, linewidths=0.5,
#                cbar=False, ax=ax)
#    hide_tick_marks(ax)
#    ax.set_title('HC prop. same ML choice (full model only)')
#    ax.set_xlabel('')
#    ax.set_ylabel('')
#
#    # Entropy correlation.
#    ax = next(axs_iter)
#    hc_H_all_conn = hc_H.loc[all_conn_on_steps]
#    Hcorr = hc_H_all_conn.corr().iloc[1:, :-1]
#    mask = np.ones_like(Hcorr)
#    mask[np.tril_indices_from(mask)] = False
#    sns.heatmap(Hcorr, mask=mask, annot=True, linewidths=0.5,
#                cbar=False, ax=ax)
#    hide_tick_marks(ax)
#    ax.set_title('HC entropy corr. (full model only)')
#    ax.set_xlabel('')
#    ax.set_ylabel('')

    # Save figure.
    fname = dirname + 'GS_HC_conn_config_analysis.png'
    save_fig(fname, fig)


def plot_confusion_matrices(anim_data, hc_state, hc_maxLL, dirname,
                            conn_on=None, cbar=False, annot=True,
                            min_annot_val=5):
    """Plot confusion matrices of ."""

    names = ['real loc', 'HC loc']

    # Select steps with full model only (all connectivities on).
    full_mod_steps = ((conn_on.sum(axis=1) == len(conn_on.columns))
                      if conn_on is not None else anim_data.index)
    anim_data_full = anim_data.loc[full_mod_steps]
    full_idxs = anim_data_full.index
    g_real_s = full_idxs.groupby(anim_data_full.s)

    # Get mean estimate for each real state (extention of conf mat to PD).
    conf_mat_PD = {s: hc_state.loc[list(idxs)].groupby(level='loc').mean()
                   for s, idxs in g_real_s.items()}
    conf_mat_PD = pd.concat(conf_mat_PD, names=names)

    # Get canonical confusion matrix on Maximum Likelihood state estimate.
    conf_mat_LL = {s: hc_maxLL.loc[idxs].apply(
                    lambda col: col.value_counts(normalize=True))
                   for s, idxs in g_real_s.items()}
    conf_mat_LL = pd.concat(conf_mat_LL, names=names).fillna(0)

    # Set up figure.
    fig, axs = plt.subplots(2, len(hc_state.columns), figsize=(15, 10))
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.94,
                        wspace=0.6, hspace=0.6)  # tightlayout cancels this

    # Plot each confusion matrix on heatmap.
    yttl = 1.15 if cbar else 1.075
    CM_res = [('PD', conf_mat_PD, None), ('ML', conf_mat_LL, 100)]
    for irow, (ntype, conf_mat, vmax) in enumerate(CM_res):
        for icol, col in enumerate(conf_mat):
            CM = (100 * conf_mat[col].unstack().fillna(0)).astype(int)
            annot_mat = False
            if annot:
                annot_mat = CM.astype(str)
                annot_mat[CM < min_annot_val] = ''
            ax = axs[irow, icol]
            sns.heatmap(CM, vmin=0, vmax=vmax, square=True, cbar=cbar, ax=ax,
                        annot=annot_mat, fmt='')
            # Format axes.
            ax.xaxis.tick_top()
            rot_xtick_labels(ax, rot=90, ha='center')
            mcorr, scorr = np.diag(CM).mean(), np.diag(CM).std()
            ttl = '{}  {}: {:.1f} +- {:.1f}'.format(ntype, col, mcorr, scorr)
            ax.set_title(ttl, y=yttl)
            hide_tick_marks(ax)

    # Save figure.
    fname = dirname + 'HC_confusion_matrix.png'
    save_fig(fname, fig)


def plot_habits(Q, village, dirname):
    """
    Plot habits in the form of a Q value matrix: (states, action) -> value map.
    Only works for direct village location <--> HC state mapping!
    Learning in HC will mess up habit following analysis!
    """

    # Set up figure.
    fig, ax = init_ax_fig(figsize=(6, 6))

    # Final state from each start location after following habit.
    fin_s = analysis.follow_habit_moves(Q, village)
    final_s = pd.DataFrame(0, columns=village.S, index=village.S)
    final_s.columns.name = 'start location'
    final_s.index.name = 'final location'
    for s, fs in fin_s.items():
        final_s.loc[fs, s] = 1
    sns.heatmap(final_s, linewidth=1, cbar=False, ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    hide_tick_marks(ax)
    rot_xtick_labels(ax, rot=90, ha='center')

    # Save figure.
    fname = dirname + 'habits.png'
    save_fig(fname, fig)


# %% Plot parameter sweep analysis results
# ----------------------------------------

def plot_navig_param_sweep(res, nres, fname, title=None, annot=True):
    """Plot parameter sweep analysis results."""

    # Set up figure.
    cbar = not annot
    nrow, ncol = 1, len(res.columns)
    fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol+3*cbar, 5*nrow))
    if nrow == 1 and ncol == 1:
        axs = np.array([axs])
    axs_iter = iter(axs.flatten())
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.95,
                        wspace=0.2, hspace=0.6)

    for i, col_name in enumerate(res):

        mean_res = res[col_name].groupby(level=[0, 1]).mean().unstack()
        mean_res = (100 * mean_res).astype(int)

        ax = next(axs_iter)
        sns.heatmap(mean_res, vmin=0, vmax=100, cbar=cbar, annot=annot,
                    fmt='d', ax=ax)

        ax.invert_yaxis()
        ttl = '{} {}'.format(col_name, nres)
        ax.set_title(ttl)

        if i != 0:   # if i != nrow-1 or j != 0:
            hide_axis_labels(ax)
        hide_tick_marks(ax)
        rot_ytick_labels(ax, rot=0, va='center')

    # Add super title.
    if title is not None:
        fig.suptitle(title, y=1.05)

    # Save figure.
    save_fig(fname, fig)


def plot_learning_param_sweep(sel_real_h, sel_hc_h, split_by, fname,
                              title=None, annot=True):
    """Plot parameter sweep analysis results."""

    split_by_vals = sel_real_h.index.get_level_values(split_by).unique()

    # Set up figure.
    cbar = not annot
    nrow, ncol = 2, len(split_by_vals)
    fig, axs = plt.subplots(nrow, ncol, figsize=(5*ncol+3*cbar, 5*nrow))
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.95,
                        wspace=0.2, hspace=0.3)

    for i, (res_name, res) in enumerate([('real H', sel_real_h),
                                         ('HC H', sel_hc_h)]):
        for j, col_name in enumerate(split_by_vals):

            # Select by split-by value and take mean across reset values.
            c_res = res.xs(col_name, level=split_by)
            mean_res = c_res.groupby(level=[0, 1]).mean().unstack()

            ax = axs[i, j]
            sns.heatmap(mean_res, vmin=0, cbar=cbar, annot=annot,
                        fmt='.2f', ax=ax)

            ax.invert_yaxis()
            ttl = '{}, {} {}'.format(res_name, split_by, col_name)
            ax.set_title(ttl)

            if i != nrow-1 or j != 0:
                hide_axis_labels(ax)
            hide_tick_marks(ax)
            rot_ytick_labels(ax, rot=0, va='center')

    # Add super title.
    if title is not None:
        fig.suptitle(title, y=1.05)

    # Save figure.
    save_fig(fname, fig)


# %% Plot learning results
# ------------------------

def plot_learning_stats(res, s_real_h, s_hc_h, gs_hc_pos, vc_hc_snr,
                        gs_circular, gs_xsize, gs_ysize, fdir=None,
                        lw=1, alpha=0.5):
    """Plot results from connectivity learning."""

    # Plot simulation results over time.
    fig, axs = plt.subplots(5, 1, figsize=(12, 10))
    ax_list = axs.flatten()
    axs_iter = iter(ax_list)

    # Magnitude of weight change (amount of learning) over time.
    ax = next(axs_iter)
    res[['dVC_HC_max', 'dGS_HC_max']].plot(ax=ax, alpha=alpha, lw=lw)
    ax.axhline(0, c='grey')
    ax.set_ylabel('max delta weight (length)')

#    # Change in mean location of GS - HC connectivity.
#    d_gs_hc_pos = gs_hc_pos.diff()
#    dx, dy = [d_gs_hc_pos.xs(c, level=1, axis=1) for c in ['x', 'y']]
#    d_gs_hc_mean = (dx**2 + dy**2) ** 0.5
#
#    ax = next(axs_iter)
#    d_gs_hc_mean.plot(legend=False, alpha=alpha, lw=lw, ax=ax)
#    ax.axhline(0, c='grey')
#    ax.set_ylabel('delta GS - HC mean')

    # Maximum values of GS - HC connectivity per HC 'state' (N highest).
    # Could add entropy instead, but that takes a lot of time to calculate.
    n_s = len(s_real_h.columns)
    n_max = res['gs_hc_max'].apply(lambda v: np.sort(v)[-n_s:])
    n_max = pd.DataFrame.from_items(zip(n_max.index, n_max.values)).T

    ax = next(axs_iter)
    n_max.plot(legend=False, lw=lw, alpha=alpha, ax=ax)
    ax.axhline(0, c='grey')
    ax.set_ylabel('GS - HC max')

#    # GS position estimate difference over time
#    # (relative to an arbitrary reference point).
#    xdist = np.abs(res['gs_x'] - res['x'])
#    ydist = np.abs(res['gs_y'] - res['y'])
#    # TODO: this part does not work, needs fixing!
#    if gs_circular:
#        xdist = xdist % (gs_xsize/2)
#        ydist = ydist % (gs_ysize/2)
#    gs_dist = (xdist**2 + ydist**2) ** 0.5
#
#    ax = next(axs_iter)
#    gs_dist.plot(ax=ax, c='r', lw=lw)
#    ax.axhline(0, c='grey')
#    ax.set_ylabel('GS pos estim diff')

    # GS entropy.
    ax = next(axs_iter)
    res['gs_h'].plot(ax=ax, lw=lw)
    ax.axhline(0, c='grey')
    ax.set_ylabel('GS entropy')

    # VC - HC SNR (degree of binary-ness).
    # ax = next(axs_iter)
    # vc_hc_snr.plot(legend=False, lw=lw, alpha=alpha, ax=ax)
    # ax.axhline(0, c='grey')
    # ax.set_ylabel('VC-HC SNR')

#    # HC entropy.
#    ax = next(axs_iter)
#    res['hc_h'].plot(ax=ax, c='g', lw=lw)
#    ax.axhline(0, c='grey')
#    ax.set_ylabel('HC ro entropy')

    # Entropy across HC states (ML) for each world state.
    ax = next(axs_iter)
    s_hc_h.plot(legend=False, lw=lw, alpha=alpha, ax=ax)
    ax.axhline(0, c='grey')
    ax.set_ylabel('HC state entropy')

    # Entropy across world states for each HC state (ML).
    ax = next(axs_iter)
    s_real_h.plot(legend=False, lw=lw, alpha=alpha, ax=ax)
    ax.axhline(0, c='grey')
    ax.set_ylabel('Loc. recog. entropy')

    # Remove x axis from all but bottom plot.
    for i in range(len(ax_list)-1):
        ax = ax_list[i]
        ax.set_xlabel('')
        hide_tick_labels(ax.xaxis)

    # Save figure.
    if fdir is not None:
        fname = fdir+'stats.png'
        save_fig(fname, fig)


def plot_gs_hc_mean_traj(gs_hc_pos, gs_hc_h, gs_hc_max, GS, last_n=None,
                         center=True, n_real_s=None, ttl=None, ax=None,
                         fdir=None):
    """Plot mean position of GS - HC connectivity per HC state over time."""

    # Take last n samples and split by axis.
    gs_hc_pos_n = gs_hc_pos.tail(last_n) if last_n is not None else gs_hc_pos
    gs_hc_x, gs_hc_y = [gs_hc_pos_n.xs(c, level=1, axis=1).copy()
                        for c in ('x', 'y')]

    # Get axis limits.
    xmin, xmax = GS.xmin, GS.xmax
    ymin, ymax = GS.ymin, GS.ymax

    # Center trajectories for clearer plot.
    if center:
        # HC states with strongest connectivity.
        max_hc_s = gs_hc_max.sort_values(ascending=False).iloc[:n_real_s].index
        # Get original axis widths.
        xh, yh = (xmax-xmin)/2, (ymax-ymin)/2
        xprd, yprd = 2*xh+GS.xres, 2*yh+GS.yres
        # Center coordinates by taking strongest HC states as reference.
        for i in range(2):  # do it several times to improve centering

            gs_hc_x = utils.center_axis(gs_hc_x, xmin, xmax, xprd, max_hc_s)
            gs_hc_y = utils.center_axis(gs_hc_y, ymin, ymax, yprd, max_hc_s)

        # Adjust axis limits.
        xmin, xmax = -xh, xprd-xh
        ymin, ymax = -xh, xprd-xh

    S = gs_hc_pos_n.columns.get_level_values(0).unique()

    # Set up figure and some plotting params.
    fig, ax = init_ax_fig(ax, figsize=(10, 10))
    ax.set_facecolor('black')
    colors = plt.cm.rainbow(np.linspace(0, 1, len(S)))

    # Plot trajectories.
    gs_hc_alpha = gs_hc_max / gs_hc_max.max()
    for (s, alpha), col in zip(gs_hc_alpha.items(), colors):
        # Trajectory.
        x = np.array(gs_hc_x[s])
        y = np.array(gs_hc_y[s])
        ax.plot(x, y, '--', lw=1, c=col, alpha=alpha, zorder=1)
        # Final point.
        xlast, ylast = x[-1], y[-1]
        rad = 10  # fixed size
        # rad = np.sqrt(np.pi * h)  # entropy-dependent size
        circle = plt.Circle((xlast, ylast), radius=rad, color=col,
                            alpha=alpha, zorder=2)
        ax.add_artist(circle)
        # Add name.
        ax.text(xlast, ylast, s, color='white', fontsize='medium',
                va='center', ha='center', alpha=alpha)

    # Format figure.
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ttl = ttl if ttl is not None else 'GS - HC mean position'
    ax.set_title(ttl)
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.axis('equal')  # this changes axis limits!
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Save figure.
    if fdir is not None:
        fname = fdir+'gs_hc_traj.png'
        save_fig(fname, fig)


def plot_learned_vc_hc(optVC_HC, VC_HC, vfeats, s_real, s_hc, co_occs,
                       s_name_srtd, fdir=None):
    """Plot result of VC - HC learning, plus state co-occurance."""

    # VC -> HC
    lrndVC_HC = pd.DataFrame(VC_HC, index=s_hc, columns=vfeats)
    nopt, nres = optVC_HC.shape[0], VC_HC.shape[0]
    vc_hc_corr = pd.DataFrame(0, index=s_hc, columns=s_real)
    for ires, iopt in product(range(nres), range(nopt)):
        vres, vopt = VC_HC[ires], optVC_HC[iopt]
        vc_hc_corr.iloc[ires, iopt] = np.corrcoef(vres, vopt)[0, 1]

    optVC_HC_df = pd.DataFrame(optVC_HC, index=s_real, columns=vfeats)

    # Normalize co-occurance matrix per real state (difference in their
    # visit frequency is a matter of maze design, i.e. not of interest).
    co_occs_normed = co_occs / co_occs.sum(0)
    co_occs_normed = pd.DataFrame(co_occs_normed, index=s_hc, columns=s_real)

    # Match HC state order with that of world.
    lrndVC_HC = lrndVC_HC.reindex(s_name_srtd)
    vc_hc_corr = vc_hc_corr.reindex(s_name_srtd)
    co_occs_normed = co_occs_normed.reindex(s_name_srtd)

    # Plot results.
    fig, axs = plt.subplots(1, 4, figsize=(18, 12))
    axs_iter = iter(axs.flatten())

    for rslt, ttl, cbar in [(optVC_HC_df, 'optimal VC - HC', False),
                            (lrndVC_HC, 'learned VC - HC', True),
                            (vc_hc_corr, 'weight correlation', True),
                            (co_occs_normed, 'estimate co-occ', True)]:
        ax = next(axs_iter)
        sns.heatmap(rslt, square=True, xticklabels=True, yticklabels=True,
                    cbar=cbar, ax=ax)
        ax.xaxis.tick_top()
        rot_xtick_labels(ax, rot=90, ha='center')
        rot_ytick_labels(ax, rot=0, va='center')
        ax.set_title(ttl, y=1.10)
        hide_tick_marks(ax)
        ax.set_xlabel('')
        ax.set_ylabel('')

    # Save figure.
    if fdir is not None:
        fname = fdir+'vc_hc_co_occs_learned.png'
        save_fig(fname, fig)


def plot_reward_learning(R, vs_state, dls_state, dirname):
    """Plot reward and habit learning."""

    # Set up figure.
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.94,
                        wspace=0.2, hspace=0.6)

    # Plot reward prediction accuracy of ventral striatum.
    ax = axs[0]
    (vs_state.unstack() - R).plot(ax=ax)
    ax.set_xlabel('step')
    ax.set_ylabel('difference (expected - real)')
    ax.set_title('vS reward expectation learning')
    ax.legend().set_title('')

    # Plot change in Q values of dorsomedial striatum.
    ax = axs[1]
    dls_Q = dls_state.unstack()
    dls_Q.plot(ax=ax, legend=False)
    ax.set_xlabel('step')
    ax.set_ylabel('Q')
    ax.set_title('dlS habit learning')

    # Save figure.
    fname = dirname + 'reward_habit_learning.png'
    save_fig(fname, fig)


# %% Plot simulation results on navigation types.

def plot_rolling_stats_by_type(res, window=50, step=1, ttl=None, fdir=None):
    """Plot trial stats per nagivation type over time."""

    num_cols = res.select_dtypes('number').columns
    stats_to_plot = set(num_cols) - set(['nav type', 'trial', 'start loc'])
    n_plots = len(stats_to_plot)

    # Set up figure.
    fig, axs = plt.subplots(n_plots, 1, figsize=(12, 3*n_plots))
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.94,
                        wspace=0.2, hspace=0.6)
    axs = axs.flatten()

    # Plot each stats.
    for i, (stats, ax) in enumerate(zip(stats_to_plot, axs)):
        nav_roll = {nav: rg[stats].rolling(window, min_periods=1, center=True)
                    for nav, rg in res.groupby('nav type')}
        rmean = pd.DataFrame({nav: list(roll.mean())
                              for nav, roll in nav_roll.items()})
        type_order = [nav for nav in ['opt', 'HAS', 'IBP', 'rnd']
                      if nav in rmean.columns]
        rmean = rmean[type_order]
        rmean[::step].plot(ax=ax, legend=(i == 0))
        ax.set_ylabel(stats)

    # Format specific axes.
    axs[-1].set_xlabel('trial')

    fig.suptitle(ttl, y=1.02, fontsize='x-large')

    # Save figure.
    if fdir is not None:
        ffig = fdir + 'rolling_stats.png'
        save_fig(ffig, fig)


def plot_learned_navig_by_type(res, village, per_loc, ttl=None, fdir=None):
    """Plot navig performance per nagivation type per starting location."""

    figsize = (6, 4) if per_loc else (2, 3)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Remove error bars from random navigation (to save plotting space).
    cres = res.copy()
    if 'random' in cres['nav type'].unique():
        cres_rnd = cres.loc[cres['nav type'] == 'random']
        msteps = cres_rnd.groupby(['start loc']).mean()
        new_steps = list(msteps.loc[cres_rnd['start loc'], 'n steps'])
        cres.loc[cres['nav type'] == 'random', 'n steps'] = new_steps

    # Plot number of steps taken to reach goal state per type.
    type_order = ['optimal', 'HAS', 'IBP', 'random']
    if per_loc:  # for each start loc separately
        means = cres.groupby(['nav type', 'start loc']).mean()
        sort_by = type_order  # or ['IBP']
        loc_ord = means['n steps'].unstack().T.sort_values(sort_by).index
        sns.barplot(data=cres, x='start loc', y='n steps', hue='nav type',
                    order=loc_ord, hue_order=type_order, errcolor='grey',
                    errwidth=1.25, ax=ax)
    else:   # average across start locs
        sns.barplot(data=cres, x='nav type', y='n steps', order=type_order,
                    errcolor='grey', errwidth=2.5, ax=ax)
        ax.set_xlabel('')
        rot_xtick_labels(ax, rot=45)

    ax.set_ylabel('# steps to goal')
    hide_top_right_spines(ax)
    hide_tick_marks(ax, axis='x')

    fig.suptitle(ttl, y=1.02)

    # Save figure.
    if fdir is not None:
        ffig = fdir + ('per_loc' if per_loc else 'combined') + '_res.png'
        save_fig(ffig, fig)


def plot_cooccs_mat(s_real, s_hc, co_occs, s_name_srtd=None, vmin=0, vmax=1,
                    cbar=True, ttl=None, ax=None, fdir=None):
    """Plot village - HC state co-occurance matrix."""

    # Normalize co-occurance matrix per real state (difference in their
    # visit frequency is a matter of maze design, i.e. not of interest).
    co_occs_norm = process.norm_co_occ_matrix(co_occs)
    co_occs_norm = pd.DataFrame(co_occs_norm, index=s_hc, columns=s_real)

    # Match HC state order with that of the environment.
    if s_name_srtd is None:
        s_iord, s_name_srtd = process.get_s_order(co_occs, s_hc)
    co_occs_norm = co_occs_norm.reindex(s_name_srtd)

    # Init figure.
    fig, ax = init_ax_fig(ax)

    # Plot co-occ matrix.
    sns.heatmap(co_occs_norm, square=True, vmin=vmin, vmax=vmax,
                xticklabels=True, yticklabels=True, cbar=cbar, ax=ax)
    ax.xaxis.tick_top()
    rot_xtick_labels(ax, rot=90, ha='center')
    rot_ytick_labels(ax, rot=0, va='center')
    ttl = ttl if ttl is not None else 'co-occurrance ratio'
    ax.set_title(ttl, y=1.10)
    hide_tick_marks(ax)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Save figure.
    if fdir is not None:
        ffig = fdir + 'coocs_mat.png'
        save_fig(ffig, fig)


def plot_trajs_cooc_habits_by_type(village, res, dmodels, dcoocs, ttl=None,
                                   fdir=None):
    """Plot both habits on village and co-occ matrix per type."""

    # Set up figure.
    ntypes = len(dmodels.keys())
    fig, axs = plt.subplots(3, ntypes, figsize=(6*ntypes, 18))
    if ntypes == 1:
        axs = axs[:, np.newaxis]
    plt.subplots_adjust(left=.07, bottom=.06, right=.95, top=.94,
                        wspace=0.2, hspace=0.6)

    # Plot GS - HC conn trajectories, co-occs and habits by type.
    for i, nav in enumerate(dcoocs):

        # Init data to plot.
        model = dmodels[nav]
        co_occs = dcoocs[nav]
        GS, HC, vS, dlS = [model[r] for r in ('GS', 'HC', 'vS', 'dlS')]

        # GS - HC conn trajectories.
        ax = axs[0, i]
        nav_res = res.loc[res['nav type'] == nav]
        r = process.format_GS_HC_rec(nav_res, HC.s_names, HC.GS_HC)
        gs_hc_pos, gs_hc_h, gs_hc_max = r
        plot_gs_hc_mean_traj(gs_hc_pos, gs_hc_h, gs_hc_max, GS, None, False,
                             len(village.S), ttl=nav, ax=ax)

        # Co-occs.
        ax = axs[1, i]
        plot_cooccs_mat(village.S, HC.s_names, co_occs, cbar=False,
                        ttl='', ax=ax)

        # Habits.
        ax = axs[2, i]
        plot_village(village, vS, dlS, add_title=False, ax=ax)

    fig.suptitle(ttl, y=1.02, fontsize='xx-large')

    # Save figure.
    if fdir is not None:
        ffig = fdir + 'trajs_coocs.png'
        save_fig(ffig, fig)
