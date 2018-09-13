#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Brain model.

Includes model of:
    - hippocampus: central, integrated state representation (hidden state)
    - visual cortex: detailed visual input (observations)
    - grid system: location representations

@author: David Samu
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.signal import convolve2d

from mb_mf_nav import utils


# %% HC

class Hippocampus:
    """Hippocampus model."""

    def __init__(self, s_names, GS_HC, VC_HC, ro_pow, **kws):
        """Init HC."""

        # State parameters.
        self.s_names = s_names
        self.ns = len(self.s_names)
        self.ro_pow = ro_pow

        # State estimate PD vector per input + integrated.
        self.s_types = ['VC', 'GS', 'both']
        self.s = {n: np.zeros(self.ns) for n in self.s_types}

        # "Connectivity" between each grid cell and HC state variable.
        # Alternatively, the P(GC, HC) generative model.
        self.GS_HC = GS_HC

        # Bottom-up connectivity from visual input and HC state variable.
        # Alternatively, the P(VC, HC) generative model.
        self.VC_HC = VC_HC

    def reset(self, var=0.0):
        """Initialize / reset activity to somewhat random uniform values."""

        for stype in self.s:
            self.s[stype][:] = 1./self.ns
            if var != 0:
                rnd = utils.rand_mat(self.ns, 1, 2)
                self.s[stype] += 1./self.ns * var * rnd
                self.s[stype] = self.s[stype] / self.s[stype].sum()

    def visual_input(self, vvis, tau=1):
        """Calculate input to HC from visual stimulus (observations)."""

        # P(s_i) = soft_max [ summa vc_j w(vc_j, loc(s_i)) * P(gc_j)
        vc_i = self.VC_HC.dot(vvis)
        self.s['VC'] = utils.softmax(vc_i, tau)

    def grid_input(self, p_gs):
        """Calculate input to HC from GS position estimate."""

        # P(s_i) ~ summa gs_j w_ji * P(gs_j), where w_ji = dist(gs_j, loc(s_i))
        gs_i = (self.GS_HC * p_gs).sum(axis=(1, 2))
        self.s['GS'] = gs_i / gs_i.sum()

    def integrate_inputs(self):
        """Integrate input from GS and VC."""

        both = np.array(self.s['GS'] * self.s['VC'])
        self.s['both'] = both / both.sum()

    def full_update(self, vvis, p_gs, vc_hc=True, gs_hc=True):
        """Update HC state estimate."""

        self.reset()   # remove resetting once HC has internal dynamics!
        if vc_hc and vvis is not None:
            self.visual_input(vvis)  # receive VC visual input (observations)
        if gs_hc and p_gs is not None:
            self.grid_input(p_gs)  # receive input from GS position estimate
        self.integrate_inputs()  # integrate two input streams

    def state_read_out(self):
        """
        Read-out of state estimate, where maximum value is accentuated by
        power function.
        """

        s_ro = utils.pow_read_out(self.s['both'], self.ro_pow)
        return s_ro

    def as_DF(self):
        """Return location estimate as Pandas DataFrame."""

        hc_df = pd.DataFrame(self.s, index=self.s_names)
        hc_df.columns.name = 'type'
        hc_df.index.name = 's'

        return hc_df


# %% GS

class GridSystem:
    """Medial enthorinal cortex grid cell system model."""

    def __init__(self, xmin, xmax, nx, xvec, ymin, ymax, ny, yvec, circular,
                 activ_sprd, mot_sig, **kwargs):
        """Init Grid System."""

        # Params.
        # x axis: columns.
        self.xmin = xmin
        self.xmax = xmax
        self.nx = nx
        self.xvec = xvec
        self.xres = (xmax - xmin) / (nx-1)

        # y axis: rows.
        self.ymin = ymin
        self.ymax = ymax
        self.ny = ny
        self.yvec = yvec
        self.yres = (ymax - ymin) / (ny-1)

        # Motor noise level for internally generated motor commands (at IBP).
        self.int_mot_sig = mot_sig

        # Minimum activity in each position.
        self.min_activ = 1e-4 / (nx * ny)

        # Is activity shift circular or it falls off of grid?
        self.circular = circular

        # Activity spreading parameters.
        self.n_std = 2  # number of STDs to go from centre
        self.activ_sprd = activ_sprd  # magnitude of activity spread
        self.conv_filter = None
        if activ_sprd != 0:
            xs = activ_sprd / self.xres  # STD relative to resolution
            ys = activ_sprd / self.yres
            xd = int(np.ceil(self.n_std * xs))  # number of units to convolve
            yd = int(np.ceil(self.n_std * ys))  # with in each direction
            self.conv_filter = np.outer(*[norm.pdf(np.arange(-d, d+1), 0, s)
                                          for d, s in [(xd, xs), (yd, ys)]])
            self.conv_filter = self.conv_filter / self.conv_filter.sum()

        # Position representation.
        self.P = np.zeros((ny, nx))

    def reset(self, var=0.1):
        """Initialize / reset activity to uniform with some added noise."""

        self.P = 1 / (self.nx * self.ny)  # uniform value
        if var != 0:
            self.P *= (1 + var * utils.rand_mat(self.ny, self.nx, 2))
            self.P = self.P / self.P.sum()  # PD over all positions

    def reset_to_pos(self, xpos, ypos):
        """Initialize / reset activity to position with some variance."""

        M = utils.dist_mat(xpos, ypos, self.xvec, self.yvec, self.circular)
        self.P = norm.pdf(M, scale=self.activ_sprd)
        self.P = self.P / self.P.sum()  # PD over all positions

    def pass_through_lateral_conn(self):
        """
        Simulate spreading of activity via lateral connectivity by convolution
        with 2D Gaussian filter. This step increases entropy.
        """

        if self.conv_filter is not None:
            boundary = 'wrap' if self.circular else 'fill'
            self.P = convolve2d(self.P, self.conv_filter, 'same', boundary)

        self.P = self.P / self.P.sum()  # rescale to PD

    def motor_update(self, vmot):
        """Update position estimate with (noisy) motor input."""

        # Weighted shift, with shifted GS activity spread across neighbouring
        # units proportional to their distance to the "hypothetical" shifted
        # activity of each unit.
        # Has natural tendency to dissolve peak activity (increase population
        # entropy), but free from rounding errors.

        vx, vy = vmot

        # Get lower amount of shift along axes.
        xshf = int(np.floor(vx / self.xres))
        yshf = int(np.floor(vy / self.yres))

        # Get weights of shifts.
        wx = 1 - (vx - xshf * self.xres) / self.xres
        wy = 1 - (vy - yshf * self.yres) / self.yres

        # Get four shift configurations (lower or upper along each axis).
        shfts = [(xshf+ix, yshf+iy, abs(ix-wx) * abs(iy-wy))
                 for ix in [0, 1] for iy in [0, 1]]

        # Get weighted sum of four shifted GS activity matrices.
        # This step increases entropy quite a bit, especially if shifted
        # activity has to be split equally among inheriting units (i.e. falls
        # to the center of the grid).
        # If activity shifting is circular, entropy is relatively stable,
        # otherwise it drops massively if activity falls off of grid!
        self.P = sum([w * utils.shift(self.P, xshf, yshf, self.circular, 0)
                      for xshf, yshf, w in shfts if w != 0])

        # Introduce minimum activity to allow all units to have non-zero
        # probability and be able to receiving HC feedback.
        # This step increases entropy, but only very slightly.
        self.P[self.P < self.min_activ] = self.min_activ
        self.P = self.P / self.P.sum()

        self.pass_through_lateral_conn()

    def HC_update(self, GS_HC, hc_ro):
        """Update position estimate with HC feedback."""

        # Backward inference using GS - HC connectivity (generative model).
        # Weighted mean of GS - HC connectivity, with HC estimate as weights.
        hc_fb = np.average(GS_HC, 0, hc_ro)
        hc_fb = hc_fb / hc_fb.sum()

        self.P = hc_fb * self.P
        self.pass_through_lateral_conn()

    def as_DF(self):
        """Return position estimate as Pandas DataFrame."""

        gs_df = pd.DataFrame(self.P, columns=self.xvec, index=self.yvec)
        gs_df.columns.name = 'x'
        gs_df.index.name = 'y'

        return gs_df


# %% VC

class VisualCortex:
    """Visual cortex model."""

    def __init__(self, features, vis_alpha, vis_beta, **kws):
        """Init VC."""

        # Noise level parameters for scene reconstruction.
        self.alpha = vis_alpha
        self.beta = vis_beta

        # Visual feature names.
        self.features = features

        # Activity vector representing values of visual features of scene.
        self.v = np.zeros(len(features))

    def reset(self):
        """Reset activity."""

        self.v = np.zeros(len(self.features))

    def update(self, v_input):
        """Update (set) activity by bottom-up input."""

        self.v = v_input

    def reconstr_scene(self, HC_VC, hc_ro):
        """Reconstruct visual scene using HC location estimate input."""

        v = HC_VC.dot(hc_ro)  # pass HC input through connectivity
        self.v = utils.noisy_vis_input(v, self.alpha, self.beta)  # add noise

    def as_Ser(self,):
        """Return visual feature values as Pandas Series."""
        return pd.Series(self.v, index=self.features)


# %% PFC

class PrefrontalCortex:
    """
    Prefrontal Cortex model, performing a working memory and controller
    function (scene understanding, action sampling, policy selection, etc).
    """

    def __init__(self, n_seq, n_ahead, u_soft_tau, gamma, **kws):
        """Init PFC."""

        # Parameters.
        self.n_seq = n_seq    # number of traj. sequences that can be stored
        self.n_ahead = n_ahead  # max number of steps in each traj. sequence
        self.u_soft_tau = u_soft_tau  # read-out power of action probabilities
        self.gamma = gamma   # discount factor for evaluating action sequences

        # Variables stored in working memory.
        self.plan = []  # current action plan (list of actions) to execute
        self.hist = []  # history of recently performed actions

        # Variables for IBP.
        self.u_seq = []  # list of potential (sampled) action plans
        self.r_exp = []  # expected accumulated reward of sampled action plans

    def reset(self):
        """Reset state."""

        self.reset_ibp()
        self.reset_wm()

    def reset_wm(self):
        """Reset variables in working memory."""

        self.plan = []
        self.hist = []

    def reset_ibp(self):
        """Reset IBP-related variables."""

        self.u_seq = []
        self.r_exp = []

    def new_sample(self):
        """Start new action sequence sample."""

        self.u_seq.append([])
        self.r_exp.append(0)

    def sample_action(self, vfeatures, vvis, opp_u, U):
        """Get new action sample from visual scene."""

        # Sample action.
        p_u = self.get_p_action(vfeatures, vvis, opp_u)
        u_samp = np.random.choice(U, p=p_u)

        return u_samp

    def store_action_in_wm(self, u):
        """Store action in WM into last sample."""

        self.u_seq[-1].append(u)

    def get_p_action(self, vfeat, vvis, opp_u):
        """Return probability of sampling of each action."""

        # Interpret scene: get probability of each action.
        # Convert visual features from range -1 (wall) to +1 (corridor) to
        # probability distribution of action executabilities.
        p_u = utils.softmax(vvis, self.u_soft_tau)
        U = vfeat  # comes from simple visual model (features == actions)

        # Prevent taking opposite of previous action to improve IBP.
        if opp_u is not None and len(self.u_seq) and len(self.u_seq[-1]):
            u_opp = opp_u[self.u_seq[-1][-1]]
            p_u[U.index(u_opp)] = 0
            p_u /= p_u.sum()

        return p_u

    def update_sample_reward(self, r):
        """Update expected reward of currently sampled action sequence."""
        self.r_exp[-1] += r * self.gamma ** (len(self.u_seq[-1])-1)

    def select_plan(self):
        """Select best policy (action sequence) from sampled sequences."""

        i_best = self.r_exp.index(max(self.r_exp))
        self.plan = self.u_seq[i_best]

    def get_next_action(self):
        """Return next action from action plan, move plan forward."""

        u = self.plan.pop(0)
        return u

    def update_action_history(self, u):
        """Update action history."""

        self.hist.append(u)


# %% vS

class VentralStriatum:
    """
    Ventral Striatum model, learning and providing reward expectation for each
    state.
    """

    def __init__(self, ns, alpha, **kws):
        """Init vS."""

        # Params.
        self.ns = ns  # number of states to represent reward for
        self.alpha = alpha  # learning rate

        # Reward expectation function (state - reward mapping).
        self.r = np.zeros(ns, dtype=float)

    def update(self, s, r):
        """
        Update expected reward at state estimate s (PD over all states).
        """
        self.r += self.alpha * s * (r - self.r)

    def reward_estim(self, s):
        """
        Return reward estimate at given state estimate s (PD over all states).
        """
        r_exp = s.dot(self.r)
        return r_exp

    def as_Ser(self, s_names):
        """Return reward expectations as Pandas Series."""
        return pd.Series(self.r, index=s_names)


# %% dlS

class DorsolateralStriatum:
    """
    Dorsolateral Striatum model, learning and functioning as the habitual
    state-action mapping system.
    """

    def __init__(self, ns, nu, tau, alpha, gamma, **kws):
        """Init dlS."""

        # Params.
        self.tau = tau       # softmax tau for action selection
        self.alpha = alpha   # learning rate
        self.gamma = gamma   # discount factor

        # Q: State - action - value mapping matrix. More rewarding states are
        # more likely to be selected during HAS navigation.
        self.Q = np.zeros((nu, ns), dtype=float)

    def get_discounted_r(self, r, i):
        """Return discounted value of r from i steps away."""
        return self.gamma**i * r

    def habit_update(self, s, u_idx, r_disc):
        """Update Q state-action habit function with reward."""

        r_exp = s.dot(self.Q[u_idx])      # expected reward of state - action
        v_upd = self.alpha * (r_disc - r_exp)  # value update
        self.Q[u_idx] += v_upd * s        # update by weighted state estimate

    def action_values(self, s):
        """Return action values at given state estimate s."""
        return self.Q.dot(s)

    def habit_action(self, s, U):
        """Sample an action from action-value PD at given state estimate s."""

        q_vals = self.action_values(s)
        # u = U[q_vals.argmax()]  # could just select maximum with this
        p_u = utils.softmax(q_vals, self.tau)
        u = np.random.choice(U, p=p_u)

        return u

    def as_DF(self, s_names, u_names):
        """Return reward expectations as Pandas DataFrame."""
        return pd.DataFrame(self.Q, index=u_names, columns=s_names)
