#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Definition of village (navigation environment or maze).

@author: David Samu
"""

from itertools import product, combinations

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from mb_mf_nav import utils


# %% Base environment class
# -------------------------

class Environment:
    """Base class of discrete environments defining shared operations."""

    def __init__(self, res=1):

        # Data fields.
        self.res = None    # spatial resolution
        self.S = None      # names of states (list)
        self.S_pos = None  # positions of states (dict)
        self.U = None      # names of actions (list)
        self.U_dir = None  # action -> motor movement function (dict)
        self.U_opp = None  # action -> action to opposite direction (dict)
        self.sus = None    # state, action -> state function (dict)
        self.vfeatures = None  # names of visual features (list)
        self.V = None      # state -> visual observation (dict)
        self.R = None      # state -> rewards function (dict)
        self.animal_artist = None  # animal artist to draw on canvas

        # Data fields treating environment as a graph.
        self.G = None      # NetworkX graph representation of environment
        self.spl = None    # shortest path lengths between state pairs (dict)
        self.diam = None   # diameter of graph: longest of the shortest paths

        # Actions (controls).
        self.U_dir = {'n': (0, 1), 'ne': (1, 1), 'e': (1, 0),
                      'se': (1, -1), 's': (0, -1), 'sw': (-1, -1),
                      'w': (-1, 0), 'nw': (-1, 1)}
        self.U_dir = {u: res * np.array(self.U_dir[u]) for u in self.U_dir}
        self.U = [u for u in self.U_dir]

        # To prevent back-tracking along trajectory.
        self.U_opp = {u: self.get_opp_dir(u) for u in self.U_dir}

    # -------------------------------------------#
    # Functions to set up or modify environment. #
    # -------------------------------------------#

    def setup_observations(self):
        """Set up observation vector at each locations."""

        # Location-dependent observation vector: binary vector to each
        # direction of whether moving to each way is possible. Rudimentary
        # model of 360 visual field with known self-orientation (provided
        # by e.g. head direction cells).
        self.vfeatures = self.U
        V = {s: np.array([u in self.poss_u(s) for u in self.U], dtype=int)
             for s in self.S}

        # Convert 0/1 encoding to -1/+1: sign of value codes whether wall
        # or corridor is expected at given direction at a given location.
        # Because set of visual features "overlap" among locations, such
        # competition is necessary between presence / absence of features
        # (otherwise noise would provide weak evidence to each feature,
        # making locations with more features more probable).
        self.V = {s: 2*v-1 for s, v in V.items()}

    def setup_graph_structures(self):
        """Set up graph representation and related data fields of environ."""

        self.G = nx.Graph(self.paths())
        self.spl = dict(nx.all_pairs_shortest_path_length(self.G))
        self.diam = nx.diameter(self.G)

    def add_path(self, s1, s2):
        """Add new path to environment."""

        # Does path already exist?
        if self.check_path(s1, s2):
            print('Path [{} - {}] already exist!'.format(s1, s2))
            return

        # Is path physically possible?
        u = self.get_u(s1, s2)
        if u is None:
            print('No action possible to connect {} and {}!'.format(s1, s2))
            return
        # TODO: Could also check for crossing paths!

        # Add path to transition function.
        self.sus[s1][u] = s2
        self.sus[s2][self.get_opp_dir(u)] = s1

        # Update observations per location and graph data structures.
        self.setup_observations()
        self.setup_graph_structures()

    def remove_path(self, s1, s2):
        """Remove existing path from environment."""

        # Is path in state transition function?
        if not self.check_path(s1, s2):
            print('Path [{} - {}] does not exist!'.format(s1, s2))
            return

        # Remove path from state transition function.
        for s_pre, s_post in [(s1, s2), (s2, s1)]:
            u = utils.find_key(self.sus[s_pre], s_post)
            del self.sus[s_pre][u]

        # Update observations per location and graph data structures.
        self.setup_observations()
        self.setup_graph_structures()

    def change_reward(self, loc, r_new):
        """Change reward at given location."""
        self.R[loc] = r_new

    # ------------------------------#
    # Functions to perform queries. #
    # ------------------------------#

    def get_dimensions(self):
        """Return limits and size along both dimenstions."""

        pos = np.array(list(self.S_pos.values())).squeeze()
        xmax, ymax = pos.max(axis=0)
        xmin, ymin = pos.min(axis=0)
        xsize = xmax - xmin
        ysize = ymax - ymin

        env_dims = dict(xmin=xmin, xmax=xmax, xsize=xsize,
                        ymin=ymin, ymax=ymax, ysize=ysize,
                        res=self.res)

        return env_dims

    def paths(self):
        """Return all (unique) paths as (s1, s2) pairs."""

        if self.S is None or self.sus is None:
            return []

        paths = [(s1, s2) for s1, s2 in combinations(self.S, 2)
                 if s2 in self.sus[s1].values()]

        return paths

    def check_path(self, s1, s2):
        """Check if path exists or not."""

        all_paths = self.paths()
        exists = (s1, s2) in all_paths or (s2, s1) in all_paths
        return exists

    def n_paths(self):
        """Return number of paths in environment."""

        n_paths = len(self.paths())
        return n_paths

    def dist(self, s1, s2):
        """Return distance between two locations."""

        return np.linalg.norm(self.S_pos[s1]-self.S_pos[s2])

    def get_u(self, s1, s2):
        """
        Return action leading from state s1 to s2. Path may in fact not exist!
        """

        u = utils.find_key(self.U_dir, self.S_pos[s2]-self.S_pos[s1])
        return u

    def max_r_state(self):
        """Return state with maximum reward."""

        s_rmax = max(self.R, key=self.R.get)
        return s_rmax

    def pos_r_states(self):
        """Return states with some positive reward."""

        s_rpos = [s for s in self.R if self.R[s] > 0]
        return s_rpos

    # ----------------------------------#
    # Functions to perform simulations. #
    # ----------------------------------#

    def observation(self, vis_alpha, vis_beta):
        """
        Return both precise and noisy observation of animal from its current
        locations.
        """

        ovis = self.V[self.s]  # precise visual input
        vvis = utils.noisy_vis_input(ovis, vis_alpha, vis_beta)  # add noise
        return ovis, vvis

    def motor_feedback(self, u, mot_sig):
        """Return precise and noisy motor feedback to action."""

        umot = self.U_dir[u]   # proprioceptive motor feedback
        vmot = utils.noisy_mot_input(umot, mot_sig)  # add noise
        return umot, vmot

    def poss_u(self, s=None, excl_return=False, excl_sx=None):
        """Return list of possible actions from position s."""

        if s is None:
            s = self.s

        # Init exclude states.
        if excl_sx is None:
            excl_sx = []
        if excl_return:
            excl_sx.append(self.s_prev)

        # Go through all actions and find executable ones.
        u_poss = [u for u, sx in self.sus[s].items()
                  if sx not in excl_sx]

        return u_poss

    def optimal_u(self, s=None, s_goal=None, **poss_u_kws):
        """Return list of optimal actions from position s."""

        # Init start and goal locations.
        if s is None:
            s = self.s
        if s_goal is None:
            s_goal = self.max_r_state()

        # Select actions from all possible actions that get closest to goal.
        u_poss = self.poss_u(**poss_u_kws)
        u_nstep = {u: self.spl[self.sus[s][u]][s_goal] for u in u_poss}
        u_opt = [u for u in u_poss if u_nstep[u] == min(u_nstep.values())]

        return u_opt

    def replace_animal(self, loc):
        """Replace animal to location loc."""

        self.s = loc
        self.s_prev = None

    def move_animal(self, u):
        """Move animal with action u, return subsequent state and reward."""

        r = 0
        if u in self.sus[self.s]:
            self.s_prev = self.s
            self.s = self.sus[self.s][u]
            r = self.R[self.s]
        else:
            pass
            # print('\t\tvillage: invalid action: {} - {}'.format(self.s, u))

        return self.s, r

    def animal_coords(self):
        """Return coordinates of animal."""

        return self.S_pos[self.s]

    def init_animal_artist(self):
        """Locate and set up artist of animal."""

        self.animal_artist = plt.Circle([0, 0], 0.1*self.res, alpha=0.4,
                                        color='r', ec='r', zorder=99)
        self.animal_artist.set_clip_on(False)

    def redraw_animal(self):
        """Redraw animal (move marker artist) after animal has moved."""

        self.animal_artist.center = self.animal_coords()

    # -------------------------------#
    # Functions to perform analysis. #
    # -------------------------------#

    def get_opp_dir(self, u):
        """Return action which is opposite direction to the one passed."""

        x_opp, y_opp = -self.U_dir[u]

        # Assuming there is only one opposite direction.
        for ui, udir in self.U_dir.items():
            if udir[0] == x_opp and udir[1] == y_opp:
                return ui

        return None

    def follow_path(self, u_seq, s=None, gamma=1):
        """Return end loc from location loc along action sequence u_seq."""

        if s is None:
            s = self.s

        r = 0
        for i, u in enumerate(u_seq):
            if u in self.sus[s]:
                s = self.sus[s][u]
                r += self.R[s] * gamma ** i  # accumulate discounted reward
            else:  # invalid action sequence
                s = None
                break

        return s, r


# %% Randomly generated discrete environment
# ------------------------------------------

class RandomEnv(Environment):
    """Randomly generated navigation environment."""

    def __init__(self, nx, ny, p_state, p_path, keep_crossing_paths=False,
                 keep_islands=False, keep_cul_de_sac=False, res=100, **kws):
        """Init random environment."""

        super().__init__(res)

        # Register params.
        self.nx = nx
        self.ny = ny
        self.p_state = p_state
        self.p_path = p_path
        self.keep_crossing_paths = keep_crossing_paths
        self.keep_islands = keep_islands
        self.keep_cul_de_sac = keep_cul_de_sac
        self.res = res

        # States.
        S = {'s_'+str(xi)+'_'+str(yi): (xi, yi)            # all combination
             for xi, yi in product(range(nx), range(ny))}  # of states
        sk = np.random.choice(list(S.keys()), round(p_state*nx*ny),
                              replace=False)
        S = {k: S[k] for k in sk}  # select n states randomly
        self.S_pos = {s: res * np.array(S[s]) for s in S}  # state positions
        self.S = sorted([s for s in S])      # state names

        # State transition function: s -> u -> s' dict of dict
        # Generate full (valid) state transition function of selected states.
        loc_s = {tuple(loc): s for s, loc in self.S_pos.items()}
        sus = {s: {u: tuple(self.S_pos[s]+self.U_dir[u]) for u in self.U}
               for s in self.S}   # next position after taking u at s
        sus = {s: {u: loc_s[sus[s][u]] for u in sus[s] if sus[s][u] in loc_s}
               for s in sus}  # next state, if state exist
        sus = {s: us for s, us in sus.items() if len(us)}  # remove isolated

        # Subsample paths randomly.
        ssx_all = utils.flatten([[(s, sx) for u, sx in us.items() if s < sx]
                                 for s, us in sus.items()])
        i_ssx_sel = np.random.choice(range(len(ssx_all)),
                                     round(p_path*len(ssx_all)), replace=False)
        ssx_sel = [ssx_all[i] for i in i_ssx_sel]
        sus = {s: {u: sx for u, sx in sus[s].items()
               if (s, sx) in ssx_sel or (sx, s) in ssx_sel}
               for s in sus}  # next state, if state exist
        sus = {s: us for s, us in sus.items() if len(us)}  # remove isolated

        # Remove crossing paths.
        if not keep_crossing_paths:
            # Currently done rather simplistically assuming naming pattern!!
            for s in sus:
                x, y = [int(v/self.res) for v in self.S_pos[s]]
                for dx, dy in product([-1, 1], repeat=2):
                    # Get states that can form crossing paths to given dir.
                    sx = 's_{}_{}'.format(x+dx, y+dy)
                    sc = 's_{}_{}'.format(x+dx, y)
                    scx = 's_{}_{}'.format(x, y+dy)
                    if (sx in sus[s].values() and sc in sus and
                        scx in sus[sc].values()):
                        # Select one of the crossing paths randomly.
                        irem = np.random.choice([0, 1])
                        sr, sxr = [(s, sx), (sc, scx)][irem]
                        # Remove path along both directions.
                        sus[sr] = {u: sx for u, sx in sus[sr].items()
                                   if sx != sxr}
                        sus[sxr] = {u: sx for u, sx in sus[sxr].items()
                                    if sx != sr}
            sus = {s: us for s, us in sus.items() if len(us)}

        # Remove isolated islands of states.
        if not keep_islands:
            # Get all islands.
            islands = [set([s] + list(sus[s].values())) for s in sus]
            merged = True
            while merged:
                merged = False
                for i, j in combinations(range(len(islands)), 2):
                    if islands[i] & islands[j]:
                        islands.append(islands[i] | islands[j])
                        del islands[j]
                        del islands[i]
                        merged = True
                        break
            # Select largest island.
            len_islands = {len(island): island for island in islands}
            largest = len_islands[max(len_islands)]
            # Keep only states of largest island.
            sus = {s: sus[s] for s in sus if s in largest}

        # Remove states that are dead ends (cul-de-sacs).
        # Careful: This can remove potentially all states if p_path is too low!
        if not keep_cul_de_sac:
            while True:  # have to go iteratively and remove new cds states
                s_cds = [s for s in sus if len(sus[s]) <= 1]
                if not len(s_cds):  # no more cds to remove
                    break
                sus = {s: {u: sx for u, sx in sus[s].items()
                           if sx not in s_cds} for s in sus if s not in s_cds}
        self.sus = sus  # final state transition function

        # Remove states with no connections.
        self.S_pos = {s: p for s, p in self.S_pos.items() if s in self.sus}
        self.S = [s for s in self.S if s in self.sus]

        # Set up observations per location and graph data structures.
        self.setup_observations()
        self.setup_graph_structures()

        # Reward per location.
        self.R = {loc: 1 if loc == self.S[len(self.S)-1] else 0
                  for loc in self.S}

        # Animal-related variables.
        self.s = self.S[0]
        self.init_animal_artist()


# %% Manually-defined discrete environments
# -----------------------------------------

class Village(Environment):
    """
    Village navigation environment.

    Implements the village used in Winocur et al, Hippocampus, 2010.
    """

    def __init__(self, res=100):
        """Init village."""

        super().__init__(res)

        self.res = res

        # Locations (states).
        S = {'N': (2, 4), 'E': (4, 2), 'S': (2, 0), 'W': (0, 2), 'C': (2, 2),
             'water': (1, 3), 'toys': (3, 3), 'food': (3, 1), 'peer': (1, 1),
             'CN': (2, 3), 'CE': (3, 2), 'CS': (2, 1), 'CW': (1, 2)}
        self.S_pos = {loc: res * np.array(S[loc]) for loc in S}
        self.S = [loc for loc in self.S_pos]

        # State transistion function: s -> u -> s' dict of dict
        self.sus = {'N': {'sw': 'water', 's': 'CN', 'se': 'toys'},
                    'E': {'nw': 'toys', 'w': 'CE', 'sw': 'food'},
                    'S': {'nw': 'peer', 'n': 'CS', 'ne': 'food'},
                    'W': {'ne': 'water', 'e': 'CW', 'se': 'peer'},
                    'water': {'sw': 'W', 'ne': 'N'},
                    'toys': {'nw': 'N', 'se': 'E'},
                    'food': {'sw': 'S', 'ne': 'E'},
                    'peer': {'nw': 'W', 'se': 'S'},
                    'CN': {'n': 'N', 'sw': 'CW', 's': 'C', 'se': 'CE'},
                    'CE': {'e': 'E', 'nw': 'CN', 'w': 'C', 'sw': 'CS'},
                    'CS': {'s': 'S', 'nw': 'CW', 'n': 'C', 'ne': 'CE'},
                    'CW': {'w': 'W', 'ne': 'CN', 'e': 'C', 'se': 'CS'},
                    'C': {'n': 'CN', 'e': 'CE', 's': 'CS', 'w': 'CW'}}

        # Set up observations per location and graph data structures.
        self.setup_observations()
        self.setup_graph_structures()

        # Reward (utility).
        self.R = {loc: 0 for loc in self.S}
        # rwds = [('food', 2), ('water', 1), ('peer', 0), ('toys', -3)]
        # rwds = [('food', 5), ('water', 1)]
        rwds = [('food', 5)]
        for loc, r in rwds:
            self.R[loc] = r

        # Animal-related variables.
        self.s = 'C'
        self.init_animal_artist()
