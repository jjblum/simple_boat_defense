import Boat
import math
import numpy
import Strategies


class Team(object):
    """
        Class that allows the broadcast of strategies to several cooperative boats
        It allows more intelligent assignments within a team. For example,
        if you want the defenders to get in a formation around the asset, you should
        assign their spots based on who is closest to each spot.
    """
    def __init__(self, boat_list):
        self._teamMembers = boat_list

    def delegate(self):
        for boat in self._teamMembers:
            # boat.strategy =
            None


class Overseer(object):
    """
        Class that produces strategies for the boats based on the state of the system.
        Focuses on offense and defensive tactics.
        Limiting the amount of information an Overseer has can affect system performance.
    """
    def __init__(self, assets, defenders, attackers):
        self._assets = assets
        self._attackers = attackers
        self._defenders = defenders
        self._teams = list()
        self._atk_vs_asset_pairwise_distances = None
        self._def_vs_atk_pairwise_distances = None

    @property
    def atk_vs_asset_pairwise_distances(self):
        return self._atk_vs_asset_pairwise_distances

    @atk_vs_asset_pairwise_distances.setter
    def atk_vs_asset_pairwise_distances(self, atk_vs_asset_pairwise_distances_in):
        self._atk_vs_asset_pairwise_distances = atk_vs_asset_pairwise_distances_in

    @property
    def def_vs_atk_pairwise_distances(self):
        return self._def_vs_atk_pairwise_distances

    @def_vs_atk_pairwise_distances.setter
    def def_vs_atk_pairwise_distances(self, def_vs_atk_pairwise_distances_in):
        self._def_vs_atk_pairwise_distances = def_vs_atk_pairwise_distances_in

    @property
    def attackers(self):
        return self._attackers

    @attackers.setter
    def attackers(self, attackers_in):
        self._attackers = attackers_in

    @property
    def defenders(self):
        return self._defenders

    @defenders.setter
    def defenders(self, defenders_in):
        self._defenders = defenders_in

    @property
    def assets(self):
        return self._assets

    @assets.setter
    def assets(self, assets_in):
        self._assets = assets_in

    def updateAttack(self):
        return

    def updateDefense(self):
        # e.g. if attackers get within a certain distance of the asset, assign the closest defender to intercept
        return

    def updateAsset(self):
        return