import Boat
import math
import numpy as np
import copy
import Strategies
import Metrics
import Polygon as poly
import Polygon.Utils as polyUtils



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
        self._defenseMetric = None

    @property
    def defenseMetric(self):
        return self._defenseMetric

    @defenseMetric.setter
    def defenseMetric(self, defenseMetric_in):
        self._defenseMetric = defenseMetric_in

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

        T = self._defenseMetric.timeThreshold

        # where will attackers be in T seconds? assume straight line constant velocity
        for attacker in self._attackers:
            if not attacker.hasBeenTargeted:
                x0 = attacker.state[0]
                y0 = attacker.state[1]
                u = attacker.state[2]
                th = attacker.state[4]
                thdot = attacker.state[5]
                x1 = x0 + u*np.cos(th)*T
                y1 = y0 + u*np.sin(th)*T
                #r = np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))
                #x1 += thdot*r*np.cos(th + np.pi/2.)*T
                #y1 += thdot*r*np.sin(th + np.pi/2.)*T
                # check if this is in any defender polygons
                defenders_who_can_intercept = list()
                defenders_who_are_not_busy = list()
                difficulty_of_defense = list()  # how difficult it will be to intercept
                for defender in self._defenders:
                    polygon = defender.TTAPolygon
                    if polygon.isInside(x1, y1) and np.abs(thdot) < np.deg2rad(1.0):  # attacker is on a straight line
                        defenders_who_can_intercept.append(defender)
                        heading_to_intercept = np.arctan2(y1 - defender.state[1], x1 - defender.state[0])
                        difficulty_of_defense.append(np.abs(defender.state[4] - heading_to_intercept))
                        if not defender.busy:
                            defenders_who_are_not_busy.append(defender)

                # assign best defender
                if defenders_who_can_intercept != []:
                    # sorted defenders
                    sorted_defenders = [defenders_who_can_intercept[d] for d in np.argsort(difficulty_of_defense)]
                    for defender in sorted_defenders:
                        if defender in defenders_who_are_not_busy:
                            defender.strategy = Strategies.DestinationOnly(defender, [copy.deepcopy(x1), copy.deepcopy(y1)])
                            defender.busy = True
                            attacker.hasBeenTargeted = True
                            break
                else:
                    if np.sqrt(np.power(x1 - self.assets[0].state[0], 2) + np.power(self.assets[0].state[1], 2)) < 10.0:
                        print "WARNING: no defenders can intercept an attacker!"





        return

    def updateAsset(self):
        return