import Boat
import math
import numpy as np
import copy
import Strategies
import Metrics
import Polygon as poly
import Polygon.Utils as polyUtils
import matplotlib.pyplot as plt


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


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

    def update(self):
        self.updateDefense()
        self.updateAttack()

    def updateAttack(self):
        asset = self._assets[0]
        ass_x = asset.state[0]
        ass_y = asset.state[1]
        for attacker in self._attackers:
            if attacker.hasBeenTargeted:
                distance_to_interception = np.sqrt(np.power(attacker.pointOfInterception[0] - attacker.state[0], 2) +
                                                   np.power(attacker.pointOfInterception[1] - attacker.state[1], 2))
                distance_to_asset = np.sqrt(np.power(ass_x - attacker.state[0], 2) + np.power(ass_y - attacker.state[1], 2))
                if distance_to_asset > distance_to_interception and distance_to_interception < 10.0:
                    # if you can't hit the asset before interception AND the defender is within X meters
                    attacker.strategy = Strategies.TimedStrategySequence(attacker, [
                        (Strategies.Circle_LOS, (attacker, [ass_x, ass_y], distance_to_asset-3.0, "ccw", attacker.design.maxSpeed)),
                        (Strategies.MoveTowardAsset, (attacker,))
                    ], [np.random.uniform(5.0, 30.0), 999.0])
        return

    def updateDefense(self):
        # e.g. if attackers get within a certain distance of the asset, assign the closest defender to intercept
        T = self._defenseMetric.timeThreshold

        for defender in self._defenders:
            if defender.target not in self._attackers:  # update defenders that have removed their targets
                defender.busy = False
                defender.target = None
                defender.strategy = Strategies.Circle_LOS(defender, [0., 0.], 10.0, surgeVelocity=2.5)

        # TODO - need a simple interception alternative for when the boats are close and an interception can happen easily


        # where will attackers be in T seconds? assume straight line constant velocity
        for attacker in self._attackers:
            x0 = attacker.state[0]
            y0 = attacker.state[1]
            u = np.round(attacker.state[2], 3)
            th = wrapToPi(attacker.state[4])
            thdot = attacker.state[5]
            x1 = x0 + u*np.cos(th)*np.array(T)
            y1 = y0 + u*np.sin(th)*np.array(T)
            #r = np.sqrt(np.power(x1 - x0, 2) + np.power(y1 - y0, 2))
            #x1 += thdot*r*np.cos(th + np.pi/2.)*T
            #y1 += thdot*r*np.sin(th + np.pi/2.)*T

            if attacker.hasBeenTargeted:
                # determine if target is not on an intercept course anymore
                if type(attacker.targetedBy.strategy) != Strategies.MoveTowardBoat:
                    if np.abs(th - np.arctan2(attacker.pointOfInterception[1] - y0, attacker.pointOfInterception[0] - x0)) > np.deg2rad(15.0):
                        attacker.hasBeenTargeted = False
                        attacker.targetedBy.busy = False
                        attacker.targetedBy.strategy = Strategies.Circle_Tracking(attacker.targetedBy, [0., 0.], attacker, radius_growth_rate=0.5)
                        attacker.targetedBy = None
                    else:
                        # TODO - update the intercept ?
                        None


            if not attacker.hasBeenTargeted:  # want to check if the re-intercept if statemenet caught anything
                # check if this is in any defender polygons
                defenders_who_can_intercept = list()
                defenders_who_are_not_busy = list()
                difficulty_of_defense = list()  # how difficult it will be to intercept
                for defender in self._defenders:
                    polygons = defender.TTAPolygon
                    for t in range(len(polygons)):
                        polygon = polygons[t]

                        #TODO: determine if for any time > contourTTA the attacker's projection will be in the contour
                        T_ = T[t]
                        #polygon_points = np.array(polyUtils.pointList(polygon))
                        #polygon_points = np.row_stack((polygon_points, polygon_points[0, :]))  # concatenate last point
                        #diff_list = np.diff(polygon_points, axis=0)
                        #polygon_segment_lengths = np.sqrt(np.power(diff_list[:, 0], 2) + np.power(diff_list[:, 1], 2))
                        #polygon_phis = np.arctan2(diff_list[:,1], diff_list[:, 0])
                        #polygon_points = polygon_points[:-1, :]  # remove that concatenated point
                        #for i in range(polygon_points.shape[0]):
                        #    A = np.array([[np.cos(polygon_phis[i]), -u*np.cos(th)], [np.sin(polygon_phis[i]), -u*np.sin(th)]])
                        #    b = np.array([[x0 - polygon_points[i, 0]],[y0 - polygon_points[i, 1]]])
                        #    alpha_dt = np.squeeze(np.dot(np.linalg.inv(A), b))
                        #    if alpha_dt[0] >= 0. and alpha_dt[0] <= polygon_segment_lengths[i]:
                        #        # feasible intercept on this contour line segment
                        #        if alpha_dt[1] >= T_:
                        #            # can intercept b/c the attacker will arrive after the defender can reach this point
                        #            plt.plot(polygon_points[:,0], polygon_points[:,1], 'r-')
                        #            plt.plot(polygon_points[i,0], polygon_points[i,1], 'ms')
                        #            plt.plot([polygon_points[i,0], polygon_points[i,0] + alpha_dt[0]*np.cos(polygon_phis[i])], [polygon_points[i,1], polygon_points[i,1] + alpha_dt[0]*np.sin(polygon_phis[i])], 'm-', linewidth=3.0)
                        #            plt.plot([x0,x0+alpha_dt[1]*u*np.cos(th)],[y0,y0+alpha_dt[1]*u*np.sin(th)],'b-')
                        #            #plt.show()
                        #            break

                        x1_ = x1[t]
                        y1_ = y1[t]
                        if polygon.isInside(x1_, y1_) and np.abs(thdot) < np.deg2rad(1.0):  # attacker is on a straight line
                            defenders_who_can_intercept.append(defender)
                            heading_to_intercept = np.arctan2(y1_ - defender.state[1], x1_ - defender.state[0])
                            difficulty_of_defense.append(np.abs(defender.state[4] - heading_to_intercept))
                            if not defender.busy:
                                defenders_who_are_not_busy.append(defender)
                                defender.pointOfInterception = [x1_, y1_]
                                print "Defender {} can intercept using the {} TTA contour".format(defender.uniqueID, T_)
                                break

                # assign best defender
                if len(defenders_who_can_intercept) > 0:
                    # sorted defenders
                    sorted_defenders = [defenders_who_can_intercept[d] for d in np.argsort(difficulty_of_defense)]
                    for defender in sorted_defenders:
                        if defender in defenders_who_are_not_busy:  # currently only defenders that aren't busy can intercept this attacker
                            #defender.strategy = Strategies.StrategySequence(defender, [
                            #    (Strategies.PointAtLocation, (defender, [copy.deepcopy(x1), copy.deepcopy(y1)])),
                            #    (Strategies.DestinationOnly, (defender, [copy.deepcopy(x1), copy.deepcopy(y1)]))
                            #])
                            x1_ = defender.pointOfInterception[0]
                            y1_ = defender.pointOfInterception[1]
                            defender.strategy = Strategies.DestinationOnlyExecutor(defender, [copy.deepcopy(x1_), copy.deepcopy(y1_)])
                            defender.busy = True
                            defender.target = attacker
                            defender.pointOfInterception = [copy.deepcopy(x1_), copy.deepcopy(y1_)]
                            attacker.hasBeenTargeted = True
                            attacker.targetedBy = defender
                            attacker.pointOfInterception = [copy.deepcopy(x1_), copy.deepcopy(y1_)]
                            break
                #else:
                    #if np.sqrt(np.power(x1 - self.assets[0].state[0], 2) + np.power(self.assets[0].state[1], 2)) < 10.0:
                        #print "WARNING: no defenders can intercept an attacker!"
                        #None





        return

    def updateAsset(self):
        return