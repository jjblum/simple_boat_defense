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


def absoluteAngleDifference(angle1, angle2):
    while angle1 < 0.:
        angle1 += 2*np.pi
    while angle2 < 0.:
        angle2 += 2*np.pi
    angle1 = np.mod(angle1, 2*np.pi)
    angle2 = np.mod(angle2, 2*np.pi)
    return np.abs(angle1 - angle2)



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
        self._thCoeff = 2.54832785865
        self._rCoeff = 0.401354269952
        self._u0Coeff = 0.0914788305811

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
            if attacker.hasBeenTargeted and not attacker.evading:
                distance_to_interception = np.sqrt(np.power(attacker.pointOfInterception[0] - attacker.state[0], 2) +
                                                   np.power(attacker.pointOfInterception[1] - attacker.state[1], 2))
                distance_to_asset = np.sqrt(np.power(ass_x - attacker.state[0], 2) + np.power(ass_y - attacker.state[1], 2))
                if distance_to_asset > distance_to_interception and distance_to_interception < 10.0:
                    # if you can't hit the asset before interception AND the defender is within X meters
                    attacker.evading = True
                    if np.random.uniform(0., 1.) < 0.5:
                        direction = "ccw"
                    else:
                        direction = "cw"
                    attacker.strategy = Strategies.Circle_LOS(attacker, [ass_x, ass_y], distance_to_asset-3.0, direction, attacker.design.maxSpeed)
                    #attacker.strategy = Strategies.TimedStrategySequence(attacker, [
                    #    (Strategies.Circle_LOS, (attacker, [ass_x, ass_y], distance_to_asset-3.0, direction, attacker.design.maxSpeed)),
                    #    (Strategies.MoveTowardAsset, (attacker,))
                    #], [np.random.uniform(5.0, 30.0), 999.0])
            if attacker.evading:
                if np.random.uniform(0., 1.) < 0.01:
                    attacker.evading = False
                    attacker.strategy = Strategies.MoveTowardAsset(attacker)

        return

    def updateDefense(self):
        asset = self._assets[0]
        for defender in self._defenders:
            if defender.target is not None:
                if defender.target not in self._attackers:  # update defenders that have removed their targets
                    defender.busy = False
                    defender.target.hasBeenTargeted = False
                    defender.target = None
                    defender.strategy = Strategies.Circle_LOS(defender, [0., 0.], 10.0, surgeVelocity=2.5)
                else:
                    phi = np.arctan2(defender.pointOfInterception[1] - defender.target.state[1], defender.pointOfInterception[0] - defender.target.state[0])
                    if absoluteAngleDifference(defender.target.state[4], phi) > np.rad2deg(15.) or defender.target.state[2] < 0.1:
                        # attacker is no longer headed to the intercept point (either by heading or slowing down)
                        defender.busy = False
                        defender.strategy = Strategies.Circle_Tracking(defender, [0., 0.], defender.target, radius_growth_rate=0.25)
                        defender.target.hasBeenTargeted = False
                        defender.target.pointOfInterception = None
                        defender.target = None
                        defender.pointOfInterception = None

        able_defenders = [defender for defender in self._defenders if not defender.busy]
        ND = len(able_defenders)
        defenders_X = np.zeros((ND, 2))
        defenders_th = np.zeros((ND,))
        defender_u = np.zeros((ND,))
        for i in range(ND):
            defender = self._defenders[i]
            defenders_X[i, 0] = defender.state[0]
            defenders_X[i, 1] = defender.state[1]
            defender_u[i] = defender.state[2]
            defenders_th[i] = defender.state[4]

        # TODO - need a simple interception alternative for when the boats are close and an interception can happen easily

        # where will attackers be in T seconds? assume straight line constant velocity
        for attacker in self._attackers:
            if not attacker.hasBeenTargeted:
                x0 = attacker.state[0]
                y0 = attacker.state[1]
                u = np.round(attacker.state[2], 3)
                th = wrapToPi(attacker.state[4])
                thdot = attacker.state[5]
                angleToAsset = attacker.globalAngleToBoat(asset)
                distance_to_asset = attacker.distanceToBoat(asset)
                if np.abs(angleToAsset - th) < 0.01 and u > 0.1 and np.abs(thdot) < np.deg2rad(5.0):
                    #print "Attacker {} is on a straight line intercept with the asset!".format(attacker.uniqueID)
                    # find the time when it will hit, assuming the asset isn't moving for now
                    dx = asset.state[0] - x0
                    dy = asset.state[1] - y0
                    tx = 1./(u*np.cos(th))*dx
                    ty = 1./(u*np.sin(th))*dy
                    # tx and ty will be very similar, so just average them
                    t_impact = (tx + ty)/2.
                    NG = 20
                    fraction = np.linspace(0., 1., NG)
                    discrete_intercept_line = np.column_stack((x0 + fraction*t_impact*u*np.cos(th), y0 + fraction*t_impact*u*np.sin(th)))
                    discrete_distances = np.fliplr(np.atleast_2d(fraction))*distance_to_asset
                    discrete_distances = discrete_distances.T
                    # starts at attacker, traverses in toward asset

                    # can any defenders that are not busy intercept on that line?
                    gridx, defx = np.meshgrid(discrete_intercept_line[:, 0], defenders_X[:, 0])
                    gridy, defy = np.meshgrid(discrete_intercept_line[:, 1], defenders_X[:, 1])
                    x_pairs = np.column_stack((np.ravel(defx), np.ravel(gridx)))
                    y_pairs = np.column_stack((np.ravel(defy), np.ravel(gridy)))
                    # each NG rows are for a single defender
                    dx = x_pairs[:, 1] - x_pairs[:, 0]
                    dy = y_pairs[:, 1] - y_pairs[:, 0]
                    R = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
                    global_angle = np.arctan2(dy, dx)
                    global_angle[global_angle < 0.] += 2*np.pi  # must be on [0, 2*pi] interval
                    defenders_th[defenders_th < 0.] += 2*np.pi
                    local_angle = global_angle - np.repeat(defenders_th, NG, axis=0)  # the TTA model only uses positive theta!
                    local_angle = np.abs(wrapToPi(local_angle))  # must be on the [-pi, pi] interval
                    TTA = self._thCoeff*local_angle + self._rCoeff*R + self._u0Coeff*np.repeat(defender_u, NG, axis=0)
                    # remember, each NG rows are for a single defender -- TTA shape is (ND*NG,)
                    TTA_by_defender = np.reshape(TTA, (ND, NG))

                    ######
                    REQUIRED_TIME_BUFFER = 3.0  # extra seconds!
                    MAX_ALLOWABLE_INTERCEPT_DISTANCE = 10.0
                    ######
                    able_to_intercept = np.logical_and(
                            (np.repeat(np.atleast_2d(fraction*t_impact), ND, axis=0) - TTA_by_defender) > REQUIRED_TIME_BUFFER,
                            np.repeat(np.atleast_2d(discrete_distances.T), ND, axis=0) < MAX_ALLOWABLE_INTERCEPT_DISTANCE
                    )
                    defender_TTA_dict = dict()  # defender boat object: (where it can intercept, maximum distance from asset it can intercept)
                    max_intercept_distances = list()
                    for i in range(ND):
                        if np.any(able_to_intercept[i, :]):
                            defender_max_intercept_distance = np.max(discrete_distances[able_to_intercept[i, :]])
                            max_intercept_distances.append(defender_max_intercept_distance)
                            defender_TTA_dict[able_defenders[i]] = (np.where(able_to_intercept[i, :]), defender_max_intercept_distance)
                            # tuple --> (indices of where it can intercept, maximum distance away from asset where it can intercept)
                        else:
                            None
                    """
                        now decide which defender should intercept
                        what criteria?
                        Maximum time to intercept (less loitering?)
                        Minimum distance from defender to intercept (?)
                        Maximum distance from asset
                        Closest to maximum allowable distance from asset? --> let's go with this for now
                    """
                    if len(max_intercept_distances) > 0:
                        # someone can intercept
                        defender_with_max_distance_index = np.argmax(np.array(max_intercept_distances))
                        defender_with_max_distance = able_defenders[defender_with_max_distance_index]
                        intercept_distace = min(MAX_ALLOWABLE_INTERCEPT_DISTANCE, max_intercept_distances[defender_with_max_distance_index])
                        intercept_fraction = 1. - intercept_distace/distance_to_asset
                        intercept_point = [x0 + intercept_fraction*t_impact*u*np.cos(th), y0 + intercept_fraction*t_impact*u*np.sin(th)]

                        attacker.hasBeenTargeted = True
                        attacker.targetedBy = defender_with_max_distance
                        attacker.pointOfInterception = copy.deepcopy(intercept_point)
                        defender_with_max_distance.target = attacker
                        defender_with_max_distance.pointOfInterception = copy.deepcopy(intercept_point)
                        defender_with_max_distance.busy = True
                        defender_with_max_distance.strategy = Strategies.StrategySequence(defender_with_max_distance, [
                            (Strategies.PointAtLocation, (defender_with_max_distance, defender_with_max_distance.pointOfInterception)),
                            (Strategies.DestinationOnly, (defender_with_max_distance, defender_with_max_distance.pointOfInterception)),
                            (Strategies.PointAtBoat, (defender_with_max_distance, attacker)),
                            (Strategies.MoveTowardBoat, (defender_with_max_distance, attacker))
                        ])
                        #print "Defender {} should be intercepting".format(defender_with_max_distance.uniqueID)
                    else:
                        print "NO ONE CAN INTERCEPT! OH NOES!"
                        None

                    None



        return

    def updateAsset(self):
        return