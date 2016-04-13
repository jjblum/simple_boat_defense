import Boat
import numpy as np
import copy
import Strategies


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
    def __init__(self, assets, defenders, attackers, dynamic_or_static="static"):
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
        self._max_allowable_intercept_distace = 20.
        self._dynamic_or_static = dynamic_or_static

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

    @property
    def maxAllowableInterceptDistance(self):
        self._max_allowable_intercept_distace

    @maxAllowableInterceptDistance.setter
    def maxAllowableInterceptDistance(self, maid):
        self._max_allowable_intercept_distace = maid

    def update(self):
        self.updateDefense()
        self.updateAttack()

    def updateAttack(self):
        # attacker evaluates its own TTA vs. defender TTA and if an attack is favorable, moves straight at asset
        asset = self._assets[0]
        for attacker in self._attackers:
            x0 = attacker.state[0]
            y0 = attacker.state[1]
            u = np.round(attacker.state[2], 3)
            th = wrapToPi(attacker.state[4])
            globalAngle = asset.globalAngleToBoat(attacker)  # angle from asset to attacker in global frame
            localAngle = attacker.localAngleToBoat(asset)  # angle from attacker to asset in attacker's frame
            distanceToAsset = attacker.distanceToBoat(asset)
            NG = 25
            #fraction = np.linspace(0., 1., NG)
            #discrete_intercept_line = (np.array([fraction*x0, fraction*y0]) + np.array([(1-fraction)*asset.state[0], (1-fraction)*asset.state[1]])).T
            defenderTTA_dict = self._defenseMetric.minTTA_dict()
            radii = self._defenseMetric.radii()
            weak_angle = list()
            #for radius in radii:
                # TODO - deal with finding median or mean angle correctly, i.e. 0 and 360 shouldn't produce 180, and -180 and 180 shouldn't produce 0
                # what if you describe an angle with two numbers: its angular distance from each axis
                # 170 becomes (170, 10), -172 becomes (172, -2) - smaller of the two differences is 12
                # 5 becomes (5, 175), -11 becomes (-11, 349) - smaller of the two differences is 16
                # do this with a list to find the mean or median?
                # 170, 190 should produce 180
            #    weak_angle.append(np.deg2rad(np.floor(np.argmax(defenderTTA_dict[radius]))))
            weak_angle = wrapToPi(np.deg2rad(np.floor(np.argmax(defenderTTA_dict[radii[2]]))))
            # print "weak angle = {:.0f} deg".format(np.rad2deg(np.median(weak_angle)))
            if attacker.feinting and attacker.strategy.finished:
                attacker.feinting = False
            #if not attacker.evading:
            #    attacker.strategy = Strategies.MoveToAngleAlongCircle(attacker, [asset.state[0], asset.state[1]], weak_angle)


            None


            goodAttack = False
            if goodAttack:
                print "good attack"
                attacker.evading = False
                attacker.strategy = Strategies.MoveTowardAsset(attacker)
            elif not attacker.feinting:
                attacker.feinting = True
                if np.random.uniform(0., 1.) < 0.5:
                    direction = "cw"
                else:
                    direction = "ccw"
                feint_distance = np.min((40.0, distanceToAsset/2.))
                attacker.strategy = Strategies.TimedStrategySequence(attacker, [
                    (Strategies.FeintTowardAsset, (attacker, feint_distance))
                ], [30.0])
                print "feint"
                #attacker.strategy = Strategies.FeintTowardAsset(attacker, distanceToInitiateRetreat=20.0)
                #attacker.strategy = Strategies.DestinationOnlyAlongCircle(attacker, [0., 0.], [distanceToAsset*np.cos(weak_angle), distanceToAsset*np.sin(weak_angle)])

        return

    def updateDefense(self):
        asset = self._assets[0]
        for defender in self._defenders:
            if defender.target is not None:
                if defender.target not in self._attackers:  # update defenders that have removed their targets
                    defender.busy = False
                    defender.target.hasBeenTargeted = False
                    defender.target = None
                    defender.pointOfInterception = None
                    defender.strategy = Strategies.Circle_LOS(defender, [0., 0.], 20.0, surgeVelocity=2.5)
                elif defender.pointOfInterception is not None:
                    phi = np.arctan2(defender.pointOfInterception[1] - defender.target.state[1], defender.pointOfInterception[0] - defender.target.state[0])
                    if absoluteAngleDifference(defender.target.state[4], phi) > np.deg2rad(15.): #or defender.target.state[2] < 0.1:
                        # attacker is no longer headed to the intercept point (either by heading or slowing down)
                        defender.busy = False
                        #if self._dynamic_or_static == "dynamic":
                        #    defender.strategy = Strategies.Circle_Tracking(defender, [0., 0.], defender.target, radius_growth_rate=0.0)
                        #elif self._dynamic_or_static == "static":
                        defender.strategy = Strategies.StrategySequence(defender, [
                            (Strategies.PointAtLocation, (defender, [defender.originalState[0], defender.originalState[1]])),
                            (Strategies.DestinationOnly, (defender, [defender.originalState[0], defender.originalState[1]])),
                            (Strategies.ChangeHeading, (defender, defender.originalState[4]))
                        ])
                        defender.target.hasBeenTargeted = False
                        defender.target.targetedByCount -= 1
                        defender.target.pointOfInterception = None
                        defender.pointOfInterception = None
                    if defender.distanceToBoat(asset) > defender.target.distanceToBoat(asset):
                        # defender is totally out of position, need another boat to intercept
                        defender.busy = False
                        defender.strategy = Strategies.StrategySequence(defender, [
                            (Strategies.PointAtLocation, (defender, [defender.originalState[0], defender.originalState[1]])),
                            (Strategies.DestinationOnly, (defender, [defender.originalState[0], defender.originalState[1]])),
                            (Strategies.ChangeHeading, (defender, defender.originalState[4]))
                        ])
                        defender.target.hasBeenTargeted = False
                        defender.target.targetedByCount -= 1
                        defender.target.pointOfInterception = None
                        defender.pointOfInterception = None


        """
            4 possible reasons a defender should not intercept:
                1) it is busy intercepting something else
                2) another defender is already intercepting
                3) the attacker is not on a straight line course
                4) the defender cannot reach the line of interception in time

            If a non-busy defender doesn't intercept someting it clearly should, something else is going on erroneously
        """

        able_defenders = [defender for defender in self._defenders if not defender.busy]
        ND = len(able_defenders)
        defenders_X = np.zeros((ND, 2))
        defenders_th = np.zeros((ND,))
        defender_u = np.zeros((ND,))
        for i in range(ND):
            defender = able_defenders[i]
            defenders_X[i, 0] = defender.state[0]
            defenders_X[i, 1] = defender.state[1]
            defender_u[i] = defender.state[2]
            defenders_th[i] = defender.state[4]

        # TODO - need a simple interception alternative for when the boats are close and an interception can happen easily

        # where will attackers be in T seconds? assume straight line constant velocity
        for attacker in self._attackers:
            if attacker.targetedByCount < 1:
                x0 = attacker.state[0]
                y0 = attacker.state[1]
                u = np.round(attacker.state[2], 3)
                th = wrapToPi(attacker.state[4])
                thdot = attacker.state[5]
                angleToAsset = attacker.globalAngleToBoat(asset)
                distance_to_asset = attacker.distanceToBoat(asset)

                ######
                REQUIRED_TIME_BUFFER = 2.0  # extra seconds!
                MAX_ALLOWABLE_INTERCEPT_DISTANCE = self._max_allowable_intercept_distace
                MAX_PREDICTION_TIME = 15.0  # maximum seconds of assumed constant velocity
                MIX_OF_INTERCEPT_AND_DEFENSIVE_ALTERNATIVE = 0.9  # 1 is fully aggressive, 0 is fully passive
                ######

                if np.abs(thdot) < np.deg2rad(5.0):
                    #print "Attacker {} is on a straight line intercept with the asset!".format(attacker.uniqueID)
                    # find the time when it will hit, assuming the asset isn't moving for now
                    NG = int(np.floor(MAX_PREDICTION_TIME + 1))
                    time_window = np.linspace(0., MAX_PREDICTION_TIME, NG)
                    discrete_intercept_line = np.column_stack((x0 + time_window*u*np.cos(th), y0 + time_window*u*np.sin(th)))
                    discrete_distances = np.sqrt(np.power(discrete_intercept_line[:, 0] - asset.state[0], 2) + np.power(discrete_intercept_line[:, 1] - asset.state[0], 2))
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
                    able_to_intercept = np.logical_and(
                            np.repeat(np.atleast_2d(time_window), ND, axis=0) - TTA_by_defender > REQUIRED_TIME_BUFFER,
                            np.repeat(np.atleast_2d(discrete_distances.T), ND, axis=0) < MAX_ALLOWABLE_INTERCEPT_DISTANCE
                    )
                    #able_to_intercept = (np.repeat(np.atleast_2d(time_window), ND, axis=0) - TTA_by_defender) > REQUIRED_TIME_BUFFER
                    defender_TTA_dict = dict()  # defender boat object: minimum time to an intercept
                    min_intercept_times = list()
                    for i in range(ND):
                        if np.any(able_to_intercept[i, :]):
                            defender_min_intercept_time = np.min(time_window[able_to_intercept[i, :]])
                            min_intercept_times.append(defender_min_intercept_time)
                            defender_TTA_dict[able_defenders[i]] = defender_min_intercept_time
                        else:
                            defender_TTA_dict[able_defenders[i]] = 999.
                    """
                        now decide which defender should intercept
                        what criteria?
                        Minimum time to intercept
                    """
                    if len(min_intercept_times) > 0:
                        # someone can intercept
                        defender_with_min_time_index = np.argmax(np.array(min_intercept_times))
                        defender_with_min_time = able_defenders[defender_with_min_time_index]
                        intercept_time = defender_TTA_dict[defender_with_min_time]
                        aggressive_point = np.array([x0 + intercept_time*u*np.cos(th), y0 + intercept_time*u*np.sin(th)])
                        passive_point = np.array([asset.state[0] + 0.25*distance_to_asset*np.cos(angleToAsset + np.pi), asset.state[1] + 0.25*distance_to_asset*np.sin(angleToAsset + np.pi)])
                        intercept_point = list(MIX_OF_INTERCEPT_AND_DEFENSIVE_ALTERNATIVE*aggressive_point + (1 - MIX_OF_INTERCEPT_AND_DEFENSIVE_ALTERNATIVE)*passive_point)

                        attacker.hasBeenTargeted = True
                        attacker.targetedBy = defender_with_min_time
                        attacker.targetedByCount += 1
                        attacker.pointOfInterception = copy.deepcopy(intercept_point)
                        defender_with_min_time.target = attacker
                        defender_with_min_time.pointOfInterception = copy.deepcopy(intercept_point)
                        defender_with_min_time.busy = True
                        defender_with_min_time.strategy = Strategies.StrategySequence(defender_with_min_time, [
                            (Strategies.PointAtLocation, (defender_with_min_time, defender_with_min_time.pointOfInterception)),
                            (Strategies.DestinationOnly, (defender_with_min_time, defender_with_min_time.pointOfInterception, 1.0, True)),
                            (Strategies.PointAtBoat, (defender_with_min_time, attacker)),
                            (Strategies.MoveTowardBoat, (defender_with_min_time, attacker))
                        ])
                        defender_with_min_time.numberOfInterceptionAttempts += 1
                        #print "Defender {} should be intercepting".format(defender_with_max_distance.uniqueID)
                        #print "Defender {} has attempted {} interceptions".format(defender_with_max_distance.uniqueID, defender_with_max_distance.numberOfInterceptionAttempts)
                    else:
                        #print "NO ONE CAN INTERCEPT! OH NOES!"
                        None

                    None



        return

    def updateAsset(self):
        return