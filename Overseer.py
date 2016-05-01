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
    def __init__(self, assets, defenders, attackers, random_or_TTA_attackers="random", static_or_dynamic_defense="static"):
        self._assets = assets
        self._attackers = attackers
        self._defenders = defenders
        self._teams = list()
        self._atk_vs_asset_pairwise_distances = None
        self._def_vs_atk_pairwise_distances = None
        self._defenseMetric = None
        self._max_allowable_intercept_distace = 20.
        self._atk_type = random_or_TTA_attackers
        self._def_type = static_or_dynamic_defense
        self._defender_thCoeff = defenders[0].design.thCoeff
        self._defender_rCoeff = defenders[0].design.rCoeff
        self._defender_u0Coeff = defenders[0].design.u0Coeff
        self._attacker_thCoeff = attackers[0].design.thCoeff
        self._attacker_rCoeff = attackers[0].design.rCoeff
        self._attacker_u0Coeff = attackers[0].design.u0Coeff

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

        defenderTTA_dict = self._defenseMetric.minTTA_dict()
        radii = self._defenseMetric.radii()

        # top 3 spots to attack
        # remember that there is 360 1-degree values for defender TTA, so the index is actually the angle
        # TODO - make this more flexible so that TTA can be some other angle increment
        weak_spots = wrapToPi(np.deg2rad(np.argsort(np.round(defenderTTA_dict[radii[1]], 2))[-3:]))

        for attacker in self._attackers:

            globalAngle = asset.globalAngleToBoat(attacker)
            distanceToAsset = asset.distanceToBoat(attacker)

            if self._atk_type == "TTA":
                angleDifferences = [absoluteAngleDifference(weak_spot, globalAngle) for weak_spot in weak_spots]
                weak_angle = weak_spots[np.argmin(angleDifferences)]
                #print "Attacker {}: weak angle = {:.0f} deg".format(attacker.uniqueID, np.rad2deg(np.median(weak_angle)))
                #print "Attacker {}: {}, isFinished = {}".format(attacker.uniqueID, type(attacker.strategy), attacker.strategy.finished)

                if attacker.strategy.finished and type(attacker.strategy) is Strategies.MoveToAngleAlongCircle:
                    if self.assets[0].time < 6.0:
                        return
                    attacker.strategy = Strategies.MoveTowardAsset(attacker)
                    self.defenseMetric.attackHistory.append((attacker.time, globalAngle))

                if attacker.strategy.finished and type(attacker.strategy) is not Strategies.MoveToAngleAlongCircle:
                        attacker.strategy = Strategies.MoveToAngleAlongCircle(attacker, [asset.state[0], asset.state[1]], weak_angle, radius=np.random.uniform(20., 40.), radius_rate=-0.25)

                if type(attacker.strategy) is Strategies.MoveToAngleAlongCircle:
                    attacker.strategy.updateGoal(weak_angle)

                    # TODO - check if time to arrive at asset is less than defender time to arrive, and if it is, attack immediately
                    localAngleToAsset = np.abs(wrapToPi(attacker.localAngleToBoat(asset)))
                    u = attacker.state[2]
                    TTA = self._attacker_thCoeff*localAngleToAsset + self._attacker_rCoeff*distanceToAsset + self._attacker_u0Coeff*u
                    if TTA < defenderTTA_dict[radii[1]][np.floor(np.rad2deg(globalAngle))]:
                        attacker.strategy = Strategies.MoveTowardAsset(attacker)
                        self.defenseMetric.attackHistory.append((attacker.time, globalAngle))

            elif self._atk_type == "random":
                if attacker.strategy.finished:
                    attacker.strategy = Strategies.MoveTowardAsset(attacker)
                    self.defenseMetric.attackHistory.append((attacker.time, globalAngle))

    def updateDefense(self):
        asset = self._assets[0]
        #print "\n"
        for defender in self._defenders:

            if defender.target is not None:
                if defender.target not in self._attackers:  # update defenders that have removed their targets
                    defender.busy = False
                    defender.target.hasBeenTargeted = False
                    defender.target = None
                    defender.pointOfInterception = None
                    if self._def_type == "dynamic":
                        defender.strategy = Strategies.Circle_LOS(defender, [0., 0.], 10.0, surgeVelocity=2.5, direction="ccw")
                    elif self._def_type == "static" or self._def_type == "turned":
                        defender.strategy = Strategies.StrategySequence(defender, [
                            (Strategies.PointAtLocation, (defender, [defender.originalState[0], defender.originalState[1]])),
                            (Strategies.DestinationOnly, (defender, [defender.originalState[0], defender.originalState[1]])),
                            (Strategies.ChangeHeading, (defender, defender.originalState[4]))
                        ])
                elif defender.pointOfInterception is not None:
                    phi = np.arctan2(defender.pointOfInterception[1] - defender.target.state[1], defender.pointOfInterception[0] - defender.target.state[0])
                    if absoluteAngleDifference(defender.target.state[4], phi) > np.deg2rad(15.): #or defender.target.state[2] < 0.1:
                        # attacker is no longer headed to the intercept point (either by heading or slowing down)
                        defender.busy = False
                        defender.target.hasBeenTargeted = False
                        defender.target.targetedByCount -= 1
                        defender.target.pointOfInterception = None
                        defender.target = None
                        defender.pointOfInterception = None
                        if self._def_type == "dynamic":
                            defender.strategy = Strategies.Circle_LOS(defender, [0., 0.], 10.0, surgeVelocity=2.5, direction="ccw")
                        elif self._def_type == "static" or self._def_type == "turned":
                            defender.strategy = Strategies.StrategySequence(defender, [
                                (Strategies.PointAtLocation, (defender, [defender.originalState[0], defender.originalState[1]])),
                                (Strategies.DestinationOnly, (defender, [defender.originalState[0], defender.originalState[1]])),
                                (Strategies.ChangeHeading, (defender, defender.originalState[4]))
                            ])
                    elif defender.distanceToBoat(asset) > defender.target.distanceToBoat(asset):
                        # defender is totally out of position, need another boat to intercept
                        defender.busy = False
                        defender.target.hasBeenTargeted = False
                        defender.target.targetedByCount -= 1
                        defender.target.pointOfInterception = None
                        defender.target = None
                        defender.pointOfInterception = None
                        if self._def_type == "dynamic":
                            defender.strategy = Strategies.Circle_LOS(defender, [0., 0.], 10.0, surgeVelocity=2.5, direction="ccw")
                        elif self._def_type == "static" or self._def_type == "turned":
                            defender.strategy = Strategies.StrategySequence(defender, [
                                (Strategies.PointAtLocation, (defender, [defender.originalState[0], defender.originalState[1]])),
                                (Strategies.DestinationOnly, (defender, [defender.originalState[0], defender.originalState[1]])),
                                (Strategies.ChangeHeading, (defender, defender.originalState[4]))
                            ])



            """
            4 possible reasons a defender should not intercept:
                1) it is busy intercepting something else
                2) another defender is already intercepting
                3) the attacker is not on a straight line course
                4) the defender cannot reach the line of interception in time

            If a non-busy defender doesn't intercept someting it clearly should, something else is going on erroneously
            """

        # TODO - need a simple interception alternative for when the boats are close and an interception can happen easily

        # where will attackers be in T seconds? assume straight line constant velocity
        for attacker in self._attackers:
            if attacker.targetedByCount < 1:
                able_defenders = [defender for defender in self._defenders if not defender.busy]
                #print "able defenders: {}".format(able_defenders)
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

                x0 = attacker.state[0]
                y0 = attacker.state[1]
                u = np.round(attacker.state[2], 3)
                th = wrapToPi(attacker.state[4])
                thdot = attacker.state[5]
                angleToAsset = asset.globalAngleToBoat(attacker)#attacker.globalAngleToBoat(asset)
                distance_to_asset = attacker.distanceToBoat(asset)

                ######
                REQUIRED_TIME_BUFFER = 2.0  # extra seconds!
                MAX_ALLOWABLE_INTERCEPT_DISTANCE = self._max_allowable_intercept_distace
                MAX_PREDICTION_TIME = 20.0  # maximum seconds of assumed constant velocity
                MIX_OF_INTERCEPT_AND_DEFENSIVE_ALTERNATIVE = 1.0  # 1 is fully aggressive, 0 is fully passive
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
                    local_angle = np.abs(wrapToPi(local_angle))  # must be on the [0, pi] interval
                    TTA = self._defender_thCoeff*local_angle + self._defender_rCoeff*R + self._defender_u0Coeff*np.repeat(defender_u, NG, axis=0)
                    # remember, each NG rows are for a single defender -- TTA shape is (ND*NG,)
                    TTA_by_defender = np.reshape(TTA, (ND, NG))
                    able_to_intercept = np.logical_and(
                            np.repeat(np.atleast_2d(time_window), ND, axis=0) - TTA_by_defender > REQUIRED_TIME_BUFFER,
                            np.repeat(np.atleast_2d(discrete_distances.T), ND, axis=0) < MAX_ALLOWABLE_INTERCEPT_DISTANCE
                    )
                    defender_TTA_dict = dict()  # defender boat object: minimum time to an intercept
                    min_intercept_times = list()
                    for i in range(ND):
                        if np.any(able_to_intercept[i, :]):
                            defender_min_intercept_time = np.min(time_window[able_to_intercept[i, :]])
                        else:
                            defender_min_intercept_time = 999.
                        defender_TTA_dict[able_defenders[i]] = defender_min_intercept_time
                        min_intercept_times.append(defender_min_intercept_time)
                    #print "min intercept times: {}".format(min_intercept_times)
                    """
                        now decide which defender should intercept
                        what criteria?
                        Minimum time to intercept
                    """
                    if len(min_intercept_times) > 0:
                        if np.min(min_intercept_times) < MAX_PREDICTION_TIME:
                            # someone can intercept
                            defender_with_min_time_index = np.argmin(np.array(min_intercept_times))
                            defender_with_min_time = able_defenders[defender_with_min_time_index]
                            intercept_time = defender_TTA_dict[defender_with_min_time]
                            aggressive_point = np.array([x0 + intercept_time*u*np.cos(th), y0 + intercept_time*u*np.sin(th)])
                            passive_point = np.array([asset.state[0] + 0.25*distance_to_asset*np.cos(angleToAsset), asset.state[1] + 0.25*distance_to_asset*np.sin(angleToAsset)])
                            intercept_point = list(MIX_OF_INTERCEPT_AND_DEFENSIVE_ALTERNATIVE*aggressive_point + (1 - MIX_OF_INTERCEPT_AND_DEFENSIVE_ALTERNATIVE)*passive_point)
                            #print intercept_point, intercept_time

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