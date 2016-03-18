import abc  # abstract base classes
import math
import numpy as np
import scipy.spatial as spatial
import matplotlib.pyplot as plt
import Controllers
import PiazziSpline
import Utility

# TODO - updating the strategy effectively deletes all the asset, attacker, defender, etc. information that the last strategy held
#        is there a way to have that information transferred from stategy to strategy?

# TODO - an "overseer" or "team" strategy that sets the strategy of more than one individuals

# TODO - the overseer should assign the appropriate defender - maybe this should be more like an auction in a real distributed system

# TODO - planning a path through vulnerabilities

# TODO - LIST OF USEFUL STRATEGIES:
#   DONE 1a) Move to asset
#   1b) Follow trajectory toward asset
#   1c) Follow trajectory onto line of attack
#   DONE 1d) Move in a circle around asset (could be the same as perimeter patrol)
#   DONE 2a) Point away from asset
#   DONE 2b) Align heading with asset's heading
#   3) Get into an ellipse (or other formation) around asset (extend long axis according to asset speed)
#   4) Intercept - linear assumption - cast a ray where target may go, find where along that trajectory you can reach
#   DONE 5) Patrol perimeter - ideal boat follows a circuit path at a specified velocity and direction
#           Will need to provide a chain of points as the path
#   6) SEQUENCE: [get into ellipse, point away from asset]
#   7) TIMED SEQUENCE: randomly switch back and forth between moving toward asset and circling around asset
#   8) Form a line between a spot and the asset (set that spot to the attacker's location)
#   9) TEAM: divide up teams of defenders evenly between attackers and form a line
#  10) SEQUENCE: Reach line of attack ASAP, with final orientation along line of attack, then intercept attacker
#  11) SEQUENCE: [get into ellipse, point away from asset, assign a defender to intercept an attacker]


class Strategy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, boat):
        self._boat = boat
        self._controller = None
        self._finished = False  # setting this to True does not necessarily mean a strategy will terminate
        self._t = boat.time
        self._assets = boat.assets
        self._attackers = boat.attackers
        self._defenders = boat.defenders
        self._strategy = self  # returns self by default (unless it is a nested strategy or sequence)
        self._strategies = list()  # not relevant for basic strategies

    @abc.abstractmethod
    def idealState(self):
        # virtual function, uses information to return an ideal state
        # this will be used for fox-rabbit style control
        return

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_in):
        self._strategy = strategy_in

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished_in):
        self._finished = finished_in

    def updateFinished(self):
        self.strategy.finished = self.strategy.controller.finished

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, controller_in):
        self._controller = controller_in

    def actuationEffortFractions(self):
        return self.controller.actuationEffortFractions()

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t
        if self._controller is not None:
            self.controller.time = t

    @property
    def boat(self):
        return self._boat

    @boat.setter
    def boat(self, boat_in):
        self._boat = boat_in

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


class StrategySequence(Strategy):
    """
        strategySequence: list of (class, (inputs)) stategies to be instantiated
        strategy: drills down to the lowest level current strategy
        strategies: a list of instantiated strategies

        We delay the instantiation in order to provide the most up to date system state for the later strategies.
        This is important for Executors that must make strategy choices based on system state.
        Previously, when there was just a simple list of strategies, this would instantiate all of them at once.
    """
    def __init__(self, boat, sequence):
        super(StrategySequence, self).__init__(boat)
        self._strategySequence = sequence
        self._currentStrategy = 0  # index of the current strategy
        self._strategies = list()
        self.start(self._currentStrategy)

    def start(self, currentStrategyIndex):
        # instantiate a strategy from the uninstantiated sequence
        self._strategies.append(self._strategySequence[self._currentStrategy][0](
                    *self._strategySequence[self._currentStrategy][1]))
        self._strategy = self._strategies[-1]
        self.controller = self.strategy.controller

    @property
    def strategySequence(self):
        return self._strategySequence

    @strategySequence.setter
    def strategySequence(self, strategySequence_in):
        self._strategySequence = strategySequence_in

    # override
    def actuationEffortFractions(self):
        return self._strategies[-1].actuationEffortFractions()

    # override
    def updateFinished(self):
        self.time = self.boat.time
        self._strategies[-1].time = self.boat.time
        self._strategies[-1].updateFinished()
        if self._strategies[-1].finished and \
                self._currentStrategy < len(self.strategySequence) - 1:
            self._currentStrategy += 1
            # must manually update strategy and controller!
            self._strategies.append(self._strategySequence[self._currentStrategy][0](
                    *self._strategySequence[self._currentStrategy][1]))
            self._strategy = self._strategies[-1]
            self.controller = self.strategy.controller
        if self._strategies[-1].finished:
            # sequence is finished when last strategy in a sequence is finished
            self.finished = True

    def idealState(self):
        return self._strategies[-1].idealState()


class Executor(Strategy):
    __metaclass__ = abc.ABCMeta

    def __init__(self, boat):
        super(Executor, self).__init__(boat)
        self._readyToPickStrategy = True  # we dont want to pick a strategy every single iteration

    @abc.abstractmethod
    def pickStrategy(self):
        # virtual method that determines the current strategy based on system state
        return

    @property  # need to override the standard controller property with the nested strategy's controller
    def controller(self):
        return self._strategy.controller

    @controller.setter  # need to override the standard controller property with the nested strategy's controller
    def controller(self, controller):
        self._controller = controller

    def updateFinished(self):  # need to override to get the finished status of the nested strategy!!!
        if self.strategy == self:  # a strategy hasn't been assigned yet
            return
        self.strategy.updateFinished()
        self.finished = self.strategy.finished

    def idealState(self):
        if self._readyToPickStrategy:
            self.pickStrategy()
        return self._strategy.idealState()


class TimedStrategySequence(StrategySequence):
    def __init__(self, boat, sequence, timing):
        super(StrategySequence, self).__init__(boat)
        self._strategySequence = sequence
        self._strategyTiming = timing
        self._currentStrategyStartTime = boat.time
        self._currentStrategy = 0
        self.start(self._currentStrategy)

    # override
    def updateFinished(self):
        # notice the OR - the timing represents a timeout
        self._strategies[-1].updateFinished()
        dt = (self.boat.time - self._currentStrategyStartTime)

        # TODO - what am i doing here? _strategies[-1] is the current strategy, isn't it? so doesn't this shortcut the entire sequence?
        if self._currentStrategy == len(self.strategySequence) - 1 and \
           dt >= self._strategyTiming[self._currentStrategy]:

            # sequence is finished when last strategy in a sequence is finished or total time has run out
            print "Strategy Sequence Finished"
            self._strategies.append(DoNothing(self.boat))
            self._strategy = self._strategies[-1]
            self.finished = True

        if (self._strategies[-1].finished or
                dt >= self._strategyTiming[self._currentStrategy]) and \
                self._currentStrategy < len(self.strategySequence) - 1:

            self._strategies[-1].finished = True
            self._currentStrategy += 1
            # must manually update strategy and controller!
            self.start(self._currentStrategy)
            self._currentStrategyStartTime = self.boat.time


class DoNothing(Strategy):
    # a strategy that prevents actuation
    def __init__(self, boat):
        super(DoNothing, self).__init__(boat)
        self.controller = Controllers.DoNothing()

    def idealState(self):
        return np.zeros((6,))


class StationKeep(Strategy):
    # a strategy that just sets the destination to the current location
    def __init__(self, boat):
        super(StationKeep, self).__init__()
        self.controller = Controllers.SurgeAndHeadingPID(boat)

    def idealState(self):
        # rabbit boat sits at the boats current location
        state = np.zeros((6,))
        state[0] = self.boat.state[0]
        state[1] = self.boat.state[1]
        state[4] = self.boat.state[4]
        self.controller.idealState = state
        return state


class ChangeHeading(Strategy):
    # a strategy that spins in place until the boat has the desired heading
    def __init__(self, boat, heading=0.0):
        super(ChangeHeading, self).__init__(boat)
        self._desiredHeading = heading
        self.controller = Controllers.HeadingOnlyPID(boat)

    @property
    def desiredHeading(self):
        return self._desiredHeading

    @desiredHeading.setter
    def desiredHeading(self, desiredHeading_in):
        self._desiredHeading = desiredHeading_in

    def idealState(self):
        # rabbit boat is at the boat's current location, just rotated
        state = np.zeros((6,))
        state[0] = self.boat.state[0]
        state[1] = self.boat.state[1]
        state[4] = self._desiredHeading
        self.controller.idealState = state


class HoldHeading(Strategy):
    # a strategy where the ideal boat moves with a fixed velocity along the boat's current heading
    def __init__(self, boat, surgeVelocity=0.0):
        super(HoldHeading, self).__init__(boat)
        self._t0 = boat.time  # time when this strategy started
        self._surgeVelocity = surgeVelocity  # [m/s]
        self.controller = Controllers.SurgeAndHeadingPID(boat)

    def idealState(self):
        # rabbit boat moves forward at fixed velocity
        state = np.zeros((6,))
        u = self._surgeVelocity
        th = self.boat.state[4]
        thdot = self.boat.state[5]
        time_expired = self.time - self._t0 + 1.0  # need to add a little extra
        state[0] = self.boat.state[0] + u*math.cos(th)*time_expired
        state[1] = self.boat.state[1] + u*math.sin(th)*time_expired
        state[2] = u
        state[3] = self.boat.state[3]
        state[4] = th
        state[5] = 0
        self.controller.idealState = state  # update this here so the controller doesn't need to import Strategies
        #print "boat {} \n\tstate = {}\n\tideal state = {}".format(self.boat.uniqueID, self.boat.state, state)
        return state


# TODO - a spin in place strategy and controller with heading rate control (HeadingOnlyPID but on thdot rather than th)
class SpinInPlace(Strategy):
    def __init__(self, boat, direction="cw"):
        super(SpinInPlace, self).__init__(boat)
        self._boat = boat
        self.controller = Controllers.HeadingOnlyPID(boat)
        self._lookahead = np.deg2rad(15.0)
        if direction == "cw":
            self._lookahead *= -1.

    def idealState(self):
        state = np.zeros((6,))
        state[4] = self.boat.state[4] + self._lookahead
        self.controller.idealState = state


class DestinationOnly(Strategy):
    # a strategy that only returns the final destination location
    def __init__(self, boat, destination, positionThreshold=1.0, driftDown=True):
        super(DestinationOnly, self).__init__(boat)
        self._destinationState = destination
        self.controller = Controllers.PointAndShootPID(boat, positionThreshold, driftDown)

    @property
    def destinationState(self):
        return self._destinationState  # as of now, even a high level strategy needs to have a handle to the controller it will ultimately use

    @destinationState.setter
    def destinationState(self, destinationState_in):
        if len(destinationState_in) == 6:
            self._destinationState = destinationState_in
        elif len(destinationState_in) == 3:
            # assuming they are using x, y, th
            state = np.zeros((6,))
            state[0] = destinationState_in[0]
            state[1] = destinationState_in[1]
            state[4] = destinationState_in[2]
            self._destinationState = state
        elif len(destinationState_in) == 2:
            # assuming they are using x, y
            state = np.zeros((6,))
            state[0] = destinationState_in[0]
            state[1] = destinationState_in[1]
            self._destinationState = state

    def idealState(self):
        self.boat.plotData = np.atleast_2d(np.array([[self.boat.state[0], self.boat.state[1]], [self._destinationState[0], self._destinationState[1]]]))
        self.controller.idealState = self.destinationState  # update this here so the controller doesn't need to import Strategies


class PointAtLocation(Strategy):
    def __init__(self, boat, target):
        super(PointAtLocation, self).__init__(boat)
        self._boat = boat
        self.controller = Controllers.HeadingOnlyPID(boat)
        self._target = target

    def idealState(self):
        dx = self._target[0] - self.boat.state[0]
        dy = self._target[1] - self.boat.state[1]
        state = np.zeros((6,))
        state[4] = np.arctan2(dy, dx)
        self.controller.idealState = state


class PointAtAsset(Strategy):
    # a strategy that just points the boat at the geometric mean of the assets
    # an example of a NESTED STRATEGY
    def __init__(self, boat):
        super(PointAtAsset, self).__init__(boat)
        self._strategy = ChangeHeading(boat)  # the lower level nested strategy
        self.controller = self._strategy.controller

    @property  # need to override the standard controller property with the nested strategy's controller
    def controller(self):
        return self._strategy.controller

    @controller.setter  # need to override the standard controller property with the nested strategy's controller
    def controller(self, controller):
        self._controller = controller

    def updateFinished(self):  # need to override to get the finished status of the nested strategy!!!
        self.strategy.updateFinished()
        self.finished = self.strategy.finished

    def angleToAsset(self):
        if len(self.assets) == 0:
            # no asset to point at, do not change heading
            return self.boat.state[4]
        x = self.boat.state[0]
        y = self.boat.state[1]
        assets_x = [b.state[0] for b in self.assets]
        assets_y = [b.state[1] for b in self.assets]
        asset_x = np.mean(assets_x)
        asset_y = np.mean(assets_y)
        return math.atan2(asset_y - y, asset_x - x)

    def idealState(self):
        self._strategy.desiredHeading = self.angleToAsset()
        return self._strategy.idealState()


class PointAwayFromAsset(Strategy):
    # a strategy that just points the boat away from the assets
    def __init__(self, boat):
        super(PointAwayFromAsset, self).__init__(boat)
        self._strategy = ChangeHeading(boat)  # the lower level nested strategy
        self.controller = self._strategy.controller

    @property  # need to override the standard controller property with the nested strategy's controller
    def controller(self):
        return self._strategy.controller

    @controller.setter  # need to override the standard controller property with the nested strategy's controller
    def controller(self, controller):
        self._controller = controller

    def updateFinished(self):  # need to override to get the finished status of the nested strategy!!!
        self.strategy.updateFinished()
        self.finished = self.strategy.finished

    def angleToAsset(self):
        if len(self.assets) == 0:
            # no asset to point at, do not change heading
            return self.boat.state[4]
        x = self.boat.state[0]
        y = self.boat.state[1]
        assets_x = [b.state[0] for b in self.assets]
        assets_y = [b.state[1] for b in self.assets]
        asset_x = np.mean(assets_x)
        asset_y = np.mean(assets_y)
        return math.atan2(asset_y - y, asset_x - x) + math.pi

    def idealState(self):
        self._strategy.desiredHeading = self.angleToAsset()
        return self._strategy.idealState()


class PointWithAsset(Strategy):
    # a strategy that just points the boat in the same direction at the asset is pointing
    def __init__(self, boat):
        super(PointWithAsset, self).__init__(boat)
        self._strategy = ChangeHeading(boat)  # the lower level nested strategy
        self.controller = self._strategy.controller

    @property  # need to override the standard controller property with the nested strategy's controller
    def controller(self):
        return self._strategy.controller

    @controller.setter  # need to override the standard controller property with the nested strategy's controller
    def controller(self, controller):
        self._controller = controller

    def updateFinished(self):  # need to override to get the finished status of the nested strategy!!!
        self.strategy.updateFinished()
        self.finished = self.strategy.finished

    def idealState(self):
        self._strategy.desiredHeading = self.assets[0].state[4]
        return self._strategy.idealState()


class MoveTowardAsset(Strategy):
    # nested strategy - uses DestinationOnly with the asset as the goal
    def __init__(self, boat, positionThreshold):
        super(MoveTowardAsset, self).__init__(boat)
        self._strategy = DestinationOnly(boat, [self.assets[0].state[0], self.assets[0].state[1]], positionThreshold)  # the lower level nested strategy
        self.controller = self._strategy.controller

    @property  # need to override the standard controller property with the nested strategy's controller
    def controller(self):
        return self._strategy.controller

    @controller.setter  # need to override the standard controller property with the nested strategy's controller
    def controller(self, controller):
        self._controller = controller

    def updateFinished(self):  # need to override to get the finished status of the nested strategy!!!
        self.strategy.updateFinished()
        self.finished = self.strategy.finished

    def idealState(self):
        self._strategy.destinationState = [self.assets[0].state[0], self.assets[0].state[1]]
        self._strategy.idealState()


class Square(StrategySequence):
    # Move around the vertices of a square given its center and edge size.
    # User can specify which corner and rotation direction they want
    # This serves as an example of a Strategy built from a sequence of more primitive Strategies and sequences
    def __init__(self, boat, positionThreshold, center, edgeSize, firstCorner="bottom_right", direction="cw"):
        super(StrategySequence, self).__init__(boat)  # run Strategy __init__, NOT StrategySequence.__init__ !!!
        self._strategySequence = list()
        self._currentStrategy = 0  # index of the current strategy

        bottom_left = list(np.array(center) + np.array([-edgeSize/2.0, -edgeSize/2.0]))
        bottom_right = list(np.array(center) + np.array([edgeSize/2.0, -edgeSize/2.0]))
        upper_right = list(np.array(center) + np.array([edgeSize/2.0, edgeSize/2.0]))
        upper_left = list(np.array(center) + np.array([-edgeSize/2.0, edgeSize/2.0]))
        vertices = list()
        headings = list()
        if firstCorner == "bottom_left":
            if direction == "cw":
                vertices = [bottom_left, upper_left, upper_right, bottom_right, bottom_left]
                headings = [math.pi/2.0, 0.0, -math.pi/2.0, math.pi]
            elif direction == "ccw":
                vertices = [bottom_left, bottom_right, upper_right, upper_left, bottom_left]
                headings = [0.0, math.pi/2.0, math.pi, -math.pi/2.0]
        elif firstCorner == "bottom_right":
            if direction == "cw":
                vertices = [bottom_right, bottom_left, upper_left, upper_right, bottom_right]
                headings = [math.pi, math.pi/2.0, 0.0, -math.pi/2.0]
            elif direction == "ccw":
                vertices = [bottom_right, upper_right, upper_left, bottom_left, bottom_right]
                headings = [math.pi/2.0, math.pi, -math.pi/2.0, 0.0]
        elif firstCorner == "upper_right":
            if direction == "cw":
                vertices = [upper_right, bottom_right, bottom_left, upper_left, upper_right]
                headings = [-math.pi/2.0, math.pi, math.pi/2.0, 0.0]
            elif direction == "ccw":
                vertices = [upper_right, upper_left, bottom_left, bottom_right, upper_right]
                headings = [math.pi, -math.pi/2.0, 0.0, math.pi/2.0]
        elif firstCorner == "upper_left":
            vertices.append(upper_left)
            if direction == "cw":
                vertices = [upper_left, upper_right, bottom_right, bottom_left, upper_left]
                headings = [0.0, -math.pi/2.0, math.pi, math.pi/2.0]
            elif direction == "ccw":
                vertices = [upper_left, bottom_left, bottom_right, upper_right, upper_left]
                headings = [-math.pi/2.0, 0.0, math.pi/2.0, math.pi]
        self._strategySequence = \
        [
            (DestinationOnlyExecutor, (boat, vertices[0], positionThreshold)),
            (StrategySequence, (boat,
            [
                (ChangeHeading, (boat, headings[0])),
                (DestinationOnly, (boat, vertices[1], positionThreshold))
            ])),
            (StrategySequence, (boat,
            [
                (ChangeHeading, (boat, headings[1])),
                (DestinationOnly, (boat, vertices[2], positionThreshold))
            ])),
            (StrategySequence, (boat,
            [
                (ChangeHeading, (boat, headings[2])),
                (DestinationOnly, (boat, vertices[3], positionThreshold))
            ])),
            (StrategySequence, (boat,
            [
                (ChangeHeading, (boat, headings[3])),
                (DestinationOnly, (boat, vertices[4], positionThreshold))
            ]))
        ]
        self.start(self._currentStrategy)


class SingleSpline(Strategy):
    # follow a single Piazzi spline to destination
    """
        Integral LOS Control for Path Following of Underactuated Marine Surface Vessels in the
        Presence of Constant Ocean Currents
        Borhug et. al 2008

        Ideal boat does not propagate based on time or velocity, it is always some distance ahead.
        This distance changes as curvature changes.
        Only heading and surge velocity are controlled.
    """
    def __init__(self, boat, destination, finalHeading=0.0, surgeVelocity=1.0, positionThreshold=1.0, N=100, driftDown=True):
        super(SingleSpline, self).__init__(boat)
        self._destination = destination
        self._surgeVelocity = surgeVelocity
        self._finalHeading = finalHeading
        self._N = N  # points along the spline
        self._length = None
        self._sx = None
        self._sy = None
        self._sth = None
        self._totalLength = None
        self._u = None
        self._splineCoeffs = None
        self.generateSpline()
        self._positionThreshold = positionThreshold
        self._sigma = 0.1
        self._lookAhead = 0.05  # want this to change with curvature
        self._errorAccumulator = 0.0  # initialize the integral action
        self._headingErrorSurgeCutoff = 45.0*math.pi/180.0  # thrust signal rolls off as a cosine, hitting zero here
        self.controller = Controllers.LineOfSight(boat, destination, positionThreshold, driftDown)

    def generateSpline(self):
        x0 = self.boat.state[0]
        x1 = self._destination[0]
        y0 = self.boat.state[1]
        y1 = self._destination[1]
        th0 = self.boat.state[4]
        th1 = self._finalHeading
        self._sx, self._sy, self._sth, \
            self._length, self._u, self._splineCoeffs = PiazziSpline.piazziSpline(x0, y0, th0, x1, y1, th1, N=self._N)
        self.boat.plotData = np.column_stack((self._sx, self._sy))
        self._totalLength = self._length[-1]
        # print np.c_[self._progress, self._sx, self._sy, self._sth]

    def idealState(self):
        # find closest point on the spline
        try:
            u_star, closest, error_y = Utility.closestPointOnSpline2D(
                    self._length, self._sx, self._sy, self.boat.state[0], self.boat.state[1], self._u, self._splineCoeffs)
        except:
            self.controller.idealState = np.zeros((6,))  # handle when u_star is NaN inside the call, it returns None
            return
        tangent_th = np.interp(u_star, self._u, self._sth)
        # we don't just go in straight lines, so find the location that is lookaheadDistance forward on the spline
        u_lookahead = max(0.0, min(u_star + self._lookAhead, 1.0))
        lookaheadState = Utility.splineToEuclidean2D(self._splineCoeffs, u_lookahead)
        dx_global = lookaheadState[0] - closest[0]
        dy_global = lookaheadState[1] - closest[1]
        # transform into the tangent frame (Frenet frame)
        dx_frenet = dx_global*math.cos(tangent_th) + dy_global*math.sin(tangent_th)
        dy_frenet = dx_global*math.sin(tangent_th) - dy_global*math.cos(tangent_th)
        # need sign of distance to spline to change - look at sign of cross product to determine "handed-ness"
        angle_from_closest_to_boat = math.atan2(closest[1] - self.boat.state[1], closest[0] - self.boat.state[0])
        sign_test = np.cross([math.cos(tangent_th), math.sin(tangent_th)], [math.cos(angle_from_closest_to_boat), math.sin(angle_from_closest_to_boat)])
        error_y *= np.sign(sign_test)

        self._errorAccumulator = 0.0
        #self._errorAccumulator += dt*self._lookAheadLength*error_y/(math.pow(error_y + self._sigma*self._errorAccumulator, 2) + math.pow(self._lookAheadLength, 2))
        #self._errorAccumulator += dt*dx*(error_y - dy)/(
        #   math.pow(error_y + self._sigma*self._errorAccumulator - dy, 2) + math.pow(dx, 2))
        state = np.zeros((6,))
        relative_angle = math.atan2((error_y - dy_frenet), dx_frenet)
        global_angle = tangent_th + relative_angle
        state[4] = global_angle

        th_lookahead = max(0.0, min(u_star + (1 + 0.5*self._surgeVelocity/self.boat.design.minSpeed)*self._lookAhead, 1.0))
        lookahead_th = np.interp(th_lookahead, self._u, self._sth)
        clipped_lookahead_dth = np.clip(np.abs(lookahead_th - tangent_th), 0.0, self._headingErrorSurgeCutoff)
        surgeVelocityCap = max(self.boat.design.maxSpeed*math.cos(math.pi/2.0*clipped_lookahead_dth/self._headingErrorSurgeCutoff), self.boat.design.minSpeed)
        # print "tangent change = {:.2f} deg, surge velocity limited to = {:.2f}".format(clipped_lookahead_dth*180./np.pi, surgeVelocityCap)

        state[2] = min(surgeVelocityCap, self._surgeVelocity)
        self.controller.idealState = state

        # remaining distance along spline
        remainingDistance = self._totalLength - np.interp(u_star, self._u, self._length)
        self.controller.remainingDistance = remainingDistance


"""
class Weave(Strategy):
    # Ideal state weaves in a sine wave
    def __init__(self, boat, weaveHalfWidth=10.0, weavePeriod=10.0):
        super(Weave, self).__init__(boat)
        self._th0 = boat.state[4]
        def mySine(t):
"""


# TODO - make lookAhead a function of distance to the line! 1-e^(-1/distance) very small distance is huge look ahead?
class Line_LOS(Strategy):
    # Lookahead is in meters here, very much unlike using a spline!
    def __init__(self, boat, x0, y0, x1, y1, surgeVelocity=1.0, headingErrorSurgeCutoff=np.deg2rad(30.0)):
        super(Line_LOS, self).__init__(boat)
        self._x0 = x0
        self._x1 = x1
        self._dx = x1-x0
        self._dy = y1-y0
        self._y0 = y0
        self._y1 = y1
        self._th = np.arctan2(self._dy, self._dx)
        self._L = np.sqrt(np.power(self._dx, 2.)+np.power(self._dy, 2.))
        self._destination = [x1, y1]
        self._surgeVelocity = surgeVelocity
        self._headingErrorSurgeCutoff = headingErrorSurgeCutoff
        self.controller = Controllers.LineOfSight(boat, self._destination, headingErrorSurgeCutoff=headingErrorSurgeCutoff, driftDown=True)
        self._lookAhead = 1.0

    def idealState(self):
        state = np.zeros((6,))
        # project point onto line
        x = self.boat.state[0]
        dx = x - self._x0
        y = self.boat.state[1]
        dy = y - self._y0
        th = np.arctan2(dy, dx)
        dth = np.abs(self._th - th)
        currentL = np.linalg.norm(np.array([dx, dy]))*np.cos(dth)
        distance = np.linalg.norm(np.array([dx, dy]))*np.sin(dth)
        self._lookAhead = 5.*(1.-np.tanh(0.1*distance))
        print "distance = {:.2f} m   lookAhead = {:.2f} m".format(distance, self._lookAhead)
        projected_state = np.array([self._x0 + currentL*np.cos(self._th), self._y0 + currentL*np.sin(self._th)])
        lookaheadState = projected_state + np.array([self._lookAhead*np.cos(self._th), self._lookAhead*np.sin(self._th)])
        boatToLookahead = np.array([lookaheadState[0] - x, lookaheadState[1] - y])
        boatToLookaheadAngle = np.arctan2(boatToLookahead[1], boatToLookahead[0])
        state[2] = self._surgeVelocity
        state[4] = boatToLookaheadAngle
        self.controller.idealState = state


class Circle_LOS(Strategy):
    def __init__(self, boat, center, radius, direction="cw", surgeVelocity=1.0):
        super(Circle_LOS, self).__init__(boat)
        self._boat = boat
        self._center = center
        self._radius = radius
        self._surgeVelocity = surgeVelocity
        self._lookAhead = np.deg2rad(12.0)  # TODO - looks like this needs to be a function of the circle radius!
        self.controller = Controllers.LineOfSight(boat, driftDown=False)
        if direction == "cw":
            self._lookAhead *= -1.
        ths = np.linspace(-np.pi, np.pi, 100)
        self.boat.plotData = np.column_stack((center[0] + radius*np.cos(ths), center[1] + radius*np.sin(ths)))

    def idealState(self):
        # angle of boat with respect to center
        dX = np.array([self.boat.state[0], self.boat.state[1]]) - np.array(self._center)
        boatAngle = np.arctan2(dX[1], dX[0])
        lookaheadAngle = boatAngle + self._lookAhead
        lookaheadState = np.array([self._center[0] + self._radius*np.cos(lookaheadAngle),
                                   self._center[1] + self._radius*np.sin(lookaheadAngle)])
        boatToLookahead = np.array([lookaheadState[0] - self.boat.state[0], lookaheadState[1] - self.boat.state[1]])
        boatToLookaheadAngle = np.arctan2(boatToLookahead[1], boatToLookahead[0])
        state = np.zeros((6,))
        state[2] = self._surgeVelocity
        state[4] = boatToLookaheadAngle
        self.controller.idealState = state
        # print "angle from boat to center = {:.2f} deg".format(np.rad2deg(boatAngle))
        # print "lookahead angle = {:.2f} deg".format(np.rad2deg(lookaheadAngle))
        # print "lookahead state = {:.2f},{:.2f}".format(lookaheadState[0], lookaheadState[1])
        # print "desired global angle = {:.2f} deg".format(np.rad2deg(boatToLookaheadAngle))


class Circle_PID(Strategy):
    """
        Ideal state travels along the perimeter of a circle at a fixed speed
        startingAngle: the location along the circle (in global frame) where the ideal state begins
        speed --> s = R*theta --> sdot = speed = R*theta_dot
        theta_dot = speed/R
        A PID will eventually merge with the circle, but it will take some time
    """
    def __init__(self, boat, center, radius, direction="cw", startingAngle=0.0, surgeVelocity=2.0):
        super(Circle_PID, self).__init__(boat)
        self._startingAngle = startingAngle
        self._surgeVelocity = surgeVelocity
        self._direction = direction
        self._R = radius
        self._center = center
        self.time = boat.time
        self._t0 = boat.time
        if self._direction == "cw":
            self._th0 = self._startingAngle - math.pi/2.0
            self._thetaDot = -surgeVelocity/radius
        else:  # self._direction == "ccw":
            self._th0 = self._startingAngle + math.pi/2.0
            self._thetaDot = surgeVelocity/radius
        self.controller = Controllers.PointAndShootPID(boat, 1.0)
        ths = np.linspace(-np.pi, np.pi, 100)
        self.boat.plotData = np.column_stack((center[0] + radius*np.cos(ths), center[1] + radius*np.sin(ths)))

    def idealState(self):
        self.time = self.boat.time
        totalTime = self.time - self._t0
        if self._direction == "cw":
            th = Controllers.wrapToPi(self._th0 - totalTime*self._thetaDot)
        else:
            th = Controllers.wrapToPi(self._th0 + totalTime*self._thetaDot)
        x = self._center[0] + self._R*math.cos(self._startingAngle + totalTime*self._thetaDot)
        y = self._center[1] + self._R*math.sin(self._startingAngle + totalTime*self._thetaDot)
        u = self._surgeVelocity
        w = 0.0
        thdot = self._thetaDot
        self.controller.idealState = np.array([x, y, u, w, th, thdot])


class DestinationOnlyExecutor(Executor):
    def __init__(self, boat, destination, positionThreshold=1.0):
        super(DestinationOnlyExecutor, self).__init__(boat)
        self._destination = destination
        self._positionThreshold = positionThreshold
        self.pickStrategy()

    def pickStrategy(self):
        # if boat is within 10 meters, do point THEN shoot strategy sequence
        # if boat is not, do point AND shoot
        state = self.boat.state
        dx = self._destination[0] - state[0]
        dy = self._destination[1] - state[1]
        distance = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))
        if distance < 10.0:
            self._strategy = StrategySequence(self.boat, [
                (ChangeHeading, (self.boat, math.atan2(dy, dx))),
                (DestinationOnly, (self.boat, self._destination, self._positionThreshold))
            ])
        else:
            self._strategy = DestinationOnly(self.boat, self._destination, self._positionThreshold)
        self._readyToPickStrategy = False  # only make this decision once!


class MoveToClosestAttacker(Strategy):
    def __init__(self, boat):
        super(MoveToClosestAttacker, self).__init__(boat)
        self.controller = Controllers.PointAndShootPID(boat, 1.0, False)

    def idealState(self):
        state = np.zeros((6,))
        if len(self._attackers) > 0:
            attackers = self._attackers
            X_defenders = np.zeros((1, 2))
            X_attackers = np.zeros((len(attackers), 2))
            X_defenders[0, 0] = self.boat.state[0]
            X_defenders[0, 1] = self.boat.state[1]
            for j in range(len(attackers)):
                boat = attackers[j]
                X_attackers[j, 0] = boat.state[0]
                X_attackers[j, 1] = boat.state[1]
            pairwise_distances = spatial.distance.cdist(X_defenders, X_attackers)
            closest_attacker = np.argmin(pairwise_distances, 1)
            state[0] = attackers[closest_attacker].state[0]
            state[1] = attackers[closest_attacker].state[1]
        else:
            state[0] = self.boat.state[0]
            state[1] = self.boat.state[1]
            state[2] = 0.0
            self.finished = True
            self.controller.finished = True
        self.controller.idealState = state
        self.boat.plotData = np.atleast_2d(np.array([
            [self.boat.state[0], self.boat.state[1]], [state[0], state[1]]
        ]))


# TODO - Executor that decides between a single spline, rotating in place first then a spline, or full point THEN shoot

# TODO - Defensive block: get onto a line of attack
class DefensiveBlock(Strategy):
    def __init__(self, boat, attacker, asset, ratioTowardAttacker=0.1):
        self.boat = boat
        self._attacker = attacker
        self._asset = asset
        self._ratioTowardAttacker = ratioTowardAttacker
        self._destination = [0.0, 0.0]
        self._strategy = DestinationOnly(boat, [0.0, 0.0])
        self._controller = self._strategy.controller

    @property  # need to override the standard controller property with the nested strategy's controller
    def controller(self):
        return self._strategy.controller

    @controller.setter  # need to override the standard controller property with the nested strategy's controller
    def controller(self, controller):
        self._controller = controller

    def updateFinished(self):  # need to override to get the finished status of the nested strategy!!!
        self.strategy.updateFinished()
        self.finished = self.strategy.finished

    def idealState(self):
        # draw a line between the attacker and the asset
        dx = self._attacker.state[0] - self._asset.state[0]
        dy = self._attacker.state[1] - self._asset.state[1]
        self._strategy.destinationState = [self._asset.state[0] + self._ratioTowardAttacker*dx,
                                           self._asset.state[1] + self._ratioTowardAttacker*dy]
        self._strategy.idealState()


class FollowWaypoints(Strategy):
    def __init__(self, boat, waypoints, headings=None, surgeVelocity=1.0, headingErrorSurgeCutoff=np.deg2rad(30.0),
                 lookAhead=0.05, positionThreshold=1.0, closed_circuit=False):
        super(FollowWaypoints, self).__init__(boat)
        self._boat = boat
        self._waypoints = waypoints
        self._headings = headings
        self._headingErrorSurgeCutoff = headingErrorSurgeCutoff
        self._closed_circuit = closed_circuit
        if closed_circuit:
            self._destination = None
        else:
            self._destination = waypoints[-1, :]  # the final waypoint
        self._surgeVelocity = surgeVelocity
        self._length = None
        self._sx = None
        self._sy = None
        self._sth = None
        self._totalLength = None
        self._u = None
        self._splineCoeffs = None
        self._totalLength = 0.0
        self._lookAhead = lookAhead
        self.generateSpline()
        self._controller = Controllers.LineOfSight(boat,
                                                   destination=self._destination,
                                                   positionThreshold=positionThreshold,
                                                   driftDown=False,
                                                   headingErrorSurgeCutoff=headingErrorSurgeCutoff)

    def generateSpline(self):
        if self._closed_circuit:
            self._sx, self._sy, self._sth, self._length, self._u, self._splineCoeffs = \
                PiazziSpline.splineClosedChain(self._waypoints, ths=self._headings)
        else:
            self._sx, self._sy, self._sth, self._length, self._u, self._splineCoeffs = \
                PiazziSpline.splineOpenChain(self._waypoints, ths=self._headings)
        self.boat.plotData = np.column_stack((self._sx, self._sy))
        self._totalLength = self._length[-1]

    def idealState(self):
        x = self.boat.state[0]
        y = self.boat.state[1]
        state = np.zeros((6,))
        try:
            u_star, global_angle = Utility.LOS_angle(
                    self._length, self._sx, self._sy, self._sth, self._u, self._splineCoeffs, x, y, lookAhead=self._lookAhead)
        except:
            self.controller.idealState = np.zeros((6,))
            return
        state[4] = global_angle
        tangent_th = np.interp(u_star, self._u, self._sth)
        dth_lookahead = max(0.0,
                            min(u_star + (1 + 0.5*self._surgeVelocity/self.boat.design.minSpeed)*self._lookAhead,
                                u_star + 3.*self._lookAhead))
        lookahead_th = np.interp(dth_lookahead, self._u, self._sth)
        clipped_lookahead_dth = np.clip(np.abs(lookahead_th - tangent_th), 0.0, self._headingErrorSurgeCutoff)
        surgeVelocityCap = max(self.boat.design.maxSpeed*math.cos(
                math.pi/2.0*clipped_lookahead_dth/self._headingErrorSurgeCutoff), self.boat.design.minSpeed)

        state[2] = min(surgeVelocityCap, self._surgeVelocity)
        self.controller.idealState = state