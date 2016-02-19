import abc  # abstract base classes
import math
import numpy
import Controllers

# TODO - updating the strategy effectively deletes all the asset, attacker, defender, etc. information that the last strategy held
#        is there a way to have that information transferred from stategy to strategy?

# TODO - set up a timed strategy, it executes for a specified amount of time then changes

# TODO - an "executive" or "team" strategy that sets the strategy of more than one individuals

# TODO - LIST OF USEFUL STRATEGIES:
#   1a) Move to asset
#   1b) Follow trajectory toward asset
#   1c) Move in a circle around asset (could be the same as perimeter patrol)
#   2a) Point away from asset
#   2b) Align heading with asset's heading
#   3) Get into an ellipse around asset (extend long axis according to asset speed)
#   4) Intercept - linear assumption - cast a ray where target may go, find where along that trajectory you can reach
#   5) Patrol perimeter - ideal boat follows a circuit path at a specified velocity and direction
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
        self._t = 0.0
        self._assets = boat.assets
        self._attackers = boat.attackers
        self._defenders = boat.defenders
        self._strategy = self  # returns self by default (unless it is a nested strategy or sequence)

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
        return self._controller.actuationEffortFractions()

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
    def __init__(self, boat, sequence):
        super(StrategySequence, self).__init__(boat)
        self._strategySequence = sequence
        self._currentStrategy = 0  # index of the current strategy
        self._strategy = self._strategySequence[0].strategy
        self.controller = self._strategy.controller

    @property
    def strategySequence(self):
        return self._strategySequence

    @strategySequence.setter
    def strategySequence(self, strategySequence_in):
        self._strategySequence = strategySequence_in

    # override
    def actuationEffortFractions(self):
        return self._strategySequence[self._currentStrategy].actuationEffortFractions()

    # override
    def updateFinished(self):
        self._strategySequence[self._currentStrategy].updateFinished()
        if self._strategySequence[self._currentStrategy].finished and \
                self._currentStrategy < len(self.strategySequence) - 1:
            self._currentStrategy += 1
            # must manually update strategy and controller!
            self._strategy = self._strategySequence[self._currentStrategy]
            self.controller = self.strategy.controller
        if self._strategySequence[-1].finished:
            # sequence is finished when last strategy in a sequence is finished
            self.finished = True

    def idealState(self):
        return self._strategySequence[self._currentStrategy].idealState()


class TimedStrategySequence(StrategySequence):
    def __init__(self, boat, sequence, timing):
        super(StrategySequence, self).__init__(boat)
        self._strategySequence = sequence
        self._strategyTiming = timing
        self._currentStrategyStartTime = boat.time
        self._currentStrategy = 0  # index of the current strategy
        self._strategy = self._strategySequence[0].strategy
        self.controller = self._strategy.controller

    # override
    def updateFinished(self):
        # notice the OR - the timing represents a timeout
        self._strategySequence[self._currentStrategy].updateFinished()
        dt = (self.boat.time - self._currentStrategyStartTime)

        if self._strategySequence[-1].finished or \
                (self._currentStrategy == len(self.strategySequence) - 1 and
                 dt >= self._strategyTiming[self._currentStrategy]):
            # sequence is finished when last strategy in a sequence is finished or total time has run out
            self._strategySequence[self._currentStrategy] = DoNothing(self.boat)
            self.finished = True

        if (self._strategySequence[self._currentStrategy].finished or
                dt >= self._strategyTiming[self._currentStrategy]) and \
                self._currentStrategy < len(self.strategySequence) - 1:

            self._strategySequence[self._currentStrategy].finished = True
            self._currentStrategy += 1
            # must manually update strategy and controller!
            self._strategy = self._strategySequence[self._currentStrategy]
            self.controller = self.strategy.controller
            self._currentStrategyStartTime = self.boat.time


class DoNothing(Strategy):
    # a strategy that prevents actuation
    def __init__(self, boat):
        super(DoNothing, self).__init__(boat)
        self.controller = Controllers.DoNothing()

    def idealState(self):
        return numpy.zeros((6,))


class StationKeep(Strategy):
    # a strategy that just sets the destination to the current location
    def __init__(self, boat):
        super(StationKeep, self).__init__()
        self.controller = Controllers.SurgeAndHeadingPID(boat)

    def idealState(self):
        # rabbit boat sits at the boats current location
        state = numpy.zeros((6,))
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
        state = numpy.zeros((6,))
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
        state = numpy.zeros((6,))
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


class DestinationOnly(Strategy):
    # a strategy that only returns the final destination location
    def __init__(self, boat, destination, positionThreshhold):
        super(DestinationOnly, self).__init__(boat)
        self._destinationState = None
        self.destinationState = destination
        self.controller = Controllers.PointAndShootPID(boat, positionThreshhold)

    @property
    def destinationState(self):
        return self._destinationState  # as of now, even a high level strategy needs to have a handle to the controller it will ultimately use

    @destinationState.setter
    def destinationState(self, destinationState_in):
        if len(destinationState_in) == 6:
            self._destinationState = destinationState_in
        elif len(destinationState_in) == 3:
            # assuming they are using x, y, th
            state = numpy.zeros((6,))
            state[0] = destinationState_in[0]
            state[1] = destinationState_in[1]
            state[4] = destinationState_in[2]
            self._destinationState = state
        elif len(destinationState_in) == 2:
            # assuming they are using x, y
            state = numpy.zeros((6,))
            state[0] = destinationState_in[0]
            state[1] = destinationState_in[1]
            self._destinationState = state

    def idealState(self):
        self.controller.idealState = self.destinationState  # update this here so the controller doesn't need to import Strategies
        return self.destinationState


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
        asset_x = numpy.mean(assets_x)
        asset_y = numpy.mean(assets_y)
        return math.atan2(asset_y - y, asset_x - x)

    def idealState(self):
        self._strategy.desiredHeading = self.angleToAsset()
        return self._strategy.idealState()


class MoveTowardAsset(Strategy):
    # nested strategy - uses DestinationOnly with the asset as the goal
    def __init__(self, boat, positionThreshhold):
        super(MoveTowardAsset, self).__init__(boat)
        self._strategy = DestinationOnly(boat, [self.assets[0].state[0], self.assets[0].state[1]], positionThreshhold)  # the lower level nested strategy
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
        return self._strategy.idealState()


class Square(StrategySequence):
    # Move around the vertices of a square given its center and edge size.
    # User can specify which corner and rotation direction they want
    # This serves as an example of a Strategy built from a sequence of more primitive Strategies and sequences
    def __init__(self, boat, positionThreshhold, center, edgeSize, firstCorner="bottom_right", direction="cw"):
        super(StrategySequence, self).__init__(boat)  # run Strategy __init__, NOT StrategySequence.__init__ !!!
        self._strategySequence = list()
        self._currentStrategy = 0  # index of the current strategy

        bottom_left = list(numpy.array(center) + numpy.array([-edgeSize/2.0, -edgeSize/2.0]))
        bottom_right = list(numpy.array(center) + numpy.array([edgeSize/2.0, -edgeSize/2.0]))
        upper_right = list(numpy.array(center) + numpy.array([edgeSize/2.0, edgeSize/2.0]))
        upper_left = list(numpy.array(center) + numpy.array([-edgeSize/2.0, edgeSize/2.0]))
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
            DestinationOnly(boat, vertices[0], positionThreshhold),
            StrategySequence(boat, [ChangeHeading(boat, headings[0]), DestinationOnly(boat, vertices[1], positionThreshhold)]),
            StrategySequence(boat, [ChangeHeading(boat, headings[1]), DestinationOnly(boat, vertices[2], positionThreshhold)]),
            StrategySequence(boat, [ChangeHeading(boat, headings[2]), DestinationOnly(boat, vertices[3], positionThreshhold)]),
            StrategySequence(boat, [ChangeHeading(boat, headings[3]), DestinationOnly(boat, vertices[4], positionThreshhold)])
        ]
        self._strategy = self._strategySequence[0].strategy
        self.controller = self._strategy.controller
