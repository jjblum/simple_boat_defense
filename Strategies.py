import abc  # abstract base classes
import math
import numpy
import Controllers

# TODO - what if instead of a distinct controller, a strategy creates the entire controller?
#        The strategy might dictate the controller.
#        For example, a strategy might return just the final destination and expect
#        point+shoot PID to make its way there. Or it might require a specific set of
#        waypoints and use a spline generator such that you'd want a path following controller.
#        So (ironically?) the strategy "controls" the controller.
#        Should the controller be in the strategy object itself?
#        For example, a PID control could follow a path, but a path follower can't just get
#        to a single destination without generating a spline to follow


class Strategy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._boat = None
        self._controller = None
        self._t = 0.0
        self._assets = []
        self._attackers = []
        self._defenders = []

    @abc.abstractmethod
    def idealState(self):
        # virtual function, uses information to return an ideal state
        # this will be used for fox-rabbit style control
        return

    @property
    def controller(self):
        return self._controller

    @controller.setter
    def controller(self, controller_in):
        self._controller = controller_in

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t
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


class DoNothing(Strategy):
    # a strategy that prevents actuation
    def __init__(self):
        super(DoNothing, self).__init__()
        self.controller = Controllers.DoNothing()

    def idealState(self):
        return numpy.zeros((6,))


class StationKeep(Strategy):
    # a strategy that just sets the destination to the current location
    def __init__(self, boat):
        super(StationKeep, self).__init__()
        self._boat = boat  # the boat object that owns this Strategy
        self.controller = Controllers.PointAndShootPID(boat)

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
    def __init__(self, boat, heading):
        super(ChangeHeading, self).__init__()
        self.boat = boat
        self._desiredHeading = heading
        self.controller = Controllers.HeadingOnlyPID(boat)

    def idealState(self):
        # rabbit boat is at the boat's current location, just rotated
        state = numpy.zeros((6,))
        state[0] = self.boat.state[0]
        state[1] = self.boat.state[1]
        state[4] = self._desiredHeading
        self.controller.idealState = state


class HoldHeading(Strategy):
    # a strategy where the ideal boat moves with a fixed velocity along the boat's current heading
    def __init__(self, boat, surgeVelocity=0.0, t0=0.0):
        super(HoldHeading, self).__init__()
        self._boat = boat  # the boat object that owns this Strategy
        self._t0 = t0  # time when this strategy started
        self._surgeVelocity = surgeVelocity  # [m/s]
        self.controller = Controllers.PointAndShootPID(boat)

    def idealState(self):
        # rabbit boat moves forward at fixed velocity
        state = numpy.zeros((6,))
        x = self.boat.state[0]
        y = self.boat.state[1]
        u = self._surgeVelocity
        w = self.boat.state[3]
        th = self.boat.state[4]
        thdot = self.boat.state[5]
        time_expired = self.time - self._t0
        state[0] = x + u*math.cos(th)*time_expired
        state[1] = y + u*math.sin(th)*time_expired
        state[2] = u
        state[3] = w
        state[4] = self.boat.state[4]
        state[5] = 0
        self.controller.idealState = state  # update this here so the controller doesn't need to import Strategies
        return state


class DestinationOnly(Strategy):
    # a strategy that only returns the final destination location
    def __init__(self):
        super(DestinationOnly, self).__init__()
        self._destinationState = numpy.zeros((6,))

    @property
    def destinationState(self):
        return self._destinationState

    @destinationState.setter
    def destinationState(self, destinationState_in):
        self._destinationState = destinationState_in

    def idealState(self):
        self.controller.idealState = self.destinationState  # update this here so the controller doesn't need to import Strategies
        return self.destinationState

