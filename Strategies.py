import abc  # abstract base classes
import math
import numpy
import Controllers


# TODO - updating the strategy effectively deletes all the asset, attacker, defender, etc. information that the last strategy held
#        obviously that isn't acceptable. We need to make sure the next strategy doesn't lose any information!
#  def strategyBatonPass(strategy_in, strategy_out_type):



class Strategy(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, boat):
        self._boat = boat
        self._controller = None
        self._t = 0.0
        self._assets = boat.assets
        self._attackers = boat.attackers
        self._defenders = boat.defenders

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
    def __init__(self, boat, surgeVelocity=0.0, t0=0.0):
        super(HoldHeading, self).__init__(boat)
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
    def __init__(self, boat):
        super(DestinationOnly, self).__init__(boat)
        self._destinationState = numpy.zeros((6,))

    @property
    def destinationState(self):
        return self._destinationState # as of now, even a high level strategy needs to have a handle to the controller it will ultimately use

    @destinationState.setter
    def destinationState(self, destinationState_in):
        self._destinationState = destinationState_in

    def idealState(self):
        self.controller.idealState = self.destinationState  # update this here so the controller doesn't need to import Strategies
        return self.destinationState


class PointAtAsset(Strategy):
    # a strategy that just points the boat at the geometric mean of the assets
    # an example of a NESTED strategy
    def __init__(self, boat):
        super(PointAtAsset, self).__init__(boat)
        self._strategy = ChangeHeading(boat, 0.0)  # the lower level nested strategy
        self.controller = self._strategy.controller

    @property
    def controller(self):
        return self._strategy.controller

    @controller.setter
    def controller(self, controller):
        self._controller = controller

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



