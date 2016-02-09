import numpy
import abc
import math
import Boat


def wrapToPi(angle):
    return (angle + numpy.pi) % (2 * numpy.pi) - numpy.pi


class Controller(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._t = 0.0
        self._boat = None
        self._idealState = []
        self._thrustFraction = 0.0
        self._momentFraction = 0.0

    @abc.abstractmethod
    def actuationEffortFractions(self):
        # virtual function, uses current state and ideal state to generate actuation effort
        # PID control, trajectory following, etc.
        return

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t

    @property
    def boat(self):
        return self._boat

    @boat.setter
    def boat(self, boat_in):
        self._boat = boat_in

    @property
    def idealState(self):
        return self._idealState

    @idealState.setter
    def idealState(self, idealState_in):
        self._idealState = idealState_in

    @property
    def thrustFraction(self):
        return self._thrustFraction

    @thrustFraction.setter
    def thrustFraction(self, thrustFraction_in):
        self._thrustFraction = thrustFraction_in

    @property
    def momentFraction(self):
        return self._momentFraction

    @momentFraction.setter
    def momentFraction(self, momentFraction_in):
        self._momentFraction = momentFraction_in


class DoNothing(Controller):
    def __init__(self):
        super(DoNothing, self).__init__()
    def actuationEffortFractions(self):
        return 0.0, 0.0


class HeadingOnlyPID(Controller):
    def __init__(self, boat):
        super(HeadingOnlyPID, self).__init__()
        self._boat = boat
        self.time = boat.time
        self._error_th_old = 0.0
        self._error_th_accum = 0.0

    def actuationEffortFractions(self):
        dt = self.boat.time - self.time
        self.time = self.boat.time

        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state
        th = state[4]
        ideal_th = self.idealState[4]
        error_th = wrapToPi(th - ideal_th)
        P = 1.0
        I = 0.0
        D = 20.0
        error_th_dot = 0.0
        if dt > 0:
            error_th_dot = (error_th - self._error_th_old)/dt
            self._error_th_old = error_th
        self._error_th_accum += dt*error_th
        error_th_signal = P*error_th + I*self._error_th_accum + D*error_th_dot
        momentFraction = numpy.clip(error_th_signal, -1.0, 1.0)

        if math.fabs(error_th) < 1.0*math.pi/180.0 and math.fabs(state[5]) < 0.5*math.pi/180.0:
            # angle error and angluar speed are both very low, turn off the controller
            self.boat.strategy.controller = DoNothing()
            return 0.0, 0.0

        return 0.0, momentFraction


class PointAndShootPID(Controller):

    def __init__(self, boat):
        super(PointAndShootPID, self).__init__()
        self._boat = boat
        self.time = boat.time

    def actuationEffortFractions(self):
        dt = self.boat.time - self.time
        self.time = self.boat.time

        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state

        error_x = state[0] - self.idealState[0]
        error_y = state[1] - self.idealState[1]
        error_u = state[2] - self.idealState[2]
        angleToGoal = math.atan2(error_y, error_x)
        error_th = state[4] - angleToGoal #  error between heading and heading to idealStates

        return thrustFraction, momentFraction