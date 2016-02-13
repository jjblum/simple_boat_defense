import numpy
import abc
import math
import Boat
import copy

def wrapToPi(angle):
    return (angle + numpy.pi) % (2 * numpy.pi) - numpy.pi


class UniversalPID(object):
    def __init__(self, P, I, D, t, name):
        self._P = P
        self._I = I
        self._D = D
        self._t = t
        self._tOld = t
        self._errorDerivative = 0.0
        self._errorAccumulation = 0.0
        self._errorOld = 0.0
        self._name = name

    def signal(self, error, t):
        dt = t - self._t
        self._t = t
        self._errorDerivative = 0.0
        if dt > 0:
            self._errorDerivative = (error - self._errorOld)/dt
        self._errorAccumulation += dt*error
        return self._P*error + self._I*self._errorAccumulation + self._D*self._errorDerivative



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
        self._headingPID = UniversalPID(1.0, 0.0, 20.0, boat.time, "heading_PID")

    def actuationEffortFractions(self):
        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state
        th = state[4]
        ideal_th = self.idealState[4]
        error_th = wrapToPi(th - ideal_th)

        error_th_signal = self._headingPID.signal(error_th, self.boat.time)
        self.time = self.boat.time

        momentFraction = numpy.clip(error_th_signal, -1.0, 1.0)


        if math.fabs(error_th) < 1.0*math.pi/180.0 and math.fabs(state[5]) < 0.5*math.pi/180.0:
            # angle error and angluar speed are both very low, turn off the controller
            # self.boat.strategy.controller = DoNothing()
            return 0.0, 0.0

        return 0.0, momentFraction


class PointAndShootPID(Controller):

    def __init__(self, boat):
        super(PointAndShootPID, self).__init__()
        self._boat = boat
        self.time = boat.time
        self._headingPID = UniversalPID(1.0, 0.0, 20.0, boat.time, "heading_PID")
        # self._positionPID = UniversalPID(1.0, 0.0, 0.0, boat.time, "position_PID")
        self._surgeVelocityPID = UniversalPID(10.0, 0.0, 0.0, boat.time, "surgeVelocity_PID")

    def actuationEffortFractions(self):
        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state

        error_x = state[0] - self.idealState[0]
        error_y = state[1] - self.idealState[1]
        error_pos = math.sqrt(math.pow(error_x, 2.0) + math.pow(error_y, 2.0))
        error_u = state[2] - self.idealState[2]

        angleToGoal = math.atan2(error_y, error_x)
        error_th = state[4] - angleToGoal  # error between heading and heading to idealStates

        error_th_signal = self._headingPID.signal(error_th, self.boat.time)
        # error_pos_signal = self._positionPID.signal(error_pos, self.boat.time)
        error_u_signal = -1.0*self._headingPID.signal(error_u, self.boat.time)

        print "u error = {}, u signal = {}".format(error_u, error_u_signal)

        self.time = self.boat.time

        momentFraction = numpy.clip(error_th_signal, -1.0, 1.0)
        # thrustFraction = numpy.clip(error_pos_signal, -1.0, 1.0)
        thrustFraction = numpy.clip(error_u_signal, -1.0, 1.0)

        return thrustFraction, momentFraction
