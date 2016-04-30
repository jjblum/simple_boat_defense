import numpy as np
import abc
import math

import Utility


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

"""
class ConstantFunction(object):
    # class with a single method that returns a specified constant
    def __init__(self, constant_in):
        self._constant = constant_in

    def f(self):
        return self._constant
"""

def dragDown(boat):
    # http://physics.stackexchange.com/questions/72503/how-do-i-calculate-the-distance-a-ship-will-take-to-stop
    # rho = 1000.0
    # L = boat.design.dragAreas[0]*rho*boat.design.dragCoeffs[0]/(2.0*boat.design.mass)
    # t_half = L/boat.state[2]
    # #surge drag = -1/L*surge velocity^2
    # #surge velocity = L/(t + t_half)
    # #t_half is the time it takes to reduce the speed to half the original value
    # #distance = L*ln( (t + t_half) / t_half )
    # #time required for 90% reduction in speed is 9*t_half
    # timeTo90pcReduction = 9.0*t_half
    # distanceTo90pcReduction = 2.3*L
    return boat.design.interpolateDragDown(boat.state[2])


class UniversalPID(object):
    def __init__(self, boat, P, I, D, t, name):
        self._boat = boat
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
        self._finished = False

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

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished_in):
        self._finished = finished_in


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
        self._headingPID = UniversalPID(boat, 1.0, 0.0, 1.0, boat.time, "heading_PID")

    def actuationEffortFractions(self):
        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state
        th = state[4]
        ideal_th = self.idealState[4]
        error_th = wrapToPi(th - ideal_th)

        error_th_signal = self._headingPID.signal(error_th, self.boat.time)
        self.time = self.boat.time

        momentFraction = np.clip(error_th_signal, -1.0, 1.0)

        if math.fabs(error_th) < np.deg2rad(5.0) and math.fabs(state[5]) < np.deg2rad(1.0):
            self.finished = True
            return 0.0, 0.0

        return 0.0, momentFraction


class SurgeAndHeadingPID(Controller):

    def __init__(self, boat):
        super(SurgeAndHeadingPID, self).__init__()
        self._boat = boat
        self.time = boat.time
        self._headingPID = UniversalPID(boat, 1.0, 0.0, 1.0, boat.time, "heading_PID")
        self._surgeVelocityPID = UniversalPID(boat, 2.0, 1.0, 0.0, boat.time, "surgeVelocity_PID")

    def actuationEffortFractions(self):
        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state

        error_x = self.idealState[0] - state[0]
        error_y = self.idealState[1] - state[1]
        error_u = state[2] - self.idealState[2]

        angleToGoal = math.atan2(error_y, error_x)
        error_th = wrapToPi(state[4] - angleToGoal)  # error between heading and heading to idealStates

        #print "boat {} heading error = {} \n\tdx = {}, dy = {}\n\tangleToGoal = {}".format(self.boat.uniqueID, error_th, error_x, error_y,angleToGoal)

        error_th_signal = self._headingPID.signal(error_th, self.boat.time)
        error_u_signal = -1.0*self._surgeVelocityPID.signal(error_u, self.boat.time)

        #if self.boat.type == "asset":
        #    print "u error = {}, u signal = {}".format(error_u, error_u_signal)

        self.time = self.boat.time

        momentFraction = np.clip(error_th_signal, -1.0, 1.0)
        # thrustFraction = np.clip(error_pos_signal, -1.0, 1.0)
        thrustFraction = np.clip(error_u_signal, -1.0, 1.0)

        return thrustFraction, momentFraction


class PointAndShootPID(Controller):

    def __init__(self, boat, positionThreshold_in, driftDown=True):
        super(PointAndShootPID, self).__init__()
        self._boat = boat
        self.time = boat.time
        self._positionThreshold = positionThreshold_in
        self._headingPID = UniversalPID(boat, 1.0, 0.0, 1.0, boat.time, "heading_PID")
        self._positionPID = UniversalPID(boat, 0.5, 0.01, 10.0, boat.time, "position_PID")
        #self._surgeVelocityPID = UniversalPID(boat, 1.0, 0.01, 10.0, boat.time, "surgeVelocity_PID")
        self._headingErrorSurgeCutoff = 90.0*math.pi/180.0  # thrust signal rolls off as a cosine, hitting zero here
        self._driftDown = driftDown

    def positionThreshold(self):
        return self._positionThreshold

    def positionThreshold(self, positionThreshold_in):
        self._positionThreshold = positionThreshold_in

    def actuationEffortFractions(self):
        thrustFraction = 0.0
        momentFraction = 0.0
        state = self.boat.state

        error_x = self.idealState[0] - state[0]
        error_y = self.idealState[1] - state[1]
        error_pos = math.sqrt(math.pow(error_x, 2.0) + math.pow(error_y, 2.0))

        # if the position error is less than some threshold and velocity is near zero, turn thrustFraction to 0
        if error_pos < self._positionThreshold and math.sqrt(math.pow(state[2], 2.0) + math.pow(state[3], 2.0)) < 0.1:
            # because this is where we might set finished to True, it
            # needs to be before any other returns that might make it impossible to reach
            self.finished = True
            return 0.0, 0.0

        if self.finished:
            return 0.0, 0.0

        angleToGoal = math.atan2(error_y, error_x)
        error_th = wrapToPi(state[4] - angleToGoal)  # error between heading and heading to idealStates

        # if the angle error is low (i.e. pointing at the goal), calculate drag down time with surge velocity
        # From that, calculate drag down distance
        # Once position error hits that distance, set thrustFraction to 0
        if self._driftDown:
            if math.fabs(error_th) < 2.0*math.pi/180.0 and math.fabs(state[5]) < 0.5*math.pi/180.0:
                dragDownTime, dragDownDistance = dragDown(self.boat)
                if error_pos < dragDownDistance:
                    #print "distance = {}, dragDownDistance = {}, DRAG DOWN... u = {}" \
                    #    .format(error_pos, dragDownDistance, self.boat.state[2])
                    return 0.0, 0.0

        error_th_signal = self._headingPID.signal(error_th, self.boat.time)
        error_pos_signal = self._positionPID.signal(error_pos, self.boat.time)

        self.time = self.boat.time

        clippedAngleError = np.clip(math.fabs(error_th), 0.0, self._headingErrorSurgeCutoff)
        thrustReductionRatio = math.cos(math.pi/2.0*clippedAngleError/self._headingErrorSurgeCutoff)
        momentFraction = np.clip(error_th_signal, -1.0, 1.0)
        thrustFraction = np.clip(error_pos_signal, -thrustReductionRatio, thrustReductionRatio)

        return thrustFraction, momentFraction


class LineOfSight(Controller):

    def __init__(self, boat, destination=None, headingErrorSurgeCutoff=np.deg2rad(30.0), positionThreshold=1.0, driftDown=True):
        super(LineOfSight, self).__init__()
        self.boat = boat
        self._headingPID = UniversalPID(boat, 0.5, 0.0, 5.0, boat.time, "heading_PID")
        self._surgeVelocityPID = UniversalPID(boat, 1.0, 0.1, 0.1, boat.time, "surgeVelocity_PID")
        self._headingErrorSurgeCutoff = headingErrorSurgeCutoff  # thrust signal rolls off as a cosine, hitting zero here
        self._destination = destination
        self._positionThreshold = positionThreshold
        self._driftDown = driftDown
        self._remainingDistance = 0.0

    @property
    def remainingDistance(self):
        return self._remainingDistance

    @remainingDistance.setter
    def remainingDistance(self, remainingDistance_in):
        self._remainingDistance = remainingDistance_in

    def actuationEffortFractions(self):

        # the strategy is the part where the goal angle is calculated, so this should be super simple, just the PID output
        error_th = wrapToPi(self.boat.state[4] - self.idealState[4])
        # print "error_th = {:.2f} deg".format(np.rad2deg(error_th))
        clippedAngleError = np.clip(math.fabs(error_th), 0.0, self._headingErrorSurgeCutoff)
        thrustReductionRatio = math.cos(math.pi/2.0*clippedAngleError/self._headingErrorSurgeCutoff)
        error_th_signal = self._headingPID.signal(error_th, self.boat.time)
        error_u_signal = self._surgeVelocityPID.signal(self.idealState[2] - self.boat.state[2], self.boat.time)
        momentFraction = np.clip(error_th_signal, -1.0, 1.0)
        thrustFraction = thrustReductionRatio*np.clip(error_u_signal, -1.0, 1.0)
        #print "thrustFraction = {}  momentFraction = {}".format(thrustFraction, momentFraction)
        if self.finished:
            return 0.0, 0.0
        # if distance to the goal is less than some threshold
        if self._destination is not None:
            distanceToGoal = math.sqrt(math.pow(self._destination[0] - self.boat.state[0], 2) +
                                       math.pow(self._destination[1] - self.boat.state[1], 2))
            if distanceToGoal < self._positionThreshold:
                self.finished = True
                thrustFraction = 0.0

        """
        # if the angle error is low (i.e. pointing at the goal), calculate drag down time with surge velocity
        # From that, calculate drag down distance
        # Once position error hits that distance, set thrustFraction to 0
        if self._driftDown:
            if error_th < 5.0*math.pi/180.0 and math.fabs(self.boat.state[5]) < 0.5*math.pi/180.0:
                dragDownTime, dragDownDistance = dragDown(self.boat)
                if self._remainingDistance < dragDownDistance:
                    #print "distance = {}, dragDownDistance = {}, DRAG DOWN... u = {}" \
                    #    .format(error_pos, dragDownDistance, self.boat.state[2])
                    thrustFraction = 0.0
        """
        return thrustFraction, momentFraction
