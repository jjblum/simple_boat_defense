import numpy
import math
import abc
import Boat


class Design(object):
    # abstract class, a design dictates how actuation fractions are translated into actual thrust and moment
    # e.g. a tank-drive propeller boat will behave differently than a vectored-thrust boat
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        # virtual, calculate the thrust and moment a specific boat design can accomplish
        return

class TankDriveDesign(Design):
    def __init__(self):
        self._maxThrustPerMotor = 25.0  # [N]
        # self._minThrustPerMotor = 0.0  # assume no drop in thrust for backdriving
        self._momentArm = 0.3556  # distance between the motors [m]

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        thrustSway = 0.0

        m0 = numpy.clip(thrustFraction + momentFraction, -1.0, 1.0)
        m1 = numpy.clip(thrustFraction - momentFraction, -1.0, 1.0)

        thrustSurge = self._maxThrustPerMotor*(m0 + m1)
        moment = self._maxThrustPerMotor*(m1 - m0)/2.0*self._momentArm

        return thrustSurge, thrustSway, moment


class VectoredTrustDesign(Design):
    def __init__(self):
        self._maxFanThrust = 30.0  # [N]
        self._minFanThrust = 0.0  # assume no backwards thrust
        self._maxAngle = 80.0*math.pi/180.0  # maximum thrust angle [rad]

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        thrustSurge = 0.0
        thrustSway = 0.0
        moment = 0.0
        # TODO - finish vectored thrust design
        return thrustSurge, thrustSway, moment