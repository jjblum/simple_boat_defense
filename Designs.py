import numpy
import math
import abc
import Boat

# TODO - figure out why linear drag is completely dominating the drag down times!
#        This seems super unrealistic. Nonlinear drag is getting it into the linear regime quickly,
#        and then linear drag dominates in a ridiculous way. Fix it.


class Design(object):
    # abstract class, a design dictates how actuation fractions are translated into actual thrust and moment
    # e.g. a tank-drive propeller boat will behave differently than a vectored-thrust boat
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._mass = 0.0  # [kg]
        self._momentOfInertia = 0.0  # [kg/m^2]
        self._dragAreas = [0.0, 0.0, 0.0]  # surge, sway, rotation [m^2]
        self._dragCoeffs = [0.0, 0.0, 0.0]  # surge, sway, rotation [-]

    @abc.abstractmethod
    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        # virtual, calculate the thrust and moment a specific boat design can accomplish
        return

    @property
    def mass(self):
        return self._mass

    @property
    def momentOfInertia(self):
        return self._momentOfInertia

    @property
    def dragAreas(self):
        return self._dragAreas

    @property
    def dragCoeffs(self):
        return self._dragCoeffs

    @property
    def dragAreas(self):
        return self._dragAreas

    @property
    def dragCoeffs(self):
        return self._dragCoeffs


class Lutra(Design):
    def __init__(self):
        super(Lutra, self).__init__()
        self._mass = 5.7833  # [kg]
        self._momentOfInertia = 0.6  # [kg/m^2]
        self._dragAreas = [0.0108589939, 0.0424551192, 0.0424551192]  # surge, sway, rotation [m^2]
        # self._dragCoeffs = [0.258717640651218, 1.088145891415693, 0.048292066650533]  # surge, sway, rotation [-]
        self._dragCoeffs = [1.5, 1.088145891415693, 2.0]  # surge, sway, rotation [-]
        self._dragDownCurve = numpy.zeros((7, 3))  # u0, time to u = 0.01, d to u = 0.01
        self._dragDownCurve[0, :] = numpy.array([0.1, 2.55, 0.1])
        self._dragDownCurve[1, :] = numpy.array([0.2, 3.2, 0.19])
        self._dragDownCurve[2, :] = numpy.array([0.5, 4.8, 0.73])
        self._dragDownCurve[3, :] = numpy.array([1.0, 5.5, 1.22])
        self._dragDownCurve[4, :] = numpy.array([1.5, 5.75, 1.51])
        self._dragDownCurve[5, :] = numpy.array([2.0, 5.85, 1.71])
        self._dragDownCurve[6, :] = numpy.array([2.5, 5.95, 1.90])

    def interpolateDragDown(self, u0):
        time = numpy.interp(u0, self._dragDownCurve[:, 0], self._dragDownCurve[:, 1])
        distance = numpy.interp(u0, self._dragDownCurve[:, 0], self._dragDownCurve[:, 2])
        return time, distance


class TankDriveDesign(Lutra):
    def __init__(self):
        super(TankDriveDesign, self).__init__()
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


class VectoredThrustDesign(Lutra):
    def __init__(self):
        super(VectoredThrustDesign, self).__init__()
        self._maxFanThrust = 30.0  # [N]
        self._minFanThrust = 0.0  # assume no backwards thrust
        self._maxAngle = 80.0*math.pi/180.0  # maximum thrust angle [rad]

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        thrustSurge = 0.0
        thrustSway = 0.0
        moment = 0.0
        # TODO - finish vectored thrust design
        return thrustSurge, thrustSway, moment