import numpy as np
import math
import abc


class Design(object):
    # abstract class, a design dictates how actuation fractions are translated into actual thrust and moment
    # e.g. a tank-drive propeller boat will behave differently than a vectored-thrust boat
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self._mass = 0.0  # [kg]
        self._momentOfInertia = 0.0  # [kg/m^2]
        self._dragAreas = [0.0, 0.0, 0.0]  # surge, sway, rotation [m^2]
        self._dragCoeffs = [0.0, 0.0, 0.0]  # surge, sway, rotation [-]
        self._maxSpeed = 0.0  # [m/s]
        self._minSpeed = 0.0
        self._maxHeadingRate = 0.0  # maximum turning speed [rad/s]
        self._maxForwardThrust = 0.0
        self._speedVsMinRadius = np.zeros((1, 2))  # 2 column array, speed vs. min turning radius

    @abc.abstractmethod
    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        # virtual, calculate the thrust and moment a specific boat design can accomplish
        return

    def minTurnRadius(self, speed):
        return np.interp(speed, self._speedVsMinRadius[:, 0], self._speedVsMinRadius[:, 1])

    @property
    def maxForwardThrust(self):
        return self._maxForwardThrust

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

    @property
    def maxSpeed(self):
        return self._maxSpeed

    @property
    def minSpeed(self):
        return self._minSpeed

    @property
    def maxHeadingRate(self):
        return self._maxHeadingRate


class Lutra(Design):
    def __init__(self):
        super(Lutra, self).__init__()
        self._mass = 5.7833  # [kg]
        self._momentOfInertia = 0.6  # [kg/m^2]
        self._maxSpeed = 2.5  # [m/s]
        self._minSpeed = 0.25  # [m/s]
        self._dragAreas = [0.0108589939, 0.0424551192, 0.0424551192]  # surge, sway, rotation [m^2]
        # self._dragCoeffs = [0.258717640651218, 1.088145891415693, 0.048292066650533]  # surge, sway, rotation [-]
        self._dragCoeffs = [1.5, 1.088145891415693, 2.0]  # surge, sway, rotation [-]
        self._dragDownCurve = np.zeros((7, 3))  # u0, time to u = 0.01, d to u = 0.01
        self._dragDownCurve[0, :] = np.array([0.1, 2.55, 0.1])
        self._dragDownCurve[1, :] = np.array([0.2, 3.2, 0.19])
        self._dragDownCurve[2, :] = np.array([0.5, 4.8, 0.73])
        self._dragDownCurve[3, :] = np.array([1.0, 5.5, 1.22])
        self._dragDownCurve[4, :] = np.array([1.5, 5.75, 1.51])
        self._dragDownCurve[5, :] = np.array([2.0, 5.85, 1.71])
        self._dragDownCurve[6, :] = np.array([2.5, 5.95, 1.90])

    def interpolateDragDown(self, u0):
        time = np.interp(u0, self._dragDownCurve[:, 0], self._dragDownCurve[:, 1])
        distance = np.interp(u0, self._dragDownCurve[:, 0], self._dragDownCurve[:, 2])
        return time, distance


class HighMassTankDriveDesign(Lutra):
    def __init__(self):
        super(HighMassTankDriveDesign, self).__init__()
        self._mass = 5.7833*1.5
        self._momentOfInertia = 0.6*1.5  # [kg/m^2]
        self._maxSpeed = 2.5  # [m/s]
        self._maxThrustPerMotor = 25.0  # [N]
        # self._minThrustPerMotor = 0.0  # assume no drop in thrust for backdriving
        self._momentArm = 0.3556  # distance between the motors [m]
        self._maxForwardThrust = 2.*self._maxThrustPerMotor
        """
        self._speedVsMinRadius = np.array([
            [0.84, 2.0],
            [1.18, 3.0],
            [1.37, 3.75],
            [1.57, 4.5],
            [1.84, 6.0],
            [2.12, 9.0],
            [2.18, 10.0],
            [2.31, 20.0],
            [2.39, 30.0]
        ])
        """
        self._dragDownCurve = np.zeros((7, 3))  # u0, time to u = 0.01, d to u = 0.01
        self._dragDownCurve[0, :] = np.array([0.1, 4.05, 0.15])
        self._dragDownCurve[1, :] = np.array([0.2, 5.05, 0.29])
        self._dragDownCurve[2, :] = np.array([0.5, 7.45, 1.09])
        self._dragDownCurve[3, :] = np.array([1.0, 8.55, 1.83])
        self._dragDownCurve[4, :] = np.array([1.5, 8.9, 2.26])
        self._dragDownCurve[5, :] = np.array([2.0, 9.05, 2.57])
        self._dragDownCurve[6, :] = np.array([2.5, 9.15, 2.81])
        # below 1 m/s, you should probably just turn in place!
        self._maxHeadingRate = 0.403  # [rad/s]
        self._thCoeff = 2.58862349223
        self._rCoeff = 0.39429396374
        self._u0Coeff = 0.0698199590918

    @property
    def thCoeff(self):
        return self._thCoeff
    @property
    def rCoeff(self):
        return self._rCoeff
    @property
    def u0Coeff(self):
        return self._u0Coeff

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        thrustSway = 0.0

        m0 = np.clip(thrustFraction + momentFraction, -1.0, 1.0)
        m1 = np.clip(thrustFraction - momentFraction, -1.0, 1.0)

        thrustSurge = self._maxThrustPerMotor*(m0 + m1)
        moment = self._maxThrustPerMotor*(m1 - m0)/2.0*self._momentArm

        return thrustSurge, thrustSway, moment


class TankDriveDesign(Lutra):
    def __init__(self):
        super(TankDriveDesign, self).__init__()
        self._maxThrustPerMotor = 25.0  # [N]
        # self._minThrustPerMotor = 0.0  # assume no drop in thrust for backdriving
        self._momentArm = 0.3556  # distance between the motors [m]
        self._maxForwardThrust = 2.*self._maxThrustPerMotor
        self._speedVsMinRadius = np.array([
            [0.84, 2.0],
            [1.18, 3.0],
            [1.37, 3.75],
            [1.57, 4.5],
            [1.84, 6.0],
            [2.12, 9.0],
            [2.18, 10.0],
            [2.31, 20.0],
            [2.39, 30.0]
        ])
        # below 1 m/s, you should probably just turn in place!
        self._maxHeadingRate = 0.403  # [rad/s]
        # TODO - turning ccw turns FASTER than turning cw???? Figure out why. Surge velocity doesn't show this.
        self._thCoeff = 2.54832785865
        self._rCoeff = 0.401354269952
        self._u0Coeff = 0.0914788305811

    @property
    def thCoeff(self):
        return self._thCoeff
    @property
    def rCoeff(self):
        return self._rCoeff
    @property
    def u0Coeff(self):
        return self._u0Coeff

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        thrustSway = 0.0

        m0 = np.clip(thrustFraction + momentFraction, -1.0, 1.0)
        m1 = np.clip(thrustFraction - momentFraction, -1.0, 1.0)

        thrustSurge = self._maxThrustPerMotor*(m0 + m1)
        moment = self._maxThrustPerMotor*(m1 - m0)/2.0*self._momentArm

        return thrustSurge, thrustSway, moment

    @property
    def speedVSMinRadius(self):
        return self._speedVsMinRadius

    def maxCircularSpeed(self, circleRadius):
        return np.interp(circleRadius, self.speedVSMinRadius[:, 1], self.speedVSMinRadius[:, 0])


class VectoredThrustDesign(Lutra):
    def __init__(self):
        super(VectoredThrustDesign, self).__init__()
        self._maxFanThrust = 30.0  # [N]
        self._minFanThrust = 0.0  # assume no backwards thrust
        self._maxAngle = 80.0*math.pi/180.0  # maximum thrust angle [rad]
        self._maxForwardThrust = self._maxFanThrust

    def thrustAndMomentFromFractions(self, thrustFraction, momentFraction):
        thrustSurge = 0.0
        thrustSway = 0.0
        moment = 0.0
        # TODO - finish vectored thrust design
        return thrustSurge, thrustSway, moment