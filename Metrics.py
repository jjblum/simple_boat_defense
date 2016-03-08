import numpy as np
import abc
import copy
# import scipy.spatial as spatial
# import Boat


class DefenseMetric(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, assets, defenders, attackers):
        self._assets = assets
        self._defenders = defenders
        self._attackers = attackers
        self._value = None  # the metric itself
        self._polarPlotData = None

    @property
    def polarPlotData(self):
        return self._polarPlotData

    @abc.abstractmethod
    def measureCurrentState(self):
        # calculate whatever metric you like
        return


# TODO - look at intrusion ratio (from that combined path following and obstacle avoidance paper) as a possible metric
class IntrustionRatio(DefenseMetric):
    def __init__(self, assets, defenders, attackers):
        super(IntrustionRatio, self).__init__(assets, defenders, attackers)

    def measureCurrentState(self):
        return


class StaticRingMinimumTimeToArrive(DefenseMetric):
    """
        Calculate the minimum time to arrive for any defender in a polar grid around the asset
        i.e. for each location, calculate the time to arrive of all the defenders, keeping the shortest arrival time
        Contours of distance can be displayed for any desired arrival time.
        Other metrics can be derived from this, assuming a straight line of attack from an attacker,
            if you know the range an attacker can be sensed and the attacker's straight-line speed, there will be
            (sensor range)/(attacker speed) seconds to respond before impact with the asset. "Critical Time"
            "Critical Defense Contour" - The minimum-time-to-arrive contour using critical time
            "Complete Coverage Radius" - the radius of inscribed circle of the Critical Defense Contour

        Time to Arrive is APPROXIMATED by the following:
            Sum the following discrete "steps", assuming the defender is given the command to arrive at some goal
            1) time to accelerate to top SPEED in a straight line
            2) time to travel around a circle such that boat faces the goal location,
               assuming top speed and a fixed radius
            3) time to travel at top speed to reach the goal
    """
    def __init__(self, assets, defenders, attackers, resolution_r=5.0, resolution_th=15.*np.pi/180.0, max_r=30.0):
        super(StaticRingMinimumTimeToArrive, self).__init__(assets, defenders, attackers)
        radii = np.arange(0.0, max_r + 0.001, resolution_r)
        self._ths = np.arange(-np.pi, np.pi + 0.001, resolution_th)
        R, TH = np.meshgrid(radii, self._ths)
        # note: self._ths == TH[:, 0]
        self._polarGrid = np.column_stack((np.ravel(R), np.ravel(TH)))  # used for plotting
        self._baseCartesianGrid = np.column_stack((np.multiply(self._polarGrid[:, 0], np.cos(self._polarGrid[:, 1])),
                                                   np.multiply(self._polarGrid[:, 0], np.sin(self._polarGrid[:, 1]))))
        self._cartesianGrid = None
        self.cartesianGridUpdate(assets[0])
        self._polarPlotData = np.column_stack((self._ths, np.ones(self._ths.shape)))

    def measureCurrentState(self):
        self.cartesianGridUpdate(self._assets[0])
        ND = len(self._defenders)

        # time to accelerate to 90% top SPEED in a straight line
        u0 = np.zeros((ND,))
        u1 = np.zeros((ND,))
        numCoef = np.zeros((ND,))
        denCoef = np.zeros((ND,))
        for i in range(ND):
            None
            b = self._defenders[i]
            F = b.design.maxForwardThrust
            a = b.design.dragAreas[0]
            c = b.design.dragCoeffs[0]
            m = b.design.mass
            rho = 1000.0  # density of water [kg/m^3]
            numCoef[i] = np.sqrt((a*c*rho)/(2.*F))
            denCoef[i] = np.sqrt((F*a*c*rho)/2.)
            u0[i] = b.state[2]
            u1[i] = 0.90*b.design.maxSpeed
        timeToMaxSpeed = m/denCoef*(np.arctanh(numCoef*u1) - np.arctanh(numCoef*u0))

        # time to travel around a circle to face goal (radius proportional to initial speed)
        #R = b.design.minTurnRadius(u0)

        # time to travel at top speed, leaving circle on tangent and reach the goal
        #D =

        self.polarPlotData[:, 1] += np.random.normal(0.0, 0.005, size=self._ths.shape)

    def minTimeToArriveContour(self, t):
        # return a numpy array, (th, r) where r is the radius attributed to arrival at time t for the given th
        return

    def criticalTime(self):
        return

    def criticalDefenseContour(self):
        return

    def cartesianGridUpdate(self, asset):
        x = asset.state[0]
        y = asset.state[1]
        self._cartesianGrid = np.column_stack((self._baseCartesianGrid[:, 0] + x, self._baseCartesianGrid[:, 1] + y))