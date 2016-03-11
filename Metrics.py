import numpy as np
import abc
import matplotlib.pyplot as plt
import copy
# import scipy.spatial as spatial
# import Boat
import Designs


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
            "Critical Time" - if you know the range an attacker can be sensed and the attacker's straight-line speed,
                              there will be (sensor range)/(attacker speed) seconds to respond before impact with the asset.
            "Critical Defense Contour" - The minimum-time-to-arrive contour using critical time
            "Complete Coverage Radius" - the radius of inscribed circle of the Critical Defense Contour

        Time to Arrive is APPROXIMATED by the following:
            Sum the following discrete "steps", assuming the defender is given the command to arrive at some goal
            1) time to accelerate to top SPEED in a straight line
            2) time to travel around a circle such that boat faces the goal location,
               assuming top speed and a fixed radius
               UPDATE: The Lutra design is not long, and so it can rotate with relative ease.
                       In this simulation, rotation in place will typically be faster than traveling around a circle.
                       Therefore, for simplicity, this will be reduced to a simple estimate of rotate-in-place time
            3) time to travel at top speed to reach the goal
    """
    def __init__(self, assets, defenders, attackers, resolution_r=5.0, resolution_th=15.*np.pi/180.0, max_r=30.0, time_threshold=5.0):
        super(StaticRingMinimumTimeToArrive, self).__init__(assets, defenders, attackers)
        self._timeThreshold = time_threshold
        self._max_r = max_r
        self._resolution_r = resolution_r
        self._resolution_th = resolution_th
        self.createPolarGrid()
        self.createCartesianGrid()
        self._cartesianGrid = None
        self.cartesianGridUpdate(assets[0])
        self._polarPlotData = np.column_stack((self._ths, np.ones(self._ths.shape)))

    def createCartesianGrid(self):
        self._baseCartesianGrid = np.column_stack((np.multiply(self._polarGrid[:, 0], np.cos(self._polarGrid[:, 1])),
                                                   np.multiply(self._polarGrid[:, 0], np.sin(self._polarGrid[:, 1]))))

    def createPolarGrid(self):
        self._radii = np.arange(self._resolution_r, self._max_r + 0.001, self._resolution_r)
        self._ths = np.arange(-np.pi, np.pi + 0.001, self._resolution_th)
        R, TH = np.meshgrid(self._radii, self._ths)  # note: self._ths == TH[:, 0]
        self._polarGrid = np.column_stack((np.ravel(R), np.ravel(TH)))  # used to set up cartesian grid

    def measureCurrentState(self):
        self.cartesianGridUpdate(self._assets[0])
        ND = len(self._defenders)
        NG = self._cartesianGrid.shape[0]
        defenders_X = np.zeros((ND, 2))
        defenders_th = np.zeros((ND,))
        for i in range(ND):
            defenders_X[i, 0] = self._defenders[i].state[0]
            defenders_X[i, 1] = self._defenders[i].state[1]
            defenders_th[i] = self._defenders[i].state[4]
        defx, gridx = np.meshgrid(defenders_X[:, 0], self._cartesianGrid[:, 0])
        defy, gridy = np.meshgrid(defenders_X[:, 1], self._cartesianGrid[:, 1])
        x_pairs = np.column_stack((np.ravel(defx), np.ravel(gridx)))
        y_pairs = np.column_stack((np.ravel(defy), np.ravel(gridy)))
        dx = x_pairs[:, 1] - x_pairs[:, 0]
        dy = y_pairs[:, 1] - y_pairs[:, 0]
        goal_angle = np.arctan2(dy, dx)
        dth_unwrapped = goal_angle - np.ravel(np.repeat(np.atleast_2d(defenders_th), gridx.shape[0], axis=0))
        dth = np.abs(np.arctan2(np.sin(dth_unwrapped), np.cos(dth_unwrapped)))
        # http://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles


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

        # turning in place a full 180 degrees will take almost 8 seconds
        # traveling at 2 m/s in a circle requires a 7 m radius circle
        # half of that circle perimeter would be 3.14*radius meters/2 m/s = 11 seconds!

        # time to rotate in place
        timeToRotateInPlace = dth/self._defenders[0].design.maxHeadingRate  # just assume all are tank drive for now
        timeToRotateInPlace = np.reshape(timeToRotateInPlace, (NG, ND))

        # time to travel at top speed to reach the goal
        # = distance/top forward speed
        D = np.sqrt(np.power(dx, 2.) + np.power(dy, 2.))
        timeToTravel = D/self._defenders[0].design.maxSpeed  # just assume all are tank drive for now
        timeToTravel = np.reshape(timeToTravel, (NG, ND))

        totalTime = np.repeat(np.atleast_2d(timeToMaxSpeed), NG, axis=0) + timeToRotateInPlace + timeToTravel
        minimumTimeToArrive = np.min(totalTime, axis=1)

        # reshape according to geometry, [th, r] shape
        minimumTimeToArrive = np.reshape(minimumTimeToArrive, (self._ths.shape[0], self._radii.shape[0]))

        # find the maximum radius that meets the threshold
        minimumTimeToArriveShifted = minimumTimeToArrive - self._timeThreshold
        indicesBelowTimeThreshold = np.where(minimumTimeToArriveShifted < 0)  # assumes that you checking a maximum radius large enough that some TTA's are bigger than the threshold
        if len(np.ravel(np.where(minimumTimeToArriveShifted > 0))) == 0:
            # you have to increase the radius!
            self._max_r *= 2.
            self.createPolarGrid()
            self.createCartesianGrid()
            return

        contourLeftIndices = np.where(np.array(indicesBelowTimeThreshold[0]) != np.roll(np.array(indicesBelowTimeThreshold[0]), 1))[0] - 1
        contourLeftIndices = contourLeftIndices[1:]  # remove the erroneous negative in the first element
        contourLeftIndices = np.array((indicesBelowTimeThreshold[0][contourLeftIndices], indicesBelowTimeThreshold[1][contourLeftIndices]))
        contourLeftIndices = contourLeftIndices.T
        contourRightIndices = copy.deepcopy(contourLeftIndices)
        contourRightIndices[:, 1] += 1
        if np.any(contourRightIndices[:, 1] >= len(self._radii)):
            # you have to increase the radius!
            self._max_r *= 2.
            self.createPolarGrid()
            self.createCartesianGrid()
            return
        # left indices and right indices form the endpoints of a line. Find where this line crosses zero
        leftRadii = self._radii[contourLeftIndices[:, 1]]
        # rightRadii = self._radii[contourRightIndices[:, 1]]  # don't need this b/c we use a fixed radius resolution!
        leftTTA = minimumTimeToArriveShifted[contourLeftIndices[:, 0], contourLeftIndices[:, 1]]
        rightTTA = minimumTimeToArriveShifted[contourRightIndices[:, 0], contourRightIndices[:, 1]]
        # TODO - scenario where the current largest radius has a negative value. It needs something beyond that is positive. Increase radius again.
        slope = (rightTTA - leftTTA)/self._resolution_r
        contourRadius = -leftTTA/slope + leftRadii
        incompleteContourDict = dict()
        for i in range(contourRadius.shape[0]):
            incompleteContourDict[contourLeftIndices[i, 0]] = contourRadius[i]
        #incompleteContourData = np.column_stack((contourLeftIndices[:, 0], contourRadius))
        contourData = np.zeros((self._ths.shape[0],))
        for i in range(self._ths.shape[0]):
            if i in incompleteContourDict.keys():
                contourData[i] = incompleteContourDict[i]


        # self.polarPlotData[:, 1] += np.random.normal(0.0, 0.005, size=self._ths.shape)
        self.polarPlotData[:, 1] = contourData
        self.polarPlotData[-1, 1] = self.polarPlotData[0, 1]  # prevent the erroneous hole at the period edge
        # TODO - what happens when the time threshold is too big for any data to work?

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