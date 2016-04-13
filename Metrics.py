import numpy as np
import abc
import matplotlib.pyplot as plt
import copy
import Polygon as poly
import Polygon.Utils as polyUtils
import Polygon.Shapes as polyShapes
import Designs


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


class DefenseMetric(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, assets, defenders, attackers):
        self._assets = assets
        self._defenders = defenders
        self._attackers = attackers
        self._value = None  # the metric itself
        self._polarPlotData = None
        self._T = 0.0  # time threshold
        self._thCoeff = 2.54832785865
        self._rCoeff = 0.401354269952
        self._u0Coeff = 0.0914788305811
        self._ths = None

    @property
    def timeThreshold(self):
        return self._T

    @timeThreshold.setter
    def timeThreshold(self, T_in):
        self._T = T_in

    @property
    def polarPlotData(self):
        return self._polarPlotData

    @abc.abstractmethod
    def measureCurrentState(self):
        # calculate whatever metric you like
        return

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value_in):
        self._value = value_in

    @property
    def ths(self):
        return self._ths


# TODO - look at intrusion ratio (from that combined path following and obstacle avoidance paper) as a possible metric
class IntrustionRatio(DefenseMetric):
    def __init__(self, assets, defenders, attackers):
        super(IntrustionRatio, self).__init__(assets, defenders, attackers)

    def measureCurrentState(self):
        return


class DefenderFrameTimeToArrive(DefenseMetric):
    def __init__(self, assets, defenders, attackers, time_threshold=[1.0, 5.0, 10.0]):
        super(DefenderFrameTimeToArrive, self).__init__(assets, defenders, attackers)
        self._time_threshold = time_threshold
        self._T = time_threshold
        self._polygons = dict()  # the polygons that represent TTA

    def measureCurrentState(self):
        # the RawTimeToArriveOffline.py file has the derivation of this
        ND = len(self._defenders)
        u0 = np.zeros((ND,))
        th = np.linspace(0., np.pi, 181)
        NTH = th.shape[0]
        R = np.zeros((ND, NTH))  # radius that matches the time threshold


        for i in range(ND):
            defender = self._defenders[i]
            defender.TTAPolygon = list()
            defender.TTAData = list()
            x = defender.state[0]
            y = defender.state[1]
            u0 = defender.state[2]
            uid = defender.uniqueID
            heading = defender.state[4]
            transform = np.array([[np.cos(heading), -np.sin(heading), x], [np.sin(heading), np.cos(heading), y], [0., 0., 1.]])
            for T in self._T:

                R[i, :] = 1./self._rCoeff*(T - self._thCoeff*th - self._u0Coeff*u0)
                R[R < 0.] = 0.
                bodyFrameData = np.row_stack((R[i, :]*np.cos(th), R[i, :]*np.sin(th), np.ones((1, NTH))))
                # mirror image the other side and invert y
                bodyFrameDataFlipped = np.fliplr(copy.deepcopy(bodyFrameData[:, 1:-1]))
                bodyFrameDataFlipped[1, :] *= -1.
                bodyFrameData = np.column_stack((bodyFrameData, bodyFrameDataFlipped))

                # transform from body frame into world frame
                worldFrameData = np.dot(transform, bodyFrameData)
                worldFrameData = worldFrameData[:2, :]  # remove extra 1's from homogenous transform
                # need to create a tuple of (x,y) 2-tuples to generate a polygon object. Start by appending to a list.
                worldFrameList = list()
                for j in range(worldFrameData.shape[1]):
                    worldFrameList.append((worldFrameData[0, j], worldFrameData[1, j]))
                worldFrameTuple = tuple(worldFrameList)
                self._polygons[uid] = polyUtils.prunePoints(polyUtils.Polygon(worldFrameTuple))  # remove any redundant points
                if u0 > 1.0 and self._T < 8.9:
                    # defender with forward velocity can't get back to where it started faster than 8.9 seconds
                    # for simplicity, just cut out the entire 3 meter circle where the plane fit for TTA breaks down
                    #R[i, np.where(R[i, :] < 3.0)] = 3.0
                    self._polygons[uid] -= polyShapes.Circle(radius=3.0, center=(x, y))
                defender.TTAPolygon.append(self._polygons[uid])
                TTAData = np.array(polyUtils.pointList(self._polygons[uid]))
                defender.TTAData.append(np.row_stack((TTAData, TTAData[0, :])))

            # Now combine the defender polygons into a single coverage polygon to simplify asset frame stuff
            #one_defender_polygon = self._defenders[0].TTAPolygon
            #for i in range(1, ND):
            #    one_defender_polygon += self._defenders[i].TTAPolygon
            #boundingBox = one_defender_polygon.boundingBox()
            ##outer_contour = np.array(one_defender_polygon.contour(0))
            ##inner_contour = np.array(one_defender_polygon.contour(1))
            #th = np.deg2rad(np.arange(0.0, 360.0+0.001, 10.0))
            ##for th_ in th:
            #    # find farthest point that is in the polygon
            #asdf = 0


class MinimumTTARings(DefenseMetric):
    def __init__(self, assets, defenders, attackers, radii=[5.0, 10.0, 20.0, 40.0], resolution_th=1.*np.pi/180.0):
        super(MinimumTTARings, self).__init__(assets, defenders, attackers)
        self._radii = radii
        self._ths = self._ths = np.arange(0, 2*np.pi+0.001, resolution_th)
        self._NR = len(radii)
        self._NTH = self._ths.shape[0]
        self._ND = len(defenders)
        self._R = None
        self._TH = None
        self._polarGrid = None
        self._baseCartesianGrid = None
        self._cartesianGrid = None
        self.createPolarGrid()
        self.createCartesianGrid()
        self._minTTA_dict = dict()
        self._minTTA = list()
        self._t = 0.

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t_in):
        self._t = t_in

    def radii(self):
        return self._radii

    def minTTA_dict(self):
        return self._minTTA_dict

    def minTTA(self):
        return self._minTTA

    def createPolarGrid(self):
        TH, R = np.meshgrid(self._ths, self._radii)
        self._polarGrid = np.column_stack((np.ravel(R), np.ravel(TH)))  # used to set up cartesian grid

    def createCartesianGrid(self):
        self._baseCartesianGrid = np.column_stack((np.multiply(self._polarGrid[:, 0], np.cos(self._polarGrid[:, 1])),
                                                   np.multiply(self._polarGrid[:, 0], np.sin(self._polarGrid[:, 1]))))

    def cartesianGridUpdate(self, asset):
        self._cartesianGrid = np.column_stack((self._baseCartesianGrid[:, 0] + asset.state[0],
                                               self._baseCartesianGrid[:, 1] + asset.state[1]))
        # cartesianGrid is NR*NTH x 2
        # It holds r constant for all theta, then incrememnts r

    def measureCurrentState(self):
        self.cartesianGridUpdate(self._assets[0])
        ND = self._ND
        NG = self._cartesianGrid.shape[0]
        NTH = self._NTH
        NR = self._NR
        defenders_X = np.zeros((ND, 2))
        defenders_th = np.zeros((ND,))
        defender_u = np.zeros((ND,))
        for i in range(ND):
            defender = self._defenders[i]
            defenders_X[i, 0] = defender.state[0]
            defenders_X[i, 1] = defender.state[1]
            defender_u[i] = defender.state[2]
            defenders_th[i] = defender.state[4]

        gridx, defx = np.meshgrid(self._cartesianGrid[:, 0], defenders_X[:, 0])
        gridy, defy = np.meshgrid(self._cartesianGrid[:, 1], defenders_X[:, 1])
        x_pairs = np.column_stack((np.ravel(defx), np.ravel(gridx)))
        y_pairs = np.column_stack((np.ravel(defy), np.ravel(gridy)))
        # each NG rows are for a single defender
        dx = x_pairs[:, 1] - x_pairs[:, 0]
        dy = y_pairs[:, 1] - y_pairs[:, 0]
        R = np.sqrt(np.power(dx, 2) + np.power(dy, 2))
        global_angle = np.arctan2(dy, dx)
        global_angle[global_angle < 0.] += 2*np.pi
        defenders_th[defenders_th < 0.] += 2*np.pi
        local_angle = global_angle - np.repeat(defenders_th, NG, axis=0)  # the TTA model only uses positive theta!
        local_angle = np.abs(wrapToPi(local_angle))
        TTA = self._thCoeff*local_angle + self._rCoeff*R + self._u0Coeff*np.repeat(defender_u, NG, axis=0)
        # remember, each NG rows are for a single defender -- TTA shape is (ND*NG,) = (ND*NTH*NR,)
        # reshape and find minimum over the defenders for each grid point
        TTA_by_defender = np.reshape(TTA, (ND, NR, NTH))
        minTTA = np.min(TTA_by_defender, axis=0)
        for i in range(minTTA.shape[0]):
            self._minTTA_dict[self._radii[i]] = minTTA[i, :]
        self._minTTA.append(minTTA)


"""
class StaticRingMinimumTimeToArrive(DefenseMetric):

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
        slope = (rightTTA - leftTTA)/self._resolution_r
        contourRadius = -leftTTA/slope + leftRadii
        incompleteContourDict = dict()
        for i in range(contourRadius.shape[0]):
            incompleteContourDict[contourLeftIndices[i, 0]] = contourRadius[i]
        contourData = np.zeros((self._ths.shape[0],))
        for i in range(self._ths.shape[0]):
            if i in incompleteContourDict.keys():
                contourData[i] = incompleteContourDict[i]

        self.polarPlotData[:, 1] = contourData
        self.polarPlotData[-1, 1] = self.polarPlotData[0, 1]  # prevent the erroneous hole at the period edge

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
"""