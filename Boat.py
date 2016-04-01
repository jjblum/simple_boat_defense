import numpy as np
import math
import copy
import Strategies
import Designs

__author__ = 'jjb'


def wrapToPi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def ode(state, t, boat):
    # derivative of state at input state and time
    # this is in Boat, not Design, because only the forces and moment are relevant
    rho = 1000.0  # density of water [kg/m^3]
    u = state[2]
    w = state[3]
    th = state[4]
    thdot = state[5]
    au = boat.design.dragAreas[0]
    aw = boat.design.dragAreas[1]
    ath = boat.design.dragAreas[2]
    cu = boat.design.dragCoeffs[0]
    cw = boat.design.dragCoeffs[1]
    cth = boat.design.dragCoeffs[2]
    qdot = np.zeros((6,))
    qdot[0] = u*math.cos(th) - w*math.sin(th)
    qdot[1] = u*math.sin(th) + w*math.cos(th)
    qdot[2] = 1.0/boat.design.mass*(boat.thrustSurge - 0.5*rho*au*cu*math.fabs(u)*u)
    qdot[3] = 1.0/boat.design.mass*(boat.thrustSway - 0.5*rho*aw*cw*math.fabs(w)*w)
    qdot[4] = thdot
    qdot[5] = 1.0/boat.design.momentOfInertia*(boat.moment - 0.5*rho*ath*cth*math.fabs(thdot)*thdot)

    # linear friction, only dominates when boat is moving slowly
    if u < 0.25:
        qdot[2] -= 1.0/boat.design.mass*5.0*u - np.sign(u)*0.001
    if w < 0.25:
        qdot[3] -= 1.0/boat.design.mass*5.0*w - np.sign(w)*0.001
    if thdot < math.pi/20.0:  # ten degrees per second
        qdot[5] -= 1.0/boat.design.momentOfInertia*5.0*thdot - np.sign(thdot)*0.001

    return qdot


class Boat(object):
    idCount = 0  # count of boat objects, start with zero index

    def __init__(self, t=0.0):
        self._t = 0.0  # current time [s]
        self._state = np.zeros((6,))
        self._idealState = np.zeros((6,))  # a "rabbit" boat to chase
        # state: [x y u w th thdot]
        self._uniqueID = Boat.idCount
        self._type = "defender"  # can be defender, asset, or attacker
        self._thrustSurge = 0.0  # surge thrust [N]
        self._thrustSway = 0.0  # sway thrust (zero for tank drive) [N]
        self._moment = 0.0  # [Nm]
        self._thrustFraction = 0.0
        self._momentFraction = 0.0
        self._timeToDefend = []  # [a vector of numbers, roughly representing distances radially away from an asset]
        self._attackers = []  # list of attacker objects
        self._assets = []  # list of asset objects (usually just 1)
        self._defenders = []  # list of defender objects
        self._boatList = []
        self._strategy = Strategies.DoNothing(self)
        self._design = Designs.TankDriveDesign()
        self._plotData = None  # [x, y] data used to display current actions
        self._TTAData = list()  # list of time-to-arrive arrays
        self._TTAPolygon = list()  # list of time-to-arrive polygon objects
        self._busy = False  # a flag that prevents another strategy from being assigned
        self._hasBeenTargeted = False  # a flag for attackers only. If a defender has been assigned to intercept, this is true.
        self._target = None  # the boat's current target agent
        self._targetedBy = None  # if the boat has been targeted, this is the intercepting agent
        self._targetedByCount = 0  # number of boats that are targeting this boat
        self._pointOfInterception = None  # the location where an interception is predicted to happen
        self._evading = False
        self._numberOfInterceptionAttempts = 0  # the number of times a defender has attempted to intercept something
        self._originalState = None  # used to allow defenders to go back to where they were first arrayed

        Boat.idCount += 1

    @property
    def time(self):
        return self._t

    @time.setter
    def time(self, t):
        self._t = t
        self.strategy.time = t

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state_in):
        self._state = state_in

    @property
    def uniqueID(self):
        return self._uniqueID

    @uniqueID.setter
    def uniqueID(self, uniqueID_in):
        self._uniqueID = uniqueID_in

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self,type_in):
        self._type = type_in

    @property
    def thrustSurge(self):
        return self._thrustSurge

    @thrustSurge.setter
    def thrustSurge(self, thrustSurge_in):
        self._thrustSurge = thrustSurge_in

    @property
    def thrustSway(self):
        return self._thrustSway

    @thrustSway.setter
    def thrustSway(self, thrustSway_in):
        self._thrustSway = thrustSway_in

    @property
    def moment(self):
        return self._moment

    @moment.setter
    def moment(self, moment_in):
        self._moment = moment_in

    @property
    def timeToDefend(self):
        return self._timeToDefend

    @timeToDefend.setter
    def timeToDefend(self, timeToDefend_in):
        self._timeToDefend = timeToDefend_in

    @property
    def attackers(self):
        return self._attackers

    @attackers.setter
    def attackers(self, attackers_in):
        self._attackers = attackers_in
        self._strategy.attackers = attackers_in

    @property
    def assets(self):
        return self._assets

    @assets.setter
    def assets(self, assets_in):
        self._assets = assets_in
        self._strategy.assets = assets_in

    @property
    def defenders(self):
        return self._defenders

    @defenders.setter
    def defenders(self, defenders_in):
        self._defenders = defenders_in
        self._strategy.defenders = defenders_in

    @property
    def boatList(self):
        return self._boatList

    @boatList.setter
    def boatList(self, boatList_in):
        self._boatList = boatList_in

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy_in):
        self._strategy = strategy_in

    @property
    def design(self):
        return self._design

    @design.setter
    def design(self, design_in):
        self._design = design_in

    @property
    def plotData(self):
        return self._plotData

    @plotData.setter
    def plotData(self, plotData_in):
        self._plotData = plotData_in

    @property
    def TTAData(self):
        return self._TTAData

    @TTAData.setter
    def TTAData(self, TTAData_in):
        self._TTAData = TTAData_in

    @property
    def TTAPolygon(self):
        return self._TTAPolygon

    @TTAPolygon.setter
    def TTAPolygon(self, TTAPolygon_in):
        self._TTAPolygon = TTAPolygon_in

    @property
    def busy(self):
        return self._busy

    @busy.setter
    def busy(self, busy_in):
        self._busy = busy_in

    @property
    def hasBeenTargeted(self):
        return self._hasBeenTargeted

    @hasBeenTargeted.setter
    def hasBeenTargeted(self, hasBeenTargeted_in):
        self._hasBeenTargeted = hasBeenTargeted_in

    @property
    def evading(self):
        return self._evading

    @evading.setter
    def evading(self, evading_in):
        self._evading = evading_in

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target_in):
        self._target = target_in

    @property
    def targetedBy(self):
        return self._targetedBy

    @targetedBy.setter
    def targetedBy(self, targetedBy_in):
        self._targetedBy = targetedBy_in

    @property
    def pointOfInterception(self):
        return self._pointOfInterception

    @pointOfInterception.setter
    def pointOfInterception(self, pointOfInterception_in):
        self._pointOfInterception = pointOfInterception_in

    @property
    def numberOfInterceptionAttempts(self):
        return self._numberOfInterceptionAttempts

    @numberOfInterceptionAttempts.setter
    def numberOfInterceptionAttempts(self, num):
        self._numberOfInterceptionAttempts = num

    @property
    def targetedByCount(self):
        return self._targetedByCount

    @targetedByCount.setter
    def targetedByCount(self, tbc):
        self._targetedByCount = tbc

    @property
    def originalState(self):
        return self._originalState

    @originalState.setter
    def originalState(self, state):
        self._originalState = state

    def __str__(self):
        return "Boat {ID}: {T} at X = {X}, Y = {Y}, TH = {TH}".format(ID=self.uniqueID,
                                                                      X=self.state[0][0],
                                                                      Y=self.state[1][0],
                                                                      T=self.type,
                                                                      TH=self.state[4][0])

    def distanceToBoat(self, otherBoat):
        point = [otherBoat.state[0], otherBoat.state[1]]
        return self.distanceToPoint(point)

    def distanceToPoint(self, point):
        return np.sqrt(np.power(self._state[0] - point[0], 2) + np.power(self._state[1] - point[1], 2))

    def globalAngleToBoat(self, otherBoat):
        point = [otherBoat.state[0], otherBoat.state[1]]
        return self.globalAngleToPoint(point)

    def localAngleToBoat(self, otherBoat):
        point = [otherBoat.state[0], otherBoat.state[1]]
        return self.localAngeToPoint(point)

    def globalAngleToPoint(self, point):
        dx = point[0] - self._state[0]
        dy = point[1] - self._state[1]
        return np.arctan2(dy, dx)

    def localAngeToPoint(self, point):
        ga = self.globalAngleToPoint(self, point)
        if ga < 0:
            ga += 2*np.pi
        a = copy.deepcopy(self._state[4])
        if a < 0:
            a += 2*np.pi
        return wrapToPi(ga - a)

    def control(self):
        self.strategy.updateFinished()
        self.strategy.idealState()

        self._thrustFraction, self._momentFraction = self.strategy.actuationEffortFractions()
        self.thrustSurge, self.thrustSway, self.moment = \
            self.design.thrustAndMomentFromFractions(self._thrustFraction, self._momentFraction)
