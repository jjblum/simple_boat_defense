import numpy as numpy
import math
import Strategies
import Designs

__author__ = 'jjb'


def ode(state,t,boat):
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
    qdot = numpy.zeros((6,))
    qdot[0] = u*math.cos(th) - w*math.sin(th)
    qdot[1] = u*math.sin(th) + w*math.cos(th)
    qdot[2] = 1.0/boat.design.mass*(boat.thrustSurge - 0.5*rho*au*cu*math.fabs(u)*u)
    qdot[3] = 1.0/boat.design.mass*(boat.thrustSway - 0.5*rho*aw*cw*math.fabs(w)*w)
    qdot[4] = thdot
    qdot[5] = 1.0/boat.design.momentOfInertia*(boat.moment - 0.5*rho*ath*cth*math.fabs(thdot)*thdot)

    # linear friction, only dominates when boat is moving slowly
    if u < 0.25:
        qdot[2] -= 1.0/boat.design.mass*5.0*u
    if w < 0.25:
        qdot[3] -= 1.0/boat.design.mass*5.0*w
    if thdot < math.pi/20.0:  # ten degrees per second
        qdot[5] -= 1.0/boat.design.momentOfInertia*5.0*thdot

    return qdot


class Boat(object):
    idCount = 0  # count of boat objects, start with zero index

    def __init__(self, t=0.0):
        self._t = 0.0  # current time [s]
        self._state = numpy.zeros((6,))
        self._idealState = numpy.zeros((6,))  # a "rabbit" boat to chase
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

    def fullPrint(self):
        print "Boat {ID}: {T} at X = {X}, Y = {Y}, TH = {TH}".format(ID=self.uniqueID,
                                                                     X=self.state[0][0],
                                                                     Y=self.state[1][0],
                                                                     T=self.type,
                                                                     TH=self.state[4][0])

    def control(self):
        self.strategy.idealState()
        self._thrustFraction, self._momentFraction = self.strategy.controller.actuationEffortFractions()
        self.thrustSurge, self.thrustSway, self.moment = \
            self.design.thrustAndMomentFromFractions(self._thrustFraction, self._momentFraction)
