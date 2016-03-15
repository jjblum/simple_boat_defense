import scipy.interpolate as interp
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import pylab
import Boat
import Strategies

WITH_PLOTTING = True
TOTAL_TIME = 20.0  # [s]
TH_RESOLUTION = 10.0  # [deg]
#U0_RESOLUTION = 0.5  # m/s
PLOT_SIZE = 25.0
dt = 0.01

if WITH_PLOTTING:
    figx = 10.0
    figy = 10.0
    fig = plt.figure(figsize=(figx, figy))
    ax_main = fig.add_axes([0.01, 0.01, 0.95*figy/figx, 0.95])  # main ax must come last for arrows to appear on it!!!
    ax_main.elev = 10
    ax_main.grid(b=False)  # no grid b/c it won't update correctly
    ax_main.set_xticks([])  # turn off axis labels b/c they wont update correctly
    ax_main.set_yticks([])  # turn off axis labels b/c they wont update correctly
    boat_arrows = None
    plt.ioff()
    fig.show()
    background_main = fig.canvas.copy_from_bbox(ax_main.bbox)  # must be below fig.show()!
    fig.canvas.draw()


def plotSystem(boat, plot_time, line_th):
    fig.canvas.restore_region(background_main)
    # gather the X and Y location
    boat_x = boat.state[0]
    boat_y = boat.state[1]
    boat_th = boat.state[4]
    axes = [-PLOT_SIZE, PLOT_SIZE, -PLOT_SIZE, PLOT_SIZE]
    ax_main.plot([-PLOT_SIZE, PLOT_SIZE], [-PLOT_SIZE, PLOT_SIZE],
                 'x', markersize=1, markerfacecolor='white', markeredgecolor='white')
    ax_main.axis(axes)  # requires those invisible 4 corner white markers
    boat_arrow = pylab.arrow(boat_x, boat_y,0.05*np.cos(boat_th),0.05*np.sin(boat_th),
                             fc="g", ec="g", head_width=0.5, head_length=1.0)
    ax_main.draw_artist(boat_arrow)
    style = 'b-'
    if boat.plotData is not None:
        ax_main.draw_artist(ax_main.plot(boat.plotData[:, 0], boat.plotData[:, 1], style, linewidth=0.25)[0])
    ax_main.draw_artist(ax_main.plot([0, 50.*np.cos(line_th)], [0, 50.*np.sin(line_th)], 'k--', linewidth=0.25)[0])
    # rectangle coords
    left, width = 0.01, 0.95
    bottom, height = 0.0, 0.96
    right = left + width
    top = bottom + height
    time_text = ax_main.text(right, top, "time = {:.2f} s".format(plot_time),
                             horizontalalignment='right', verticalalignment='bottom',
                             transform=ax_main.transAxes, size=20)
    surge_text = ax_main.text(left, top, "u = {:.2f} m/s".format(boat.state[2]),
                             horizontalalignment='left', verticalalignment='bottom',
                             transform=ax_main.transAxes, size=20)
    ax_main.draw_artist(time_text)
    ax_main.draw_artist(surge_text)
    fig.canvas.blit(ax_main.bbox)



def main():
    boat = Boat.Boat()
    # u0 = np.arange(0.0, boat.design.maxSpeed+0.001, U0_RESOLUTION)
    # u0 = np.linspace(0.001, boat.design.maxSpeed, 2.)
    u0 = [2.5]
    #th = np.arange(np.pi/2., np.pi+0.001, np.deg2rad(TH_RESOLUTION))
    th = [np.deg2rad(135.)]
    for th_ in th:
        for u0_ in u0:
            boat.state = np.zeros((6,))
            boat.plotData = None
            #boat.state[0] = -10.
            #boat.state[1] = 10.
            boat.state[2] = u0_
            boat.strategy = Strategies.Line_LOS(boat, -50.*np.cos(th_), -50.*np.sin(th_),
                                                50.*np.cos(th_), 50.*np.sin(th_),
                                                surgeVelocity=boat.design.maxSpeed,
                                                headingErrorSurgeCutoff=np.interp(u0_, [0., boat.design.maxSpeed], [np.deg2rad(15.), np.deg2rad(120.)]),
                                                lookAhead=1.0)
            """
            boat.strategy = Strategies.FollowWaypoints(boat,
                                                       np.column_stack((np.cos(th_)*np.array([-10., 1., 50.]),
                                                                        np.sin(th_)*np.array([-10., 1., 50.]))),
                                                       np.array([th_, th_, th_]),
                                                       lookAhead=0.04,
                                                       headingErrorSurgeCutoff=np.interp(u0_, [0., boat.design.maxSpeed], [np.deg2rad(15.), np.deg2rad(45.)]),
                                                       surgeVelocity=boat.design.maxSpeed,
                                                       positionThreshold=1.0,
                                                       closed_circuit=False)
            """
            t = 0.0
            step = 1
            while t < TOTAL_TIME:
                times = np.linspace(t, t+dt, 2)
                boat.control()
                states = spi.odeint(Boat.ode, boat.state, times, (boat,))
                boat.state = states[1]
                # print "{:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}".format(boat.state[0], boat.state[1], boat.state[2], boat.state[3], boat.state[4], boat.state[5])
                t += dt
                step += 1
                plotSystem(boat, t, th_)


"""
TIME_TO_ARRIVE = [5.0, 10.0, 20.0]  # TTA
TTA = TIME_TO_ARRIVE
th = np.arange(0, np.pi+0.001, np.deg2rad(1.0))  # only need half b/c we can just mirror the other side
tank = Designs.TankDriveDesign()
uMax = tank.maxSpeed
u0 = np.arange(0.0, uMax, 0.5)
thdot_max = tank.maxHeadingRate  # just assume everything is a tank drive Lutra boat for now

turning_surge_max = 1.5 # because otherwise the turning radius is unpractically large
R = np.interp(turning_surge_max, tank.speedVSMinRadius[:, 0], tank.speedVSMinRadius[:, 1])

def incidentAngleFormula(T, R, ut, uMax, theta, phi):
    apothem = R*np.cos(theta)
    sagitta = R - apothem
    return 1./T*(R/ut*(2*phi + theta)*np.sin(phi) + sagitta/uMax) - np.sin(phi)


def main():
    # create a 3 dimensional array [th, u0, TTA]
    NTH = th.shape[0]
    D = np.zeros((NTH,))  # resulting distance along line with angle th
    for tta in TTA:
        timeRemaining = np.repeat(tta, NTH, axis=0)
        timeToTurnTo90 = np.maximum(th - np.pi/2., np.zeros((NTH,)))/thdot_max
        timeRemaining -= timeToTurnTo90  # subtract time to turn to 90
        th_temp = np.minimum(th, np.pi/2.)  # now th <= 90 for all
        # go around circle until you are parallel with the line
        timeRemaining -= R*th_temp/turning_surge_max
        timeRemaining = np.maximum(timeRemaining, 0.0)  # time remaining can't be less than 0
        # need to find angle of incidence that reaches the line when time remaining = 0
        phi = np.linspace(0., np.pi/2., 1000)
        plt.plot(np.rad2deg(phi), incidentAngleFormula(10.0, R, turning_surge_max, uMax, np.deg2rad(45.), phi))
        plt.plot([0, 90], [0, 0], 'k-')
        plt.show()
        # scipy.optimize.newton



    return
"""

if __name__ == "__main__":
    main()
