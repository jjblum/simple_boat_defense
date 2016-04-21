import scipy.interpolate as interp
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LightSource, Normalize
import os.path
import pylab
import Boat
import Strategies
import Designs

WITH_PLOTTING = False
TOTAL_TIME = 100.0  # [s]
TH_RESOLUTION = 5.0  # [deg]
#U0_RESOLUTION = 0.5  # m/s
PLOT_SIZE = 30.0
dt = 0.02


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


def plotSystem(boat, plot_time, r, line_th, u0):
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
    location_text = ax_main.text((left+right)/2., top, "r = {:.2f}, th = {:.2f}, u0 = {:.2f}".format(r, np.rad2deg(line_th), u0),
                             horizontalalignment='center', verticalalignment='bottom',
                             transform=ax_main.transAxes, size=20)
    ax_main.draw_artist(time_text)
    ax_main.draw_artist(surge_text)
    ax_main.draw_artist(location_text)
    fig.canvas.blit(ax_main.bbox)



def main():

    if os.path.isfile('RawTimeToArrive.npz'):
        npzfile = np.load('RawTimeToArrive.npz')
        th = npzfile['th']
        r = npzfile['r']
        u0 = npzfile['u0']
        T_th_r_u0 = npzfile['T_th_r_u0']

        # ignore data below 3 meters radius
        r_keep = np.where(r > 3.0)
        r = r[r_keep]
        T_th_r_u0 = np.squeeze(T_th_r_u0[:, r_keep, :])  # have to remove a superfluous dimension that is created

        NTH = T_th_r_u0.shape[0]
        NR = T_th_r_u0.shape[1]
        NU0 = T_th_r_u0.shape[2]

        """
        asdf1 = np.zeros((NTH, NR, NU0))
        n = 0
        for i in range(NTH):
            for j in range(NR):
                for k in range(NU0):
                    asdf1[i, j, k] = n
                    n += 1
        """
        # loops like this fill the highest dimensions first
        # np.reshape(asdf1, (NTH*NR*NU0, 1)) is now a vector in order, [0, 1, 2...] etc.

        # meshgrid->columnstack ravels() does NOT act like the for loops!
        # The third, then 1st, then 2nd dimension is the ordering. Weird.
        # So you need R, TH, U0 ordering for the two to match
        R, TH, U0 = np.meshgrid(r, th, u0)  # three dimensional meshgrid
        A = np.column_stack((np.ravel(TH), np.ravel(R), np.ravel(U0)))
        b = np.reshape(T_th_r_u0, (A.shape[0],))
        coeffs, residuals, rank, singular_values = np.linalg.lstsq(A, b)

        print "Beyond 3 meters, the time to arrive is approximately = {}*th + {}*r + {}*u0".format(
                coeffs[0], coeffs[1], coeffs[2])

        #if np.min(r) == 0.0:
        #    T_th_r_u0[:, 0, 0] = 0  # zero radius, u0 = 0 should be zero seconds (have to do this one manually)
        #fig = plt.figure()
        #ax = fig.gca(projection='3d')
        R, TH = np.meshgrid(r, th)  # three dimensional meshgrid
        #ax.plot_surface(R, np.rad2deg(TH), T_th_r_u0[:, :, 0], color='r', alpha=0.5)
        #ax.plot_surface(R, np.rad2deg(TH), T_th_r_u0[:, :, -1], color='g', alpha=0.5)
        #ax.plot_surface(R, np.rad2deg(TH), coeffs[0]*TH + coeffs[1]*R + coeffs[2]*2.5, color='b', alpha=0.5)
        #ax.plot_surface(R, TH, coeffs[0]*TH + coeffs[1]*R - T_th_r_u0[:, :, 0], color='r')
        #ax.scatter(R, np.rad2deg(TH), coeffs[0]*TH + coeffs[1]*R + coeffs[2]*2.5 - T_th_r_u0[:, :, -1], color='r', alpha=0.5)

        plt.rcParams.update({'font.size': 18})
        error = coeffs[0]*TH + coeffs[1]*R + coeffs[2]*2.5 - T_th_r_u0[:, :, -1]
        fig, ax = plt.subplots()
        ax.imshow(error, extent=[0, 30, 0, 180],  aspect="auto")
        # Use a proxy artist for the colorbar...
        im = ax.imshow(error, extent=[0, 30, 0, 180],  aspect="auto")
        im.remove()
        fig.colorbar(im, label="(linear model - offline data) error (s)")
        plt.xlabel("Distance relative to agent (m)")
        plt.ylabel("Heading relative to agent (deg)")
        #plt.title("Least-Squares Linear Model of Time-To-Arrive")
        plt.show()

        # notice how the results are almost planar for both u0 = 0 and u0 = max EXCEPT below 3 m radius
        # So, for simplicity, a least squares plane will be used such that
        # TTA = a*th + b*r + c*u0
        # *** this shows that if you use radius >= 3 meters, you can use a linear least squares fit

        # plots of residuals show that the vast majority of error is within 0.5 seconds
        # This is well within the general noise of this simple simluation, so we should be fine with this simple solution.

    else:
        boat = Boat.Boat()
        boat.design = Designs.LowThrustTankDriveDesign()
        # u0 = np.arange(0.0, boat.design.maxSpeed+0.001, U0_RESOLUTION)
        # u0 = np.linspace(0.001, boat.design.maxSpeed, 2.)
        u0 = np.linspace(0.0, 2.5, 6)
        th = np.arange(0.0, np.pi+0.001, np.deg2rad(TH_RESOLUTION))
        r = np.linspace(0.0, 30.0, 30)
        NTH = th.shape[0]
        NR = r.shape[0]
        NU0 = len(u0)
        T_th_r_u0 = np.zeros((NTH, NR, NU0))
        for i in range(NTH):
            for j in range(NR):
                for k in range(NU0):
                    th_ = th[i]
                    r_ = r[j]
                    u0_ = u0[k]
                    boat.time = 0
                    boat.state = np.zeros((6,))
                    boat.plotData = None
                    boat.state[2] = u0_
                    #boat.strategy = Strategies.Line_LOS(boat, -50.*np.cos(th_), -50.*np.sin(th_),
                    #                                    50.*np.cos(th_), 50.*np.sin(th_),
                    #                                    surgeVelocity=boat.design.maxSpeed,
                    #                                    headingErrorSurgeCutoff=np.interp(u0_, [0., boat.design.maxSpeed], [np.deg2rad(15.), np.deg2rad(30.)]))
                    destination = [r_*np.cos(th_), r_*np.sin(th_)]
                    #print "Goal = {:.2f}, {:.2f}".format(destination[0], destination[1])
                    boat.strategy = Strategies.TimedStrategySequence(boat, [
                        (Strategies.DoNothing, (boat,)),
                        (Strategies.PointAtLocation, (boat, destination)),
                        (Strategies.DestinationOnly, (boat, destination, 0.1, False))
                    ], [0.01, 100., 100.])
                    t = 0.0
                    step = 1
                    while t < TOTAL_TIME:

                        #print boat.strategy._strategies[-1]

                        times = np.linspace(t, t+dt, 2)
                        boat.time = t
                        boat.control()
                        states = spi.odeint(Boat.ode, boat.state, times, (boat,))
                        boat.state = states[1]
                        # print "{:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}  {:.2f}".format(boat.state[0], boat.state[1], boat.state[2], boat.state[3], boat.state[4], boat.state[5])
                        if t > 0.1:
                            distance = np.sqrt(np.power(destination[0] - boat.state[0], 2.) + np.power(destination[1] - boat.state[1], 2.))
                            #print "Distance to goal = {:.2f}".format(distance)
                            #if boat.state[2] > 0.95*boat.design.maxSpeed and boat.state[5] < np.deg2rad(0.5):
                            #    t += distance/boat.design.maxSpeed
                            #    print "FINISHED R = {:.2f}, TH = {:.2f} deg, u0 = {:.2f} m/s in {:.2f} seconds".format(r_, np.rad2deg(th_), u0_, t)
                            #    T_th_r_u0[i, j, k] = t
                            #    t = 1000.
                            #elif distance < 0.1:
                            if distance < 0.1:
                                print "FINISHED R = {:.2f}, TH = {:.2f} deg, u0 = {:.2f} m/s in {:.2f} seconds".format(r_, np.rad2deg(th_), u0_, t)
                                T_th_r_u0[i, j, k] = t
                                t = 1000.
                        t += dt
                        step += 1
                        if WITH_PLOTTING:
                            plotSystem(boat, t, r_, th_, u0_)

        np.savez('RawTimeToArrive.npz', th=th, r=r, u0=u0, T_th_r_u0=T_th_r_u0)  # save the 3 dimensional array
        main()


"""
TIME_TO_ARRIVE = [5.0, 10.0, 20.0]  # TTA
TTA = TIME_TO_ARRIVE
th = np.arange(0, np.pi+0.001, np.deg2rad(45.0))  # only need half b/c we can just mirror the other side
tank = Designs.TankDriveDesign()
uMax = tank.maxSpeed
# u0 = np.arange(0.0, uMax, 0.5)
u0 = np.array([0.0, 0.5])
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
    NU0 = u0.shape[0]
    D = np.zeros((NTH,))  # resulting distance along line with angle th
    th_grid, u0_grid = np.meshgrid(th, u0)
    th_u0_pairs = np.column_stack((np.ravel(th_grid), np.ravel(u0_grid)))
    timeRemaining = np.concatenate(tuple([tta*np.ones(NTH, NU0) for tta in TTA]),axis=2)
    timeRemaining = np.repeat(tta, NTH, axis=0)
    for tta in TTA:

        timeToTurnToParallel = th/thdot_max
        timeRemaining -= timeToTurnToParallel  # subtract time to turn to 90
        # http://physics.stackexchange.com/questions/72503/how-do-i-calculate-the-distance-a-ship-will-take-to-stop
        L = 2*tank.mass/(tank.dragCoeffs[2]*tank.dragAreas[1]*1000.)  # 2M/(C*rho*A)
        t0 = L/u0  # L/t0 = u0

        drifted_surgeDistance = L*np.log((timeToTurnToParallel + t0)/t0)  # along surge direction
        driftedDistance_alongLine = drifted_surgeDistance*np.cos(th)
        driftedDistance_perpendicularToLine = drifted_surgeDistance*np.sin(th)

    return
"""

if __name__ == "__main__":
    main()
