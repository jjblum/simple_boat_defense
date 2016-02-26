import numpy
import matplotlib.pyplot as plt
import scipy.integrate as spi
import pylab
import math
import time

import Boat
import Strategies

GLOBAL_DT = 0.05  # [s]
TOTAL_TIME = 120  # [s]
BOAT_COUNT = 12
MAX_DEFENDERS_PER_RING = 8.0  # must be a float
RADII_OF_RINGS = numpy.arange(6.0, 60.0, 4.0)

figx = 20.0
figy = 10.0
fig = plt.figure(figsize=(figx, figy))
# ax = fig.add_subplot(111)
ax = fig.add_axes([0.01, 0.01, 0.95*figy/figx, 0.95])
ax.elev = 10
ax.grid(b=False)  # no grid b/c it won't update correctly
ax.set_xticks([])  # turn off axis labels b/c they wont update correctly
ax.set_yticks([])  # turn off axis labels b/c they wont update correctly
defenders_arrows = None
attackers_arrows = None
assets_arrows = None
plt.ioff()
fig.show()
background = fig.canvas.copy_from_bbox(ax.bbox)  # must be below fig.show()!
fig.canvas.draw()


def plotSystem(assets, defenders, attackers, title_string, plot_time):
    # gather the X and Y locations of all boats
    defenders_x = [boat.state[0] for boat in defenders]
    attackers_x = [boat.state[0] for boat in attackers]
    assets_x = [boat.state[0] for boat in assets]
    defenders_y = [boat.state[1] for boat in defenders]
    attackers_y = [boat.state[1] for boat in attackers]
    assets_y = [boat.state[1] for boat in assets]
    defenders_th = [boat.state[4] for boat in defenders]
    attackers_th = [boat.state[4] for boat in attackers]
    assets_th = [boat.state[4] for boat in assets]

    # find the bounds
    mean_x = numpy.mean(assets_x)
    mean_y = numpy.mean(assets_y)
    relative_x = numpy.asarray(defenders_x + attackers_x + assets_x) - mean_x
    relative_y = numpy.asarray(defenders_y + attackers_y + assets_y) - mean_y
    x_max = 1.05*max(abs(relative_x))
    y_max = 1.05*max(abs(relative_y))
    square_max = max(max([x_max, y_max]), 5.0)

    axes = [mean_x-square_max, mean_x+square_max, mean_y-square_max, mean_y+square_max]
    plt.plot([mean_x-square_max, mean_x+square_max], [mean_y-square_max, mean_y+square_max],
             'x', markersize=1, markerfacecolor='white', markeredgecolor='white')
    plt.axis(axes)  # requires those invisible 4 corner white markers

    fig.canvas.restore_region(background)

    defender_arrows = [pylab.arrow(defenders_x[j], defenders_y[j],
                                   0.05*math.cos(defenders_th[j]),
                                   0.05*math.sin(defenders_th[j]),
                                   fc="g", ec="g", head_width=0.5, head_length=1.0) for j in range(len(defenders_x))]
    attacker_arrows = [pylab.arrow(attackers_x[j], attackers_y[j],
                                   0.05*math.cos(attackers_th[j]),
                                   0.05*math.sin(attackers_th[j]),
                                   fc="r", ec="r", head_width=0.5, head_length=1.0) for j in range(len(attackers_x))]
    asset_arrows = [pylab.arrow(assets_x[j], assets_y[j],
                    0.05*math.cos(assets_th[j]),
                    0.05*math.sin(assets_th[j]),
                    fc="b", ec="b", head_width=0.5, head_length=1.0) for j in range(len(assets_x))]

    for boat in defenders:
        if boat.plotData is not None:
            ax.draw_artist(ax.plot(boat.plotData[:, 0], boat.plotData[:, 1], 'k-', linewidth=3.0)[0])

    # plt.title(title_string + " time = {tt} s".format(tt=plot_time))
    # rectangle coords
    left, width = 0.01, 0.95
    bottom, height = 0.0, 0.96
    right = left + width
    top = bottom + height
    time_text = ax.text(right, top, "time = {:.2f} s".format(plot_time),
                        horizontalalignment='right', verticalalignment='bottom',
                        transform=ax.transAxes, size=20)
    asset_position_text = ax.text(0.5*(left + right), bottom,
                                  "asset (x,y) = {:.2f},{:.2f}".format(mean_x, mean_y),
                                  horizontalalignment='center', verticalalignment='bottom',
                                  transform=ax.transAxes, size=20)
    title_text = ax.text(left, top, "{s}".format(s=title_string),
                         horizontalalignment='left', verticalalignment='bottom',
                         transform=ax.transAxes, size=20)
    real_time_passed = time.time() - real_time_zero
    time_ratio = assets[0].time/real_time_passed
    time_ratio_text = ax.text(right, top - 0.03, "speed = {:.2f}x".format(time_ratio),
                              horizontalalignment='right', verticalalignment='bottom',
                              transform=ax.transAxes, size=20)
    ax.draw_artist(title_text)
    ax.draw_artist(asset_position_text)
    ax.draw_artist(time_text)
    ax.draw_artist(time_ratio_text)
    for defender_arrow in defender_arrows:
        ax.draw_artist(defender_arrow)
    for attacker_arrow in attacker_arrows:
        ax.draw_artist(attacker_arrow)
    for asset_arrow in asset_arrows:
        ax.draw_artist(asset_arrow)
    fig.canvas.blit(ax.bbox)
    # time.sleep(GLOBAL_DT/10)

# TODO - finish intitialPositions
def initialPositions(assets, defenders, attackers):
    # asset always starts at 0,0 - default constructor state

    # defenders always start in rings around the asset
    # There is a maximum number
    number_of_rings = int(math.ceil(len(defenders)/MAX_DEFENDERS_PER_RING))
    defender_id = 0
    for ring in range(number_of_rings):
        radius = RADII_OF_RINGS[ring]
        per_ring = float(min(len(defenders) - ring*MAX_DEFENDERS_PER_RING, MAX_DEFENDERS_PER_RING))
        angles = ring*math.pi/MAX_DEFENDERS_PER_RING + \
                numpy.arange(0.0, 2*math.pi, 2*math.pi/per_ring)
        if len(angles) == 0:
            angles = [0.0]
        for j in range(len(angles)):
            defenders[defender_id].state[0] = radius*math.cos(angles[j])
            defenders[defender_id].state[1] = radius*math.sin(angles[j])
            defenders[defender_id].state[4] = math.atan2(defenders[defender_id].state[1],
                                                         defenders[defender_id].state[0])
            defender_id += 1

    # attackers always start at some uniform random polar location, fixed radius 30
    for b in attackers:
        radius = 20.0
        angle = numpy.random.uniform(0.0, 2*math.pi, 2)
        x = radius*math.cos(angle[0])
        y = radius*math.sin(angle[0])
        b.state[0] = x
        b.state[1] = y
        b.state[4] = angle[1]

    # for b in defenders:
        # radius = numpy.random.uniform(6.0, 14.0)
        # angle = numpy.random.uniform(0.0, 2*math.pi, 2)
        # x = radius*math.cos(angle[0])
        # y = radius*math.sin(angle[0])
        # b.state[0] = x
        # b.state[1] = y
        # b.state[4] = angle[1]


if __name__ == '__main__':
    # spawn boats objects

    boat_list = [Boat.Boat() for i in range(0, BOAT_COUNT)]

    # set boat types
    boat_list[0].type = "asset"
    for b in boat_list[-4:-1]:
        b.type = "attacker"

    attackers = [b for b in boat_list if b.type == "attacker"]
    defenders = [b for b in boat_list if b.type == "defender"]
    assets = [b for b in boat_list if b.type == "asset"]
    for b in boat_list:
        b.boatList = boat_list
        b.attackers = attackers
        b.defenders = defenders
        b.assets = assets

    # set initial positions
    initialPositions(assets, defenders, attackers)

    # plotSystem(boat_list, "Initial positions", 0)

    # move asset using ODE integration
    real_time_zero = time.time()
    t = 0.0
    dt = GLOBAL_DT
    step = 1
    for b in boat_list:
        if b.type == "asset":
            # asset_u0 = 0.1
            # b.state = numpy.array([0.0, 0.0, asset_u0, 0.0, 0.0, 0.0])
            # b.strategy = Strategies.DestinationOnly(b, [-5, 5], 1.0)
            # b.strategy = Strategies.ChangeHeading(b, numpy.pi/2.0)
            # b.strategy = Strategies.HoldHeading(b, 1.5)
            # b.strategy = Strategies.SingleSpline(b, [10, 10], 0.0, 1.0, 1.0)
            # b.strategy = Strategies.Square(b, 1.0, [0, 0], 20.0, "upper_right", "cw")
            # b.strategy = Strategies.Circle(b, [0.0, 0.0], 10.0)
            #b.strategy = Strategies.TimedStrategySequence(b, [
            #    (Strategies.HoldHeading, (b, 2.0)),
            #    (Strategies.ChangeHeading, (b, math.pi/2)),
            #    (Strategies.HoldHeading, (b, 2.0)),
            #    (Strategies.Square, (b, 2.0, [0, 0], 20.0, "upper_right", "cw"))
            #], [2.0, 2.0, 2.0, 15.0])
            #b.strategy = Strategies.StrategySequence(b, [
            #    (Strategies.ChangeHeading, (b, math.pi/2.0)),
            #    (Strategies.HoldHeading, (b, 2.0))
            #])

            None
        elif b.type == "defender":
            # b.strategy = Strategies.ChangeHeading(b, math.pi/2.0)
            # b.strategy = Strategies.SingleSpline(b, [assets[0].state[0], assets[0].state[1] + 10], math.pi/2.0, positionThreshold=2.0)
            #b.strategy = Strategies.StrategySequence(b, [
            #    (Strategies.DestinationOnlyExecutor, (b, [assets[0].state[0], assets[0].state[1] + 5], 1.0)),
            #    (Strategies.DestinationOnlyExecutor, (b, [assets[0].state[0], assets[0].state[1] + 10], 1.0)),
            #    (Strategies.ChangeHeading, (b, math.pi/2.0))
            #])
            #b.strategy = Strategies.DestinationOnlyExecutor(b, [assets[0].state[0], assets[0].state[1] + 10], positionThreshold=1.0)
            None
        else:
            #b.strategy = Strategies.SingleSpline(b, [assets[0].state[0], assets[0].state[1] + 10], math.pi/2.0, positionThreshold=2.0)
            # b.strategy = Strategies.PointAtAsset(b)
            # b.strategy = Strategies.PointAwayFromAsset(b)
            # b.strategy = Strategies.PointWithAsset(b)
            # b.strategy = Strategies.MoveTowardAsset(b, 1.0)
            # b.strategy = Strategies.HoldHeading(b, 1.5)
            # b.strategy = Strategies.StrategySequence(b, [Strategies.PointAtAsset(b), Strategies.HoldHeading(b, 2.0)])
            #b.strategy = Strategies.StrategySequence(b, [Strategies.PointAtAsset(b),
            #                    Strategies.DestinationOnly(b, [assets[0].state[0], b.state[1]], 1.0),
            #                    Strategies.PointAtAsset(b)])
            #b.strategy = Strategies.StrategySequence(b, [Strategies.DestinationOnly(b, [5, 5], 2.0),
            #                                             Strategies.StrategySequence(b, [Strategies.PointAtAsset(b),
            #                                                                             Strategies.HoldHeading(b, 0.5)])])
            None

    while t < TOTAL_TIME:
        times = numpy.linspace(t, t+dt, 2)
        for b in boat_list:
            b.time = t
            b.control()
            states = spi.odeint(Boat.ode, b.state, times, (b,))
            b.state = states[1]
            #if b.type == "asset":
                #print b.state[0], b.state[1], b.state[4]
        t += dt
        step += 1
        plotSystem(assets, defenders, attackers, "ATTAAAAAACK!!!!", t)
