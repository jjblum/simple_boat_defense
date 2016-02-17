import numpy
import multiprocessing as mp
import matplotlib.pyplot as plt
import scipy.integrate as spi
import pylab
import math
import time

import Boat
import Strategies

GLOBAL_DT = 0.05  # [s]
BOAT_COUNT = 30

figx = 20.0
figy = 10.0
fig = plt.figure(figsize=(figx, figy))
# ax = fig.add_subplot(111)
ax = fig.add_axes([0.01, 0.01, 0.95*figy/figx, 0.95])
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
    x_max = 1.05*max(abs(relative_x)) + mean_x
    y_max = 1.05*max(abs(relative_y)) + mean_y
    square_max = max([x_max, y_max])

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
# # asset always starts at 0,0
for b in assets:
    b.state[0] = 0.0
    b.state[1] = 0.0
# # defenders always start at some uniform random polar location, radius between 6 and 14
for b in defenders:
    radius = numpy.random.uniform(6.0, 14.0)
    angle = numpy.random.uniform(0.0, 2*math.pi, 2)
    x = radius*math.cos(angle[0])
    y = radius*math.sin(angle[0])
    b.state[0] = x
    b.state[1] = y
    b.state[4] = angle[1]
# # attackers always start at some uniform random polar location, fixed radius 30
for b in attackers:
    radius = 20.0
    angle = numpy.random.uniform(0.0, 2*math.pi, 2)
    x = radius*math.cos(angle[0])
    y = radius*math.sin(angle[0])
    b.state[0] = x
    b.state[1] = y
    b.state[4] = angle[1]

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
        # b.strategy = Strategies.HoldHeading(b, 1.5, t)
        b.strategy = Strategies.StrategySequence(b, [
            Strategies.StrategySequence(b, [
                Strategies.ChangeHeading(b, math.pi/2.0), Strategies.DestinationOnly(b, [0, 5], 1.0)
            ]),
            Strategies.StrategySequence(b, [
                Strategies.ChangeHeading(b, math.pi), Strategies.DestinationOnly(b, [-5, 5], 1.0)
            ]),
            Strategies.StrategySequence(b, [
                Strategies.ChangeHeading(b, 3.0*math.pi/2.0), Strategies.DestinationOnly(b, [-5, 0], 1.0)
            ]),
            Strategies.StrategySequence(b, [
                Strategies.ChangeHeading(b, 0.0), Strategies.DestinationOnly(b, [0, 0], 1.0)
            ])
        ])

        # TODO - try making a standalone strategy out of the the above and call it Square or something

        None
    else:
        # b.strategy = Strategies.PointAtAsset(b)
        # b.strategy = Strategies.HoldHeading(b, 1.5)
        # b.strategy = Strategies.StrategySequence(b, [Strategies.PointAtAsset(b), Strategies.HoldHeading(b, 2.0)])
        b.strategy = Strategies.StrategySequence(b, [Strategies.PointAtAsset(b),
                            Strategies.DestinationOnly(b, [assets[0].state[0], b.state[1]], 1.0),
                            Strategies.PointAtAsset(b)])
        #b.strategy = Strategies.StrategySequence(b, [Strategies.DestinationOnly(b, [5, 5], 2.0),
        #                                             Strategies.StrategySequence(b, [Strategies.PointAtAsset(b),
        #                                                                             Strategies.HoldHeading(b, 0.5)])])
        None
while t < 30:
    times = numpy.linspace(t, t+dt, 2)
    for b in boat_list:
        b.time = t
        b.control()
        states = spi.odeint(Boat.ode, b.state, times, (b,))
        b.state = states[1]
    t += dt
    step += 1
    plotSystem(assets, defenders, attackers, "My extra special run", t)
